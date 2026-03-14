from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageOps
from scipy import ndimage


DEFAULT_ROTATION_DIR = Path("data/Calibration/rotation_calibration/normal")
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/rotation_calibration")


@dataclass(frozen=True)
class SearchConfig:
    bayer_bin: int = 2
    search_downsample: int = 4
    angle_subset: tuple[int, ...] = (0, 30, 60, 90, 120, 150)
    search_steps: tuple[tuple[float, float], ...] = (
        (8.0, 32.0),
        (4.0, 16.0),
        (2.0, 8.0),
        (1.0, 4.0),
        (0.5, 2.0),
    )
    mask_erosion_px: int = 2

    @property
    def fullres_per_search_pixel(self) -> float:
        return float(self.bayer_bin * self.search_downsample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the rotation center and full-survival pixel area from rotation_calibration."
    )
    parser.add_argument(
        "--rotation-dir",
        type=Path,
        default=DEFAULT_ROTATION_DIR,
        help="Directory containing rotation_calibration normal TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON and preview images will be written.",
    )
    return parser.parse_args()


def angle_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_raw16(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32)


def bin_2x2(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    height = (arr.shape[0] // factor) * factor
    width = (arr.shape[1] // factor) * factor
    trimmed = arr[:height, :width]
    return trimmed.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def downsample_image(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = image.resize(
        (arr.shape[1] // factor, arr.shape[0] // factor),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def build_feature_image(raw: np.ndarray, config: SearchConfig) -> np.ndarray:
    arr = bin_2x2(raw, config.bayer_bin)
    arr = arr / max(float(arr.max()), 1.0)
    arr = ndimage.gaussian_filter(arr, sigma=1.0)
    grad_x = ndimage.sobel(arr, axis=1)
    grad_y = ndimage.sobel(arr, axis=0)
    feature = np.hypot(grad_x, grad_y)
    feature = downsample_image(feature, config.search_downsample)
    feature = (feature - feature.mean()) / (feature.std() + 1e-6)
    return feature.astype(np.float32, copy=False)


def rotate_array(arr: np.ndarray, angle_deg: float, center_xy: tuple[float, float], fill: float = 0.0) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    rotated = image.rotate(
        -angle_deg,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        center=center_xy,
        fillcolor=float(fill),
    )
    return np.asarray(rotated, dtype=np.float32)


def rotate_mask(mask: np.ndarray, angle_deg: float, center_xy: tuple[float, float]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    rotated = image.rotate(
        -angle_deg,
        resample=Image.Resampling.NEAREST,
        expand=False,
        center=center_xy,
        fillcolor=0,
    )
    return np.asarray(rotated, dtype=np.uint8) > 250


def build_search_stack(rotation_dir: Path, config: SearchConfig) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    raw_paths = sorted(rotation_dir.glob("*.tif"), key=angle_from_path)
    if not raw_paths:
        raise FileNotFoundError(f"No TIFF files found in {rotation_dir}")

    features: dict[int, np.ndarray] = {}
    full_size = None
    for path in raw_paths:
        angle = angle_from_path(path)
        if angle not in config.angle_subset:
            continue
        raw = load_raw16(path)
        if full_size is None:
            full_size = (raw.shape[1], raw.shape[0])
        features[angle] = build_feature_image(raw, config)

    missing = sorted(set(config.angle_subset) - set(features))
    if missing:
        raise ValueError(f"Missing expected angles for center search: {missing}")

    assert full_size is not None
    return features, full_size


def score_center(
    feature_stack: dict[int, np.ndarray],
    center_xy: tuple[float, float],
    erosion_px: int,
) -> dict[str, float]:
    angles = sorted(feature_stack)
    sample = feature_stack[angles[0]]
    valid = np.ones(sample.shape, dtype=bool)
    stack = []
    all_ones = np.ones(sample.shape, dtype=bool)

    for angle in angles:
        stack.append(rotate_array(feature_stack[angle], angle, center_xy))
        valid &= rotate_mask(all_ones, angle, center_xy)

    if erosion_px > 0:
        valid = ndimage.binary_erosion(valid, iterations=erosion_px)

    area_fraction = float(valid.mean())
    if area_fraction <= 0.05:
        return {
            "score": -1e9,
            "area_fraction": area_fraction,
            "corr_like": 0.0,
            "mean_variance": 1e9,
        }

    stack_arr = np.stack(stack, axis=0)
    valid_pixels = stack_arr[:, valid]
    valid_pixels = valid_pixels - valid_pixels.mean(axis=1, keepdims=True)
    valid_pixels = valid_pixels / (valid_pixels.std(axis=1, keepdims=True) + 1e-6)
    mean_variance = float(valid_pixels.var(axis=0).mean())
    corr_like = float(1.0 - mean_variance / 2.0)
    score = float(-mean_variance + 0.5 * area_fraction)
    return {
        "score": score,
        "area_fraction": area_fraction,
        "corr_like": corr_like,
        "mean_variance": mean_variance,
    }


def search_rotation_center(feature_stack: dict[int, np.ndarray], config: SearchConfig) -> dict[str, object]:
    sample = next(iter(feature_stack.values()))
    center = np.array([sample.shape[1] / 2.0, sample.shape[0] / 2.0], dtype=np.float64)
    history = []

    for step, radius in config.search_steps:
        best = None
        xs = np.arange(center[0] - radius, center[0] + radius + 1e-6, step)
        ys = np.arange(center[1] - radius, center[1] + radius + 1e-6, step)
        for y in ys:
            for x in xs:
                metrics = score_center(feature_stack, (float(x), float(y)), config.mask_erosion_px)
                if best is None or metrics["score"] > best["score"]:
                    best = {
                        "center_xy_search": [float(x), float(y)],
                        **metrics,
                    }
        assert best is not None
        center = np.array(best["center_xy_search"], dtype=np.float64)
        history.append(
            {
                "step_search_px": step,
                "radius_search_px": radius,
                **best,
            }
        )

    center_fullres = (center * config.fullres_per_search_pixel).tolist()
    return {
        "center_xy_search": center.tolist(),
        "center_xy_fullres": center_fullres,
        "search_history": history,
        "final_metrics": history[-1],
    }


def build_full_valid_mask(image_size: tuple[int, int], center_xy: tuple[float, float], angles: list[int]) -> np.ndarray:
    width, height = image_size
    mask = np.ones((height, width), dtype=bool)
    all_ones = np.ones((height, width), dtype=bool)
    for angle in angles:
        mask &= rotate_mask(all_ones, angle, center_xy)
    return mask


def largest_rectangle_in_mask(mask: np.ndarray) -> dict[str, int]:
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area = 0
    best_rect = {"x": 0, "y": 0, "width": 0, "height": 0, "area": 0}

    for y in range(mask.shape[0]):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(mask.shape[1] + 1):
            current_height = int(heights[x]) if x < mask.shape[1] else 0
            start = x
            while stack and stack[-1][1] > current_height:
                start, height = stack.pop()
                area = height * (x - start)
                if area > best_area:
                    best_area = area
                    best_rect = {
                        "x": int(start),
                        "y": int(y - height + 1),
                        "width": int(x - start),
                        "height": int(height),
                        "area": int(area),
                    }
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))
    return best_rect


def mask_bbox(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.where(mask)
    return {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def percentile_preview(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> Image.Image:
    lo, hi = np.percentile(arr, [p_low, p_high])
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def draw_geometry_overlay(
    base_gray: np.ndarray,
    full_mask: np.ndarray,
    largest_rect: dict[str, int],
    center_xy: tuple[float, float],
) -> Image.Image:
    base = percentile_preview(base_gray).convert("RGBA")

    rgba = np.asarray(base, dtype=np.uint8).copy()
    outside = ~full_mask
    rgba[outside, 0] = np.clip(rgba[outside, 0] * 0.35 + 160, 0, 255).astype(np.uint8)
    rgba[outside, 1] = (rgba[outside, 1] * 0.35).astype(np.uint8)
    rgba[outside, 2] = (rgba[outside, 2] * 0.35).astype(np.uint8)
    overlay = Image.fromarray(rgba, mode="RGBA")

    outline_mask = ndimage.binary_dilation(full_mask, iterations=2) ^ full_mask
    outline = Image.fromarray((outline_mask.astype(np.uint8) * 255), mode="L")
    outline_rgba = Image.new("RGBA", overlay.size, (0, 255, 0, 0))
    outline_rgba.putalpha(outline)
    overlay = Image.alpha_composite(overlay, outline_rgba)

    draw = ImageDraw.Draw(overlay)
    rect = (
        largest_rect["x"],
        largest_rect["y"],
        largest_rect["x"] + largest_rect["width"] - 1,
        largest_rect["y"] + largest_rect["height"] - 1,
    )
    draw.rectangle(rect, outline=ImageColor.getrgb("#00e676"), width=4)
    cx, cy = center_xy
    radius = 10
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="#ffea00", width=4)
    draw.line((cx - 18, cy, cx + 18, cy), fill="#ffea00", width=3)
    draw.line((cx, cy - 18, cx, cy + 18), fill="#ffea00", width=3)
    return overlay


def make_derotated_preview(
    rotation_dir: Path,
    center_xy: tuple[float, float],
    largest_rect: dict[str, int],
    full_mask: np.ndarray,
    output_path: Path,
) -> None:
    preview_angles = (0, 45, 90, 135)
    tiles = []
    for angle in preview_angles:
        path = rotation_dir / f"rotation_calibration_normal_{angle:03d}.tif"
        raw = load_raw16(path)
        derot = rotate_array(raw, angle, center_xy)
        overlay = draw_geometry_overlay(derot, full_mask, largest_rect, center_xy).convert("RGB")
        tile = overlay.resize((700, 372), resample=Image.Resampling.BILINEAR)
        label_band = Image.new("RGB", (tile.width, 38), color="#111111")
        draw = ImageDraw.Draw(label_band)
        draw.text((18, 10), f"{angle:03d} deg derotated", fill="white")
        combined = Image.new("RGB", (tile.width, tile.height + label_band.height), color="black")
        combined.paste(label_band, (0, 0))
        combined.paste(tile, (0, label_band.height))
        tiles.append(combined)

    cols = 2
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (tiles[0].width * cols, tiles[0].height * rows), color="#050505")
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile.width
        y = (idx // cols) * tile.height
        canvas.paste(tile, (x, y))
    canvas.save(output_path)


def compute_alignment_metrics(
    rotation_dir: Path,
    center_xy: tuple[float, float],
    valid_mask: np.ndarray,
) -> list[dict[str, float]]:
    reference = load_raw16(rotation_dir / "rotation_calibration_normal_000.tif")
    reference_derot = rotate_array(reference, 0.0, center_xy)
    ref_vals = reference_derot[valid_mask]
    ref_vals = (ref_vals - ref_vals.mean()) / (ref_vals.std() + 1e-6)

    metrics = []
    for angle in range(0, 166, 5):
        current = load_raw16(rotation_dir / f"rotation_calibration_normal_{angle:03d}.tif")
        current_derot = rotate_array(current, float(angle), center_xy)
        vals = current_derot[valid_mask]
        vals = (vals - vals.mean()) / (vals.std() + 1e-6)
        corr = float(np.mean(ref_vals * vals))
        rmse = float(np.sqrt(np.mean((ref_vals - vals) ** 2)))
        metrics.append(
            {
                "angle_deg": float(angle),
                "corr_to_0deg": corr,
                "rmse_to_0deg": rmse,
            }
        )
    return metrics


def save_json(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SearchConfig()
    feature_stack, image_size = build_search_stack(args.rotation_dir, config)
    search_result = search_rotation_center(feature_stack, config)

    center_fullres = tuple(float(v) for v in search_result["center_xy_fullres"])
    all_angles = list(range(0, 166, 5))
    full_mask = build_full_valid_mask(image_size, center_fullres, all_angles)
    full_mask = ndimage.binary_erosion(full_mask, iterations=4)

    rect = largest_rectangle_in_mask(full_mask)
    bbox = mask_bbox(full_mask)
    mask_area_fraction = float(full_mask.mean())
    mask_area_px = int(full_mask.sum())
    alignment_metrics = compute_alignment_metrics(args.rotation_dir, center_fullres, full_mask)

    reference = load_raw16(args.rotation_dir / "rotation_calibration_normal_000.tif")
    overlay = draw_geometry_overlay(reference, full_mask, rect, center_fullres)
    overlay.save(output_dir / "rotation_calibration_geometry_overlay.png")
    Image.fromarray((full_mask.astype(np.uint8) * 255), mode="L").save(output_dir / "rotation_calibration_valid_mask.png")
    make_derotated_preview(
        args.rotation_dir,
        center_fullres,
        rect,
        full_mask,
        output_dir / "rotation_calibration_derotated_preview.png",
    )

    geometry = {
        "rotation_dir": str(args.rotation_dir),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "search_config": {
            "bayer_bin": config.bayer_bin,
            "search_downsample": config.search_downsample,
            "angle_subset": list(config.angle_subset),
            "search_steps": [
                {"step_search_px": step, "radius_search_px": radius}
                for step, radius in config.search_steps
            ],
        },
        "center_xy_fullres": {"x": center_fullres[0], "y": center_fullres[1]},
        "center_xy_search": {
            "x": float(search_result["center_xy_search"][0]),
            "y": float(search_result["center_xy_search"][1]),
        },
        "search_history": search_result["search_history"],
        "common_valid_mask": {
            "area_fraction": mask_area_fraction,
            "area_px": mask_area_px,
            "bbox": bbox,
        },
        "largest_axis_aligned_rectangle": rect,
        "alignment_metrics": alignment_metrics,
        "output_files": {
            "geometry_overlay": str(output_dir / "rotation_calibration_geometry_overlay.png"),
            "valid_mask": str(output_dir / "rotation_calibration_valid_mask.png"),
            "derotated_preview": str(output_dir / "rotation_calibration_derotated_preview.png"),
        },
    }
    save_json(geometry, output_dir / "derotation_geometry.json")

    print(json.dumps(geometry["center_xy_fullres"], indent=2))
    print(json.dumps(geometry["common_valid_mask"], indent=2))
    print(json.dumps(geometry["largest_axis_aligned_rectangle"], indent=2))


if __name__ == "__main__":
    main()
