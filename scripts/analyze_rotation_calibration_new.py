from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RGB_DIR = REPO_ROOT / "data" / "Calibration" / "rotation_calibration_new" / "rotation_calibration_sample_rgb" / "normal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "rotation_calibration_new"
DEFAULT_CURRENT_GEOMETRY = REPO_ROOT / "analysis_outputs" / "rotation_calibration" / "derotation_geometry.json"
DEFAULT_EXISTING_ANALYSIS = DEFAULT_OUTPUT_DIR / "registration_target_analysis.json"
ANGLE_SEQUENCE = tuple(range(0, 166, 5))
PREVIEW_ANGLES = (0, 45, 90, 135, 165)
LOCAL_TRANSLATION_LIMIT_PX = 48.0
COARSE_TRANSLATION_LIMIT_PX = 180.0


class SearchConfig:
    angle_subset = (0, 30, 60, 90, 120, 150)
    search_steps = (
        (8.0, 64.0),
        (4.0, 32.0),
        (2.0, 16.0),
        (1.0, 8.0),
        (0.5, 4.0),
    )
    mask_erosion_px = 6
    support_mask_erosion_px = 18
    feature_sigma = 1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the new rotation registration target sequence.")
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-geometry", type=Path, default=DEFAULT_CURRENT_GEOMETRY)
    parser.add_argument("--existing-analysis", type=Path, default=DEFAULT_EXISTING_ANALYSIS)
    parser.add_argument("--no-reuse-existing-center", action="store_true")
    return parser.parse_args()


def angle_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-2])


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb_luma(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


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


def shift_array(arr: np.ndarray, shift_xy: tuple[float, float], fill: float = 0.0) -> np.ndarray:
    shift_x, shift_y = shift_xy
    shifted = ndimage.shift(
        arr.astype(np.float32),
        shift=(float(shift_y), float(shift_x)),
        order=1,
        mode="constant",
        cval=float(fill),
        prefilter=False,
    )
    return shifted.astype(np.float32)


def shift_mask(mask: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    shift_x, shift_y = shift_xy
    shifted = ndimage.shift(
        mask.astype(np.uint8),
        shift=(float(shift_y), float(shift_x)),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return shifted > 0


def apply_transform(
    arr: np.ndarray,
    angle_deg: float,
    center_xy: tuple[float, float],
    translation_xy: tuple[float, float] = (0.0, 0.0),
    fill: float = 0.0,
) -> np.ndarray:
    rotated = rotate_array(arr, angle_deg, center_xy, fill=fill)
    return shift_array(rotated, translation_xy, fill=fill)


def apply_transform_mask(
    mask: np.ndarray,
    angle_deg: float,
    center_xy: tuple[float, float],
    translation_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    rotated = rotate_mask(mask, angle_deg, center_xy)
    return shift_mask(rotated, translation_xy)


def percentile_gray_preview(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Image.Image:
    lo, hi = np.percentile(arr, [p_low, p_high])
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def build_support_mask(reference_luma: np.ndarray, config: SearchConfig) -> np.ndarray:
    blurred = ndimage.gaussian_filter(reference_luma, sigma=6.0)
    threshold = float(np.percentile(blurred, 38.0))
    bright = blurred > threshold

    yy, xx = np.mgrid[: reference_luma.shape[0], : reference_luma.shape[1]]
    cx = reference_luma.shape[1] / 2.0
    cy = reference_luma.shape[0] / 2.0
    rx = reference_luma.shape[1] * 0.31
    ry = reference_luma.shape[0] * 0.40
    ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    mask = bright & ellipse
    mask = ndimage.binary_closing(mask, iterations=6)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        keep_label = int(np.argmax(sizes) + 1)
        mask = labels == keep_label
    mask = ndimage.binary_erosion(mask, iterations=config.support_mask_erosion_px)
    return mask.astype(bool)


def build_feature_image(luma: np.ndarray, support_mask: np.ndarray, config: SearchConfig) -> np.ndarray:
    arr = luma.astype(np.float32)
    arr = ndimage.gaussian_filter(arr, sigma=config.feature_sigma)
    grad_x = ndimage.sobel(arr, axis=1)
    grad_y = ndimage.sobel(arr, axis=0)
    feature = np.hypot(grad_x, grad_y)
    feature = np.where(support_mask, feature, 0.0)
    feature = (feature - feature[support_mask].mean()) / (feature[support_mask].std() + 1e-6)
    feature = np.where(support_mask, feature, 0.0)
    return feature.astype(np.float32)


def build_dark_feature_weight(luma: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    smooth = ndimage.gaussian_filter(luma.astype(np.float32), sigma=1.8)
    local_background = ndimage.gaussian_filter(smooth, sigma=18.0)
    darkness = np.clip(local_background - smooth, 0.0, None)
    darkness = np.where(support_mask, darkness, 0.0)
    values = darkness[support_mask]
    if values.size == 0:
        return darkness.astype(np.float32)
    lo = float(np.percentile(values, 60.0))
    hi = float(np.percentile(values, 99.5))
    scaled = np.clip((darkness - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    scaled = np.where(support_mask, scaled, 0.0)
    return scaled.astype(np.float32)


def weighted_centroid(weights: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    masked = np.where(valid_mask, weights, 0.0).astype(np.float64)
    total = float(masked.sum())
    if total <= 1e-9:
        return float(weights.shape[1] / 2.0), float(weights.shape[0] / 2.0)
    yy, xx = np.mgrid[: weights.shape[0], : weights.shape[1]]
    cx = float((masked * xx).sum() / total)
    cy = float((masked * yy).sum() / total)
    return cx, cy


def support_bbox(mask: np.ndarray, margin_px: int = 24) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(int(ys.min()) - margin_px, 0)
    y1 = min(int(ys.max()) + margin_px + 1, mask.shape[0])
    x0 = max(int(xs.min()) - margin_px, 0)
    x1 = min(int(xs.max()) + margin_px + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def clip_shift(shift_xy: tuple[float, float], limit_px: float) -> tuple[float, float]:
    shift_x, shift_y = shift_xy
    magnitude = float(np.hypot(shift_x, shift_y))
    if magnitude <= limit_px or magnitude <= 1e-9:
        return float(shift_x), float(shift_y)
    scale = limit_px / magnitude
    return float(shift_x * scale), float(shift_y * scale)


def estimate_translation_table(
    luma_stack: dict[int, np.ndarray],
    feature_stack: dict[int, np.ndarray],
    center_xy: tuple[float, float],
    support_mask: np.ndarray,
) -> dict[int, dict[str, float]]:
    frame_mask = np.ones(support_mask.shape, dtype=bool)
    bbox_y, bbox_x = support_bbox(support_mask, margin_px=18)
    reference_valid = apply_transform_mask(frame_mask, 0.0, center_xy, (0.0, 0.0)) & support_mask
    reference_dark = build_dark_feature_weight(apply_transform(luma_stack[0], 0.0, center_xy), reference_valid)
    reference_feature = apply_transform(feature_stack[0], 0.0, center_xy)
    ref_centroid_x, ref_centroid_y = weighted_centroid(reference_dark, reference_valid)

    transforms: dict[int, dict[str, float]] = {
        0: {
            "angle_deg": 0.0,
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "centroid_shift_x_px": 0.0,
            "centroid_shift_y_px": 0.0,
            "refine_shift_x_px": 0.0,
            "refine_shift_y_px": 0.0,
            "phase_corr_peak": 1.0,
        }
    }

    ref_crop = reference_feature[bbox_y, bbox_x]
    ref_valid_crop = reference_valid[bbox_y, bbox_x]
    for angle in ANGLE_SEQUENCE[1:]:
        rotated_valid = apply_transform_mask(frame_mask, float(angle), center_xy, (0.0, 0.0)) & support_mask
        rotated_luma = apply_transform(luma_stack[angle], float(angle), center_xy)
        rotated_dark = build_dark_feature_weight(rotated_luma, rotated_valid)
        current_centroid_x, current_centroid_y = weighted_centroid(rotated_dark, rotated_valid)
        coarse_shift = clip_shift(
            (ref_centroid_x - current_centroid_x, ref_centroid_y - current_centroid_y),
            COARSE_TRANSLATION_LIMIT_PX,
        )

        coarse_feature = shift_array(apply_transform(feature_stack[angle], float(angle), center_xy), coarse_shift, fill=0.0)
        coarse_valid = shift_mask(rotated_valid, coarse_shift)
        valid_crop = ref_valid_crop & coarse_valid[bbox_y, bbox_x]
        residual_dx, residual_dy, peak = phase_correlation_shift(ref_crop, coarse_feature[bbox_y, bbox_x], valid_crop)
        residual_shift = (0.0, 0.0)
        if np.isfinite(residual_dx) and np.isfinite(residual_dy) and float(peak) >= 0.01:
            residual_shift = clip_shift((residual_dx, residual_dy), LOCAL_TRANSLATION_LIMIT_PX)
        final_shift = (coarse_shift[0] + residual_shift[0], coarse_shift[1] + residual_shift[1])
        transforms[angle] = {
            "angle_deg": float(angle),
            "shift_x_px": float(final_shift[0]),
            "shift_y_px": float(final_shift[1]),
            "centroid_shift_x_px": float(coarse_shift[0]),
            "centroid_shift_y_px": float(coarse_shift[1]),
            "refine_shift_x_px": float(residual_shift[0]),
            "refine_shift_y_px": float(residual_shift[1]),
            "phase_corr_peak": float(peak),
        }
    return transforms


def score_center(
    feature_stack: dict[int, np.ndarray],
    center_xy: tuple[float, float],
    support_mask: np.ndarray,
    erosion_px: int,
) -> dict[str, float]:
    valid = support_mask.copy()
    stack = []
    frame_mask = np.ones(support_mask.shape, dtype=bool)
    for angle in sorted(feature_stack):
        stack.append(rotate_array(feature_stack[angle], angle, center_xy))
        valid &= rotate_mask(frame_mask, angle, center_xy)
    if erosion_px > 0:
        valid = ndimage.binary_erosion(valid, iterations=erosion_px)
    area_fraction = float(valid.mean())
    if area_fraction <= 0.02:
        return {"score": -1e9, "area_fraction": area_fraction, "mean_variance": 1e9, "corr_like": 0.0}
    stack_arr = np.stack(stack, axis=0)
    valid_pixels = stack_arr[:, valid]
    valid_pixels = valid_pixels - valid_pixels.mean(axis=1, keepdims=True)
    valid_pixels = valid_pixels / (valid_pixels.std(axis=1, keepdims=True) + 1e-6)
    mean_variance = float(valid_pixels.var(axis=0).mean())
    corr_like = float(1.0 - mean_variance / 2.0)
    score = float(-mean_variance + 0.35 * area_fraction)
    return {"score": score, "area_fraction": area_fraction, "mean_variance": mean_variance, "corr_like": corr_like}


def search_rotation_center(feature_stack: dict[int, np.ndarray], support_mask: np.ndarray, config: SearchConfig) -> dict[str, object]:
    sample = next(iter(feature_stack.values()))
    center = np.array([sample.shape[1] / 2.0, sample.shape[0] / 2.0], dtype=np.float64)
    history = []
    for step, radius in config.search_steps:
        best = None
        xs = np.arange(center[0] - radius, center[0] + radius + 1e-6, step)
        ys = np.arange(center[1] - radius, center[1] + radius + 1e-6, step)
        for y in ys:
            for x in xs:
                metrics = score_center(feature_stack, (float(x), float(y)), support_mask, config.mask_erosion_px)
                if best is None or metrics["score"] > best["score"]:
                    best = {"center_xy_fullres": [float(x), float(y)], **metrics}
        assert best is not None
        center = np.array(best["center_xy_fullres"], dtype=np.float64)
        history.append({"step_px": step, "radius_px": radius, **best})
    return {"center_xy_fullres": history[-1]["center_xy_fullres"], "search_history": history, "final_metrics": history[-1]}


def phase_correlation_shift(reference: np.ndarray, moving: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float, float]:
    if not np.any(valid_mask):
        return float("nan"), float("nan"), float("nan")
    ref = np.where(valid_mask, reference, np.nan)
    mov = np.where(valid_mask, moving, np.nan)
    ref = np.where(valid_mask, ref - np.nanmean(ref), 0.0)
    mov = np.where(valid_mask, mov - np.nanmean(mov), 0.0)
    window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float64)
    ref *= window
    mov *= window
    ref_fft = np.fft.fft2(ref)
    mov_fft = np.fft.fft2(mov)
    cross_power = ref_fft * np.conj(mov_fft)
    magnitude = np.abs(cross_power)
    if not np.any(magnitude > 0):
        return float("nan"), float("nan"), float("nan")
    cross_power /= np.maximum(magnitude, 1e-12)
    corr = np.fft.ifft2(cross_power)
    corr_abs = np.abs(corr)
    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    shifts = np.array(peak, dtype=np.float64)
    shape = np.array(corr_abs.shape, dtype=np.float64)
    shifts = np.where(shifts > shape / 2.0, shifts - shape, shifts)
    dy, dx = shifts
    return float(dx), float(dy), float(corr_abs[peak])


def compute_alignment_metrics(
    luma_stack: dict[int, np.ndarray],
    feature_stack: dict[int, np.ndarray],
    center_xy: tuple[float, float],
    support_mask: np.ndarray,
    translation_table: dict[int, dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    frame_mask = np.ones(support_mask.shape, dtype=bool)
    valid_mask = support_mask.copy()
    for angle in ANGLE_SEQUENCE:
        if translation_table is None:
            shift_xy = (0.0, 0.0)
        else:
            transform = translation_table[int(angle)]
            shift_xy = (float(transform["shift_x_px"]), float(transform["shift_y_px"]))
        valid_mask &= apply_transform_mask(frame_mask, angle, center_xy, shift_xy)
    valid_mask = ndimage.binary_erosion(valid_mask, iterations=4)
    bbox_y, bbox_x = support_bbox(valid_mask, margin_px=8)
    valid_crop = valid_mask[bbox_y, bbox_x]

    reference_shift = (0.0, 0.0) if translation_table is None else (
        float(translation_table[0]["shift_x_px"]),
        float(translation_table[0]["shift_y_px"]),
    )
    reference = apply_transform(feature_stack[0], 0.0, center_xy, reference_shift)[bbox_y, bbox_x]
    metrics = []
    for angle in ANGLE_SEQUENCE:
        if translation_table is None:
            shift_xy = (0.0, 0.0)
        else:
            transform = translation_table[int(angle)]
            shift_xy = (float(transform["shift_x_px"]), float(transform["shift_y_px"]))
        current = apply_transform(feature_stack[angle], float(angle), center_xy, shift_xy)[bbox_y, bbox_x]
        ref_vals = reference[valid_crop]
        cur_vals = current[valid_crop]
        ref_vals = (ref_vals - ref_vals.mean()) / (ref_vals.std() + 1e-6)
        cur_vals = (cur_vals - cur_vals.mean()) / (cur_vals.std() + 1e-6)
        corr = float(np.mean(ref_vals * cur_vals))
        rmse = float(np.sqrt(np.mean((ref_vals - cur_vals) ** 2)))
        shift_x, shift_y, peak = phase_correlation_shift(reference, current, valid_crop)
        metrics.append(
            {
                "angle_deg": float(angle),
                "corr_to_0deg": corr,
                "rmse_to_0deg": rmse,
                "shift_x_px": shift_x,
                "shift_y_px": shift_y,
                "shift_magnitude_px": float(np.hypot(shift_x, shift_y)),
                "corr_peak": peak,
            }
        )
    return metrics


def build_common_valid_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    support_mask: np.ndarray,
    translation_table: dict[int, dict[str, float]] | None = None,
) -> np.ndarray:
    frame_mask = np.ones(shape, dtype=bool)
    valid_mask = support_mask.copy()
    for angle in ANGLE_SEQUENCE:
        if translation_table is None:
            shift_xy = (0.0, 0.0)
        else:
            transform = translation_table[int(angle)]
            shift_xy = (float(transform["shift_x_px"]), float(transform["shift_y_px"]))
        valid_mask &= apply_transform_mask(frame_mask, angle, center_xy, shift_xy)
    valid_mask = ndimage.binary_erosion(valid_mask, iterations=4)
    return valid_mask


def draw_overlay(
    base_luma: np.ndarray,
    support_mask: np.ndarray,
    valid_mask: np.ndarray,
    center_xy: tuple[float, float],
) -> Image.Image:
    base = percentile_gray_preview(base_luma).convert("RGBA")
    rgba = np.asarray(base, dtype=np.uint8).copy()
    outside = ~support_mask
    rgba[outside, 0] = np.clip(rgba[outside, 0] * 0.4 + 120, 0, 255).astype(np.uint8)
    rgba[outside, 1] = (rgba[outside, 1] * 0.4).astype(np.uint8)
    rgba[outside, 2] = (rgba[outside, 2] * 0.4).astype(np.uint8)
    overlay = Image.fromarray(rgba, mode="RGBA")
    support_outline = ndimage.binary_dilation(support_mask, iterations=2) ^ support_mask
    valid_outline = ndimage.binary_dilation(valid_mask, iterations=2) ^ valid_mask
    mask_rgba = Image.new("RGBA", overlay.size, (255, 165, 0, 0))
    mask_rgba.putalpha(Image.fromarray((support_outline.astype(np.uint8) * 255), mode="L"))
    overlay = Image.alpha_composite(overlay, mask_rgba)
    valid_rgba = Image.new("RGBA", overlay.size, (0, 255, 128, 0))
    valid_rgba.putalpha(Image.fromarray((valid_outline.astype(np.uint8) * 255), mode="L"))
    overlay = Image.alpha_composite(overlay, valid_rgba)
    draw = ImageDraw.Draw(overlay)
    cx, cy = center_xy
    draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=ImageColor.getrgb("#ffea00"), width=4)
    draw.line((cx - 18, cy, cx + 18, cy), fill="#ffea00", width=3)
    draw.line((cx, cy - 18, cx, cy + 18), fill="#ffea00", width=3)
    return overlay


def save_derotated_preview(
    rgb_dir: Path,
    center_xy: tuple[float, float],
    support_mask: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    title_prefix: str,
    translation_table: dict[int, dict[str, float]] | None = None,
) -> None:
    tiles = []
    for angle in PREVIEW_ANGLES:
        rgb_path = rgb_dir / f"rotation_calibration_sample_normal_{angle:03d}_rgb.tif"
        luma = load_rgb_luma(rgb_path)
        if translation_table is None:
            shift_xy = (0.0, 0.0)
        else:
            transform = translation_table[int(angle)]
            shift_xy = (float(transform["shift_x_px"]), float(transform["shift_y_px"]))
        derot = apply_transform(luma, float(angle), center_xy, shift_xy)
        angle_support = apply_transform_mask(support_mask, float(angle), center_xy, shift_xy)
        adjusted_center = (center_xy[0] + shift_xy[0], center_xy[1] + shift_xy[1])
        overlay = draw_overlay(derot, angle_support, valid_mask, adjusted_center).convert("RGB")
        tile = overlay.resize((620, 329), resample=Image.Resampling.BILINEAR)
        label_band = Image.new("RGB", (tile.width, 38), color="#111111")
        draw = ImageDraw.Draw(label_band)
        label = f"{title_prefix} {angle:03d} deg"
        if translation_table is not None:
            label += f"  dx={shift_xy[0]:+.1f} dy={shift_xy[1]:+.1f}"
        draw.text((16, 10), label, fill="white")
        combined = Image.new("RGB", (tile.width, tile.height + label_band.height), color="black")
        combined.paste(label_band, (0, 0))
        combined.paste(tile, (0, label_band.height))
        tiles.append(combined)
    cols = 2
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (tiles[0].width * cols, tiles[0].height * rows), color="#050505")
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile.width
        y = (index // cols) * tile.height
        canvas.paste(tile, (x, y))
    canvas.save(output_path)


def save_metrics_plot(series_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_series = next(iter(series_by_label.values()))
    angles = [row["angle_deg"] for row in first_series]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, field, title in (
        (axes[0, 0], "corr_to_0deg", "Feature corr to 0 deg"),
        (axes[0, 1], "rmse_to_0deg", "Feature RMSE to 0 deg"),
        (axes[1, 0], "shift_magnitude_px", "Phase-correlation shift magnitude"),
        (axes[1, 1], "corr_peak", "Phase-correlation peak"),
    ):
        for label, metrics in series_by_label.items():
            marker = "o" if "current" in label else "s" if "translation" not in label else "^"
            axis.plot(angles, [row[field] for row in metrics], marker=marker, label=label)
        axis.set_title(title)
        axis.set_xlabel("theta (deg)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Rotation Calibration New: Alignment Comparison", fontsize=14)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        "median_corr_to_0deg": float(np.median([row["corr_to_0deg"] for row in metrics])),
        "median_rmse_to_0deg": float(np.median([row["rmse_to_0deg"] for row in metrics])),
        "median_shift_magnitude_px": float(np.median([row["shift_magnitude_px"] for row in metrics if row["angle_deg"] != 0])),
        "median_corr_peak": float(np.median([row["corr_peak"] for row in metrics if row["angle_deg"] != 0])),
    }


def translation_summary(translation_table: dict[int, dict[str, float]]) -> dict[str, float]:
    nonzero = [row for angle, row in translation_table.items() if int(angle) != 0]
    magnitudes = [float(np.hypot(row["shift_x_px"], row["shift_y_px"])) for row in nonzero]
    coarse_magnitudes = [float(np.hypot(row["centroid_shift_x_px"], row["centroid_shift_y_px"])) for row in nonzero]
    refine_magnitudes = [float(np.hypot(row["refine_shift_x_px"], row["refine_shift_y_px"])) for row in nonzero]
    return {
        "median_shift_magnitude_px": float(np.median(magnitudes)),
        "max_shift_magnitude_px": float(np.max(magnitudes)),
        "median_centroid_shift_magnitude_px": float(np.median(coarse_magnitudes)),
        "median_refine_shift_magnitude_px": float(np.median(refine_magnitudes)),
        "median_phase_corr_peak": float(np.median([row["phase_corr_peak"] for row in nonzero])),
    }


def save_translation_csv(translation_table: dict[int, dict[str, float]], output_path: Path) -> None:
    import csv

    fieldnames = [
        "angle_deg",
        "shift_x_px",
        "shift_y_px",
        "centroid_shift_x_px",
        "centroid_shift_y_px",
        "refine_shift_x_px",
        "refine_shift_y_px",
        "phase_corr_peak",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for angle in ANGLE_SEQUENCE:
            writer.writerow(translation_table[int(angle)])


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    current_geometry = load_json(args.current_geometry)
    current_center = (
        float(current_geometry["center_xy_fullres"]["x"]),
        float(current_geometry["center_xy_fullres"]["y"]),
    )

    reference_luma = load_rgb_luma(args.rgb_dir / "rotation_calibration_sample_normal_000_rgb.tif")
    support_mask = build_support_mask(reference_luma, SearchConfig)

    luma_stack = {angle: load_rgb_luma(args.rgb_dir / f"rotation_calibration_sample_normal_{angle:03d}_rgb.tif") for angle in ANGLE_SEQUENCE}
    feature_stack = {
        angle: build_feature_image(luma_stack[angle], support_mask, SearchConfig)
        for angle in SearchConfig.angle_subset
    }

    existing_search_history: list[dict[str, object]] = []
    existing_center: tuple[float, float] | None = None
    if not args.no_reuse_existing_center and args.existing_analysis.exists():
        existing = load_json(args.existing_analysis)
        if isinstance(existing, dict) and "new_center_xy_fullres" in existing:
            existing_center = (
                float(existing["new_center_xy_fullres"]["x"]),
                float(existing["new_center_xy_fullres"]["y"]),
            )
            existing_search_history = list(existing.get("search_history", []))

    if existing_center is not None:
        search_result = {
            "center_xy_fullres": [existing_center[0], existing_center[1]],
            "search_history": existing_search_history,
            "final_metrics": existing_search_history[-1] if existing_search_history else {},
            "reused_existing_center": True,
        }
        new_center = existing_center
    else:
        search_result = search_rotation_center(feature_stack, support_mask, SearchConfig)
        search_result["reused_existing_center"] = False
        new_center = tuple(float(v) for v in search_result["center_xy_fullres"])

    full_feature_stack = {
        angle: build_feature_image(luma_stack[angle], support_mask, SearchConfig)
        for angle in ANGLE_SEQUENCE
    }
    translation_table = estimate_translation_table(luma_stack, full_feature_stack, new_center, support_mask)
    new_valid_mask = build_common_valid_mask(reference_luma.shape, new_center, support_mask)
    rt_valid_mask = build_common_valid_mask(reference_luma.shape, new_center, support_mask, translation_table)
    old_valid_mask = build_common_valid_mask(reference_luma.shape, current_center, support_mask)
    new_metrics = compute_alignment_metrics(luma_stack, full_feature_stack, new_center, support_mask)
    rt_metrics = compute_alignment_metrics(luma_stack, full_feature_stack, new_center, support_mask, translation_table)
    old_metrics = compute_alignment_metrics(luma_stack, full_feature_stack, current_center, support_mask)

    support_mask_path = args.output_dir / "registration_target_support_mask.png"
    Image.fromarray((support_mask.astype(np.uint8) * 255), mode="L").save(support_mask_path)
    Image.fromarray((new_valid_mask.astype(np.uint8) * 255), mode="L").save(args.output_dir / "registration_target_valid_mask_new.png")
    Image.fromarray((rt_valid_mask.astype(np.uint8) * 255), mode="L").save(args.output_dir / "registration_target_valid_mask_new_translation.png")
    Image.fromarray((old_valid_mask.astype(np.uint8) * 255), mode="L").save(args.output_dir / "registration_target_valid_mask_current.png")

    draw_overlay(reference_luma, support_mask, new_valid_mask, new_center).save(args.output_dir / "registration_target_overlay_new.png")
    draw_overlay(reference_luma, support_mask, old_valid_mask, current_center).save(args.output_dir / "registration_target_overlay_current.png")
    save_derotated_preview(args.rgb_dir, new_center, support_mask, new_valid_mask, args.output_dir / "registration_target_derotated_preview_new.png", "new center")
    save_derotated_preview(
        args.rgb_dir,
        new_center,
        support_mask,
        rt_valid_mask,
        args.output_dir / "registration_target_derotated_preview_new_translation.png",
        "new center + translation",
        translation_table=translation_table,
    )
    save_derotated_preview(args.rgb_dir, current_center, support_mask, old_valid_mask, args.output_dir / "registration_target_derotated_preview_current.png", "current center")
    save_metrics_plot(
        {
            "current center": old_metrics,
            "new center": new_metrics,
            "new center + translation": rt_metrics,
        },
        args.output_dir / "registration_target_metrics_comparison.png",
    )
    save_translation_csv(translation_table, args.output_dir / "registration_target_translation_table.csv")
    correction_spec = {
        "source": "rotation_calibration_new",
        "rgb_dir": str(args.rgb_dir),
        "reference_angle_deg": 0.0,
        "rotation_center_xy_fullres": {"x": new_center[0], "y": new_center[1]},
        "transform_order": [
            "rotate_by_minus_theta_about_center",
            "apply_post_rotation_translation_dx_dy",
        ],
        "translation_table": [translation_table[int(angle)] for angle in ANGLE_SEQUENCE],
    }
    correction_spec_path = args.output_dir / "rotation_translation_correction.json"
    save_json(correction_spec_path, correction_spec)

    current_summary = summarize_metrics(old_metrics)
    new_summary = summarize_metrics(new_metrics)
    rt_summary = summarize_metrics(rt_metrics)
    result = {
        "rgb_dir": str(args.rgb_dir),
        "current_center_xy_fullres": {"x": current_center[0], "y": current_center[1]},
        "new_center_xy_fullres": {"x": new_center[0], "y": new_center[1]},
        "search_history": search_result["search_history"],
        "support_mask_area_fraction": float(support_mask.mean()),
        "current_metrics_summary": current_summary,
        "new_metrics_summary": new_summary,
        "rotation_translation_metrics_summary": rt_summary,
        "rotation_translation_shift_summary": translation_summary(translation_table),
        "improvement": {
            "new_vs_current_corr_delta": new_summary["median_corr_to_0deg"] - current_summary["median_corr_to_0deg"],
            "new_vs_current_rmse_delta": new_summary["median_rmse_to_0deg"] - current_summary["median_rmse_to_0deg"],
            "new_vs_current_shift_delta_px": new_summary["median_shift_magnitude_px"] - current_summary["median_shift_magnitude_px"],
            "rotation_translation_vs_new_corr_delta": rt_summary["median_corr_to_0deg"] - new_summary["median_corr_to_0deg"],
            "rotation_translation_vs_new_rmse_delta": rt_summary["median_rmse_to_0deg"] - new_summary["median_rmse_to_0deg"],
            "rotation_translation_vs_new_shift_delta_px": rt_summary["median_shift_magnitude_px"] - new_summary["median_shift_magnitude_px"],
        },
        "new_alignment_metrics": new_metrics,
        "rotation_translation_alignment_metrics": rt_metrics,
        "current_alignment_metrics": old_metrics,
        "rotation_translation_table": [translation_table[int(angle)] for angle in ANGLE_SEQUENCE],
        "output_files": {
            "support_mask": str(support_mask_path),
            "overlay_new": str(args.output_dir / "registration_target_overlay_new.png"),
            "overlay_current": str(args.output_dir / "registration_target_overlay_current.png"),
            "valid_mask_new": str(args.output_dir / "registration_target_valid_mask_new.png"),
            "valid_mask_new_translation": str(args.output_dir / "registration_target_valid_mask_new_translation.png"),
            "valid_mask_current": str(args.output_dir / "registration_target_valid_mask_current.png"),
            "preview_new": str(args.output_dir / "registration_target_derotated_preview_new.png"),
            "preview_new_translation": str(args.output_dir / "registration_target_derotated_preview_new_translation.png"),
            "preview_current": str(args.output_dir / "registration_target_derotated_preview_current.png"),
            "metrics_plot": str(args.output_dir / "registration_target_metrics_comparison.png"),
            "translation_table_csv": str(args.output_dir / "registration_target_translation_table.csv"),
            "correction_spec": str(correction_spec_path),
        },
    }
    save_json(args.output_dir / "registration_target_analysis.json", result)
    print(json.dumps(result["new_center_xy_fullres"], indent=2))
    print(json.dumps(result["rotation_translation_metrics_summary"], indent=2))


if __name__ == "__main__":
    main()
