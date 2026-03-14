from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw


DEFAULT_ROI_DIR = Path("data/ROI_selection")
DEFAULT_GEOMETRY_JSON = Path("analysis_outputs/rotation_calibration/derotation_geometry.json")
DEFAULT_MASK_PATH = Path("analysis_outputs/rotation_calibration/rotation_calibration_valid_mask.png")
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/roi_review/safe_region_baseline")


SAMPLE_RE = re.compile(r"_s(\d+)_", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotated ROI review images using the derotation-safe mask."
    )
    parser.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR)
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_JSON)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--square-sizes",
        type=int,
        nargs="*",
        default=[500, 400, 300],
        help="Centered square ROI sizes to overlay inside the safe rectangle.",
    )
    return parser.parse_args()


def percentile_stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.uint8)
    for channel in range(rgb.shape[2]):
        plane = rgb[:, :, channel].astype(np.float32)
        lo, hi = np.percentile(plane, [1.0, 99.0])
        scaled = np.clip((plane - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[:, :, channel] = (scaled * 255).astype(np.uint8)
    return out


def annotate_image(
    image_path: Path,
    safe_rect: dict[str, int],
    center_xy: dict[str, float],
    valid_mask: np.ndarray,
    square_rect: dict[str, int] | None = None,
    span_rects: dict[str, dict[str, int]] | None = None,
    labeled_rects: list[dict[str, object]] | None = None,
) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    stretched = percentile_stretch_rgb(rgb)
    rgba = np.dstack([stretched, np.full(stretched.shape[:2], 255, dtype=np.uint8)])

    outside = valid_mask == 0
    rgba[outside, 0] = np.clip(rgba[outside, 0] * 0.35 + 150, 0, 255).astype(np.uint8)
    rgba[outside, 1] = (rgba[outside, 1] * 0.35).astype(np.uint8)
    rgba[outside, 2] = (rgba[outside, 2] * 0.35).astype(np.uint8)
    overlay = Image.fromarray(rgba, mode="RGBA")

    outline = valid_mask > 0
    outline = np.logical_xor(outline, np.pad(outline[1:, :], ((0, 1), (0, 0))))
    outline |= np.logical_xor(valid_mask > 0, np.pad((valid_mask > 0)[:, 1:], ((0, 0), (0, 1))))
    outline_img = Image.fromarray((outline.astype(np.uint8) * 255), mode="L")
    outline_rgba = Image.new("RGBA", overlay.size, (0, 255, 0, 0))
    outline_rgba.putalpha(outline_img)
    overlay = Image.alpha_composite(overlay, outline_rgba)

    draw = ImageDraw.Draw(overlay)
    rect = (
        safe_rect["x"],
        safe_rect["y"],
        safe_rect["x"] + safe_rect["width"] - 1,
        safe_rect["y"] + safe_rect["height"] - 1,
    )
    draw.rectangle(rect, outline=ImageColor.getrgb("#00e676"), width=5)

    if square_rect is not None:
        square = (
            square_rect["x"],
            square_rect["y"],
            square_rect["x"] + square_rect["width"] - 1,
            square_rect["y"] + square_rect["height"] - 1,
        )
        draw.rectangle(square, outline=ImageColor.getrgb("#00bcd4"), width=5)

    if labeled_rects is None and span_rects is not None:
        labeled_rects = [
            {"rect": span_rects["left"], "color": "#ff9800", "label": "ROI_L"},
            {"rect": span_rects["center"], "color": "#00bcd4", "label": "ROI_C"},
            {"rect": span_rects["right"], "color": "#ff4d94", "label": "ROI_R"},
        ]

    if labeled_rects is not None:
        for item in labeled_rects:
            rect_data = item["rect"]
            color = str(item["color"])
            label = str(item["label"])
            rect = (
                int(rect_data["x"]),
                int(rect_data["y"]),
                int(rect_data["x"]) + int(rect_data["width"]) - 1,
                int(rect_data["y"]) + int(rect_data["height"]) - 1,
            )
            draw.rectangle(rect, outline=ImageColor.getrgb(color), width=5)
            text_xy = (int(rect_data["x"]) + 12, max(52, int(rect_data["y"]) + 10))
            text_box = (
                text_xy[0] - 6,
                text_xy[1] - 4,
                text_xy[0] + max(58, 8 * len(label)),
                text_xy[1] + 18,
            )
            draw.rectangle(text_box, fill=(10, 10, 10, 220))
            draw.text(text_xy, label, fill=ImageColor.getrgb(color))
        label_set = {str(item["label"]) for item in labeled_rects}
        if {"ROI_L", "ROI_C", "ROI_R"}.issubset(label_set):
            rect_by_label = {str(item["label"]): item["rect"] for item in labeled_rects}
            left_rect = rect_by_label["ROI_L"]
            center_rect = rect_by_label["ROI_C"]
            right_rect = rect_by_label["ROI_R"]
            line_y = int(center_rect["y"]) + int(center_rect["height"]) // 2
            line_x0 = int(left_rect["x"]) + int(left_rect["width"]) // 2
            line_x1 = int(right_rect["x"]) + int(right_rect["width"]) // 2
            draw.line((line_x0, line_y, line_x1, line_y), fill="#ffd54f", width=4)
            for point_x in (line_x0, line_x1):
                draw.ellipse((point_x - 8, line_y - 8, point_x + 8, line_y + 8), fill="#ffd54f")

    cx = center_xy["x"]
    cy = center_xy["y"]
    draw.line((cx - 18, cy, cx + 18, cy), fill="#ffea00", width=3)
    draw.line((cx, cy - 18, cx, cy + 18), fill="#ffea00", width=3)

    title_band = Image.new("RGBA", (overlay.width, 48), color=(8, 8, 8, 220))
    title_draw = ImageDraw.Draw(title_band)
    label = image_path.stem.replace("_normal_000_rgb", "")
    title_draw.text((18, 14), label, fill="white")
    merged = Image.new("RGBA", (overlay.width, overlay.height + title_band.height), color=(0, 0, 0, 255))
    merged.paste(title_band, (0, 0))
    merged.paste(overlay, (0, title_band.height))
    return merged


def centered_square_from_safe_rect(safe_rect: dict[str, int], size: int) -> dict[str, int]:
    if size > safe_rect["width"] or size > safe_rect["height"]:
        raise ValueError(
            f"Requested square size {size} does not fit inside safe rectangle "
            f"{safe_rect['width']}x{safe_rect['height']}."
        )
    x = safe_rect["x"] + (safe_rect["width"] - size) // 2
    y = safe_rect["y"] + (safe_rect["height"] - size) // 2
    return {
        "x": int(x),
        "y": int(y),
        "width": int(size),
        "height": int(size),
        "area": int(size * size),
    }


def integral_image(mask: np.ndarray) -> np.ndarray:
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), constant_values=0)
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return integral


def window_sum(integral: np.ndarray, x: int, y: int, width: int, height: int) -> int:
    x2 = x + width
    y2 = y + height
    return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def fixed_y_square_span(valid_mask: np.ndarray, safe_rect: dict[str, int], size: int) -> dict[str, object]:
    center_y = safe_rect["y"] + safe_rect["height"] / 2.0
    top = int(round(center_y - size / 2.0))
    if top < 0 or top + size > valid_mask.shape[0]:
        raise ValueError(f"Square size {size} exceeds image bounds at fixed y-center.")

    integ = integral_image(valid_mask > 0)
    valid_x = []
    for x in range(0, valid_mask.shape[1] - size + 1):
        if window_sum(integ, x, top, size, size) == size * size:
            valid_x.append(x)

    if not valid_x:
        raise ValueError(f"No valid x-position found for {size}x{size} at fixed y-center.")

    left_x = min(valid_x)
    right_x = max(valid_x)
    center_rect = centered_square_from_safe_rect(safe_rect, size)
    return {
        "size": int(size),
        "y_fixed_top": int(top),
        "x_valid_min": int(left_x),
        "x_valid_max": int(right_x),
        "x_valid_span_px": int(right_x - left_x),
        "left": {"x": int(left_x), "y": int(top), "width": int(size), "height": int(size), "area": int(size * size)},
        "center": center_rect,
        "right": {"x": int(right_x), "y": int(top), "width": int(size), "height": int(size), "area": int(size * size)},
    }


def build_contact_sheet(images: list[Image.Image], output_path: Path, cols: int = 3) -> None:
    if not images:
        raise ValueError("No images to place in contact sheet.")
    rows = math.ceil(len(images) / cols)
    tile_w = images[0].width
    tile_h = images[0].height
    canvas = Image.new("RGBA", (tile_w * cols, tile_h * rows), color=(5, 5, 5, 255))
    for idx, image in enumerate(images):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        canvas.paste(image, (x, y))
    canvas.save(output_path)


def sample_sort_key(path: Path) -> tuple[int, str]:
    match = SAMPLE_RE.search(path.stem)
    if match:
        return (int(match.group(1)), path.stem)
    return (10**9, path.stem)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(args.geometry_json.read_text(encoding="utf-8"))
    safe_rect = geometry["largest_axis_aligned_rectangle"]
    center_xy = geometry["center_xy_fullres"]
    valid_mask = np.asarray(Image.open(args.mask_path).convert("L"), dtype=np.uint8)

    samples = sorted(args.roi_dir.glob("*_rgb.tif"), key=sample_sort_key)
    base_annotated_images = []
    base_entries = []
    for image_path in samples:
        annotated = annotate_image(image_path, safe_rect, center_xy, valid_mask)
        output_path = args.output_dir / f"{image_path.stem}_annotated.png"
        annotated.convert("RGB").save(output_path)
        base_annotated_images.append(annotated)
        base_entries.append(
            {
                "sample_image": str(image_path),
                "annotated_output": str(output_path),
            }
        )

    build_contact_sheet(base_annotated_images, args.output_dir / "roi_review_contact_sheet.png", cols=3)

    square_outputs = []
    for size in args.square_sizes:
        square_rect = centered_square_from_safe_rect(safe_rect, int(size))
        annotated_images = []
        sample_entries = []
        span_info = fixed_y_square_span(valid_mask, safe_rect, int(size))
        for image_path in samples:
            annotated = annotate_image(
                image_path,
                safe_rect,
                center_xy,
                valid_mask,
                square_rect=square_rect,
            )
            output_path = args.output_dir / f"{image_path.stem}_sq{size}_annotated.png"
            annotated.convert("RGB").save(output_path)
            annotated_images.append(annotated)
            sample_entries.append(
                {
                    "sample_image": str(image_path),
                    "annotated_output": str(output_path),
                }
            )
        contact_sheet = args.output_dir / f"roi_review_contact_sheet_sq{size}.png"
        build_contact_sheet(annotated_images, contact_sheet, cols=3)

        span_annotated_images = []
        span_sample_entries = []
        for image_path in samples:
            annotated = annotate_image(
                image_path,
                safe_rect,
                center_xy,
                valid_mask,
                span_rects=span_info,
            )
            output_path = args.output_dir / f"{image_path.stem}_sq{size}_span_annotated.png"
            annotated.convert("RGB").save(output_path)
            span_annotated_images.append(annotated)
            span_sample_entries.append(
                {
                    "sample_image": str(image_path),
                    "annotated_output": str(output_path),
                }
            )
        span_contact_sheet = args.output_dir / f"roi_review_contact_sheet_sq{size}_span.png"
        build_contact_sheet(span_annotated_images, span_contact_sheet, cols=3)

        square_outputs.append(
            {
                "size": int(size),
                "square_rect": square_rect,
                "contact_sheet": str(contact_sheet),
                "annotated_images": sample_entries,
                "fixed_y_span": {
                    "y_fixed_top": span_info["y_fixed_top"],
                    "x_valid_min": span_info["x_valid_min"],
                    "x_valid_max": span_info["x_valid_max"],
                    "x_valid_span_px": span_info["x_valid_span_px"],
                    "left_rect": span_info["left"],
                    "center_rect": span_info["center"],
                    "right_rect": span_info["right"],
                    "contact_sheet": str(span_contact_sheet),
                    "annotated_images": span_sample_entries,
                },
            }
        )

    summary = {
        "geometry_json": str(args.geometry_json),
        "mask_path": str(args.mask_path),
        "safe_rectangle": safe_rect,
        "rotation_center": center_xy,
        "annotated_images": base_entries,
        "contact_sheet": str(args.output_dir / "roi_review_contact_sheet.png"),
        "centered_square_overlays": square_outputs,
    }
    (args.output_dir / "roi_review_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
