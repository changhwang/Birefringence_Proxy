from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from build_roi_review import annotate_image, build_contact_sheet, sample_sort_key


DEFAULT_ROI_DIR = Path("data/ROI_selection")
DEFAULT_GEOMETRY_JSON = Path("analysis_outputs/rotation_calibration/derotation_geometry.json")
DEFAULT_MASK_PATH = Path("analysis_outputs/rotation_calibration/rotation_calibration_valid_mask.png")
DEFAULT_LAYOUT_JSON = Path("configs/roi_presets/horizontal_center_3split.json")
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/roi_review/horizontal_center_3split")


SAMPLE_RE = re.compile(r"(QLB_s\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-sample custom ROI presets on the 0 deg RGB review images."
    )
    parser.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR)
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_JSON)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[500, 400, 300],
        help="Subset of sizes from the layout JSON to render.",
    )
    return parser.parse_args()


def sample_label_from_path(path: Path) -> str:
    match = SAMPLE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer sample label from {path}")
    return match.group(1)


def build_square_rect(x: int, y: int, size: int) -> dict[str, int]:
    return {
        "x": int(x),
        "y": int(y),
        "width": int(size),
        "height": int(size),
        "area": int(size * size),
    }


def resolve_sample_layout(spec: dict[str, object], sample_label: str) -> dict[str, object]:
    default = dict(spec["default"])
    override = dict(spec.get("sample_overrides", {}).get(sample_label, {}))
    if not override:
        return default

    layout_type = override.get("type", default.get("type"))
    if layout_type == "single":
        x = int(override.get("x", default.get("x", default.get("center_x"))))
        return {"type": "single", "x": x}

    if layout_type == "span":
        return {
            "type": "span",
            "left_x": int(override.get("left_x", default["left_x"])),
            "center_x": int(override.get("center_x", default["center_x"])),
            "right_x": int(override.get("right_x", default["right_x"])),
        }

    raise ValueError(f"Unsupported layout type: {layout_type}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(args.geometry_json.read_text(encoding="utf-8"))
    safe_rect = geometry["largest_axis_aligned_rectangle"]
    center_xy = geometry["center_xy_fullres"]
    valid_mask = np.asarray(Image.open(args.mask_path).convert("L"), dtype=np.uint8)
    layout = json.loads(args.layout_json.read_text(encoding="utf-8"))

    samples = sorted(args.roi_dir.glob("*_rgb.tif"), key=sample_sort_key)
    rendered_sizes = []

    for size in args.sizes:
        size_key = str(size)
        if size_key not in layout:
            raise KeyError(f"Size {size} not found in {args.layout_json}")

        spec = layout[size_key]
        y = int(spec["y"])
        size = int(spec["size"])

        annotated_images = []
        image_entries = []
        for image_path in samples:
            sample_label = sample_label_from_path(image_path)
            sample_layout = resolve_sample_layout(spec, sample_label)
            entry = {
                "sample_image": str(image_path),
                "sample_label": sample_label,
                "y": int(y),
                "size": int(size),
            }

            if sample_layout["type"] == "single":
                x = int(sample_layout["x"])
                square_rect = build_square_rect(x, y, size)
                annotated = annotate_image(
                    image_path=image_path,
                    safe_rect=safe_rect,
                    center_xy=center_xy,
                    valid_mask=valid_mask,
                    square_rect=square_rect,
                )
                entry.update(
                    {
                        "layout_type": "single",
                        "x": x,
                    }
                )
            else:
                span_rects = {
                    "left": build_square_rect(int(sample_layout["left_x"]), y, size),
                    "center": build_square_rect(int(sample_layout["center_x"]), y, size),
                    "right": build_square_rect(int(sample_layout["right_x"]), y, size),
                }
                annotated = annotate_image(
                    image_path=image_path,
                    safe_rect=safe_rect,
                    center_xy=center_xy,
                    valid_mask=valid_mask,
                    span_rects=span_rects,
                )
                entry.update(
                    {
                        "layout_type": "span",
                        "ROI_L": {"x": int(sample_layout["left_x"]), "y": int(y), "size": int(size)},
                        "ROI_C": {"x": int(sample_layout["center_x"]), "y": int(y), "size": int(size)},
                        "ROI_R": {"x": int(sample_layout["right_x"]), "y": int(y), "size": int(size)},
                    }
                )

            output_path = args.output_dir / f"{image_path.stem}_sq{size}_custom.png"
            annotated.convert("RGB").save(output_path)
            annotated_images.append(annotated)
            entry["annotated_output"] = str(output_path)
            image_entries.append(entry)

        contact_sheet = args.output_dir / f"roi_review_contact_sheet_sq{size}_custom.png"
        build_contact_sheet(annotated_images, contact_sheet, cols=3)
        rendered_sizes.append(
            {
                "size": int(size),
                "default_layout": spec["default"],
                "y": int(y),
                "contact_sheet": str(contact_sheet),
                "annotated_images": image_entries,
            }
        )

    summary = {
        "geometry_json": str(args.geometry_json),
        "layout_json": str(args.layout_json),
        "mask_path": str(args.mask_path),
        "safe_rectangle": safe_rect,
        "rotation_center": center_xy,
        "rendered_sizes": rendered_sizes,
    }
    (args.output_dir / "custom_roi_review_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
