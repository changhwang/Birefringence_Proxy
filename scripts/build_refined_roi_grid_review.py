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
DEFAULT_LAYOUT_JSON = Path("configs/roi_presets/refined_grid_3x3_v1.json")
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/roi_review/refined_grid_3x3_v1")

SAMPLE_RE = re.compile(r"(QLB_s(\d+))", re.IGNORECASE)

PALETTE = {
    "ROI_UL": "#ff8a65",
    "ROI_UC": "#ffb74d",
    "ROI_UR": "#ffd54f",
    "ROI_CL": "#4db6ac",
    "ROI_CC": "#00bcd4",
    "ROI_CR": "#64b5f6",
    "ROI_LL": "#ba68c8",
    "ROI_LC": "#f06292",
    "ROI_LR": "#ef5350",
}

DISPLAY_LABELS = {
    "ROI_UL": "UL",
    "ROI_UC": "UC",
    "ROI_UR": "UR",
    "ROI_CL": "CL",
    "ROI_CC": "CC",
    "ROI_CR": "CR",
    "ROI_LL": "LL",
    "ROI_LC": "LC",
    "ROI_LR": "LR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render refined grid ROI presets on the 0 deg RGB review images."
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
        default=[150, 200, 300],
        help="Subset of sizes from the layout JSON to render.",
    )
    return parser.parse_args()


def sample_label_from_path(path: Path) -> str:
    match = SAMPLE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer sample label from {path}")
    return match.group(1)


def sample_index_from_label(sample_label: str) -> int:
    match = SAMPLE_RE.fullmatch(sample_label)
    if not match:
        raise ValueError(f"Could not infer sample index from {sample_label}")
    return int(match.group(2))


def build_square_rect(x: int, y: int, size: int) -> dict[str, int]:
    return {
        "x": int(x),
        "y": int(y),
        "width": int(size),
        "height": int(size),
        "area": int(size * size),
    }


def shift_rect(rect: dict[str, int], dx: int = 0, dy: int = 0) -> dict[str, int]:
    shifted = dict(rect)
    shifted["x"] = int(shifted["x"] + dx)
    shifted["y"] = int(shifted["y"] + dy)
    return shifted


def midpoint_y(rect_a: dict[str, int], rect_b: dict[str, int]) -> int:
    return int(round((int(rect_a["y"]) + int(rect_b["y"])) / 2.0))


def split_region_overrides(
    region_overrides: dict[str, object],
    keys: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    selected: dict[str, object] = {}
    remaining: dict[str, object] = {}
    for region_key, override in region_overrides.items():
        if region_key in keys:
            selected[region_key] = override
        else:
            remaining[region_key] = override
    return selected, remaining


def apply_layout_offsets(region_rects: dict[str, dict[str, int]], layout: dict[str, object]) -> dict[str, dict[str, int]]:
    adjusted = {key: dict(rect) for key, rect in region_rects.items()}
    all_dx = int(layout.get("all_dx", 0))
    all_dy = int(layout.get("all_dy", 0))
    if all_dx or all_dy:
        adjusted = {key: shift_rect(rect, dx=all_dx, dy=all_dy) for key, rect in adjusted.items()}
    return adjusted


def apply_specific_region_overrides(
    region_rects: dict[str, dict[str, int]],
    region_overrides: dict[str, object],
    layout_type: str,
) -> dict[str, dict[str, int]]:
    adjusted = {key: dict(rect) for key, rect in region_rects.items()}
    for region_key, override_obj in region_overrides.items():
        if region_key not in adjusted:
            raise KeyError(f"Region override {region_key} does not exist in layout {layout_type}")
        override = dict(override_obj)
        rect = dict(adjusted[region_key])
        if "x" in override:
            rect["x"] = int(override["x"])
        if "y" in override:
            rect["y"] = int(override["y"])
        rect = shift_rect(rect, dx=int(override.get("dx", 0)), dy=int(override.get("dy", 0)))
        adjusted[region_key] = rect
    return adjusted


def build_region_rects(layout: dict[str, object], size: int) -> dict[str, dict[str, int]]:
    layout_type = str(layout["type"])
    center_row_mode = str(layout.get("center_row_mode", "manual"))
    region_overrides = dict(layout.get("region_overrides", {}))
    if layout_type == "grid9":
        xs = {
            "L": int(layout["left_x"]),
            "C": int(layout["center_x"]),
            "R": int(layout["right_x"]),
        }
        ys = {
            "U": int(layout["upper_y"]),
            "C": int(layout["center_y"]),
            "L": int(layout["lower_y"]),
        }
        region_rects = {
            "ROI_UL": build_square_rect(xs["L"], ys["U"], size),
            "ROI_UC": build_square_rect(xs["C"], ys["U"], size),
            "ROI_UR": build_square_rect(xs["R"], ys["U"], size),
            "ROI_CL": build_square_rect(xs["L"], ys["C"], size),
            "ROI_CC": build_square_rect(xs["C"], ys["C"], size),
            "ROI_CR": build_square_rect(xs["R"], ys["C"], size),
            "ROI_LL": build_square_rect(xs["L"], ys["L"], size),
            "ROI_LC": build_square_rect(xs["C"], ys["L"], size),
            "ROI_LR": build_square_rect(xs["R"], ys["L"], size),
        }
        region_rects = apply_layout_offsets(region_rects, layout)
        if center_row_mode == "column_midpoint":
            edge_override_keys = {"ROI_UL", "ROI_UC", "ROI_UR", "ROI_LL", "ROI_LC", "ROI_LR"}
            edge_overrides, remaining_overrides = split_region_overrides(region_overrides, edge_override_keys)
            region_rects = apply_specific_region_overrides(region_rects, edge_overrides, layout_type)
            region_rects["ROI_CL"]["y"] = midpoint_y(region_rects["ROI_UL"], region_rects["ROI_LL"])
            region_rects["ROI_CC"]["y"] = midpoint_y(region_rects["ROI_UC"], region_rects["ROI_LC"])
            region_rects["ROI_CR"]["y"] = midpoint_y(region_rects["ROI_UR"], region_rects["ROI_LR"])
            return apply_specific_region_overrides(region_rects, remaining_overrides, layout_type)
        return apply_specific_region_overrides(region_rects, region_overrides, layout_type)
    if layout_type == "column3":
        x = int(layout["center_x"])
        region_rects = {
            "ROI_UC": build_square_rect(x, int(layout["upper_y"]), size),
            "ROI_CC": build_square_rect(x, int(layout["center_y"]), size),
            "ROI_LC": build_square_rect(x, int(layout["lower_y"]), size),
        }
        region_rects = apply_layout_offsets(region_rects, layout)
        if center_row_mode == "column_midpoint":
            edge_override_keys = {"ROI_UC", "ROI_LC"}
            edge_overrides, remaining_overrides = split_region_overrides(region_overrides, edge_override_keys)
            region_rects = apply_specific_region_overrides(region_rects, edge_overrides, layout_type)
            region_rects["ROI_CC"]["y"] = midpoint_y(region_rects["ROI_UC"], region_rects["ROI_LC"])
            return apply_specific_region_overrides(region_rects, remaining_overrides, layout_type)
        return apply_specific_region_overrides(region_rects, region_overrides, layout_type)
    raise ValueError(f"Unsupported layout type: {layout_type}")


def resolve_sample_layout(spec: dict[str, object], sample_label: str) -> dict[str, object]:
    sample_index = sample_index_from_label(sample_label)
    base_key = "default_long" if sample_index <= 8 else "default_short"
    resolved = dict(spec[base_key])
    override = dict(spec.get("sample_overrides", {}).get(sample_label, {}))
    for key, value in override.items():
        resolved[key] = value
    return resolved


def build_labeled_rects(region_rects: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    order = [
        "ROI_UL",
        "ROI_UC",
        "ROI_UR",
        "ROI_CL",
        "ROI_CC",
        "ROI_CR",
        "ROI_LL",
        "ROI_LC",
        "ROI_LR",
    ]
    labeled = []
    for key in order:
        if key not in region_rects:
            continue
        labeled.append(
            {
                "key": key,
                "label": DISPLAY_LABELS[key],
                "color": PALETTE[key],
                "rect": region_rects[key],
            }
        )
    return labeled


def mask_contains_rect(valid_mask: np.ndarray, rect: dict[str, int]) -> bool:
    x = int(rect["x"])
    y = int(rect["y"])
    width = int(rect["width"])
    height = int(rect["height"])
    if x < 0 or y < 0 or x + width > valid_mask.shape[1] or y + height > valid_mask.shape[0]:
        return False
    return bool(np.all(valid_mask[y : y + height, x : x + width] > 0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(args.geometry_json.read_text(encoding="utf-8"))
    safe_rect = geometry["largest_axis_aligned_rectangle"]
    center_xy = geometry["center_xy_fullres"]
    valid_mask = np.asarray(Image.open(args.mask_path).convert("L"), dtype=np.uint8)
    layout_root = json.loads(args.layout_json.read_text(encoding="utf-8"))
    size_layouts = layout_root["sizes"]

    samples = sorted(args.roi_dir.glob("*_rgb.tif"), key=sample_sort_key)
    rendered_sizes = []

    for size in args.sizes:
        size_key = str(size)
        if size_key not in size_layouts:
            raise KeyError(f"Size {size} not found in {args.layout_json}")

        spec = size_layouts[size_key]
        annotated_images = []
        image_entries = []

        for image_path in samples:
            sample_label = sample_label_from_path(image_path)
            resolved = resolve_sample_layout(spec, sample_label)
            region_rects = build_region_rects(resolved, int(spec["size"]))
            labeled_rects = build_labeled_rects(region_rects)
            annotated = annotate_image(
                image_path=image_path,
                safe_rect=safe_rect,
                center_xy=center_xy,
                valid_mask=valid_mask,
                labeled_rects=labeled_rects,
            )
            output_path = args.output_dir / f"{image_path.stem}_sq{size}_refined_grid.png"
            annotated.convert("RGB").save(output_path)
            annotated_images.append(annotated)

            image_entries.append(
                {
                    "sample_image": str(image_path),
                    "sample_label": sample_label,
                    "layout_type": resolved["type"],
                    "annotated_output": str(output_path),
                    "regions": {
                        key: {
                            **rect,
                            "display_label": DISPLAY_LABELS[key],
                            "color": PALETTE[key],
                            "inside_valid_mask": mask_contains_rect(valid_mask, rect),
                        }
                        for key, rect in region_rects.items()
                    },
                }
            )

        contact_sheet = args.output_dir / f"roi_review_contact_sheet_sq{size}_refined_grid.png"
        build_contact_sheet(annotated_images, contact_sheet, cols=3)
        rendered_sizes.append(
            {
                "size": int(size),
                "layout_spec": spec,
                "contact_sheet": str(contact_sheet),
                "annotated_images": image_entries,
            }
        )

    summary = {
        "preset_name": layout_root["preset_name"],
        "description": layout_root["description"],
        "related_existing_preset": layout_root.get("related_existing_preset"),
        "geometry_json": str(args.geometry_json),
        "layout_json": str(args.layout_json),
        "mask_path": str(args.mask_path),
        "safe_rectangle": safe_rect,
        "rotation_center": center_xy,
        "rendered_sizes": rendered_sizes,
    }
    (args.output_dir / "refined_roi_grid_review_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
