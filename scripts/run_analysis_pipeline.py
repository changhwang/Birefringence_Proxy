from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "datasets" / "analysis_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis_outputs" / "pipeline_runs" / "current"
ANGLES_DEG = tuple(range(0, 166, 5))
MODE_SEQUENCE = ("PPL", "XPL")
RAW_CHANNELS = ("G1", "G2", "R", "B")
FIT_CHANNELS = ("G", "R", "B")
QC_CHANNELS = ("G1", "G2", "R", "B", "G")
SIGNAL_NAMES = ("Afilm_PPL", "Xfilm", "Xnorm_sample", "Xnorm_blank")
PRIMARY_SIGNAL_TO_METRIC = {
    "Afilm_PPL": "A2",
    "Xfilm": "A4",
    "Xnorm_sample": "A4",
    "Xnorm_blank": "A4",
}
PRIMARY_SIGNAL_TO_AXIS = {
    "Afilm_PPL": "axis2_deg",
    "Xfilm": "axis4_deg",
    "Xnorm_sample": "axis4_deg",
    "Xnorm_blank": "axis4_deg",
}
PRIMARY_SIGNAL_TO_PERIOD = {
    "Afilm_PPL": 180.0,
    "Xfilm": 90.0,
    "Xnorm_sample": 90.0,
    "Xnorm_blank": 90.0,
}
ROUNDTRIP_ANGLES = (5.0, 25.0, 55.0)
SATURATION_LEVEL = 1020.0
MIN_VALID_ANGLES = 28


@dataclass(frozen=True)
class DatasetFrame:
    angle_deg: int
    exposure_us: float
    path: Path


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    sample_type: str
    sample_dir: Path
    blank_dir: Path
    empty_dir: Path
    dark_dir: Path
    roi_size_fullres: int


@dataclass(frozen=True)
class ROIEntry:
    label: str
    full_rect: tuple[int, int, int, int]
    half_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class TileEntry:
    tile_id: int
    roi_index: int
    roi_label: str
    x: int
    y: int
    width: int
    height: int
    valid_fraction: float


DATASET_RECORD_CACHE: dict[Path, dict[str, dict[int, DatasetFrame]]] = {}
RAW_SPLIT_CACHE: dict[Path, dict[str, np.ndarray]] = {}
DARK_MEDIAN_CACHE: dict[tuple[Path, str, float], dict[str, np.ndarray]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw birefringence analysis pipeline.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("qc", "extract", "fit", "summarize", "compare-mm"),
        help="Pipeline stage to run.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", nargs="*", default=["all"])
    parser.add_argument("--mm-csv", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def nanpercentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def circular_mean_deg(values_deg: np.ndarray, period_deg: float) -> float:
    finite = values_deg[np.isfinite(values_deg)]
    if finite.size == 0:
        return float("nan")
    angles = 2.0 * math.pi * finite / period_deg
    vector = np.exp(1j * angles).mean()
    if abs(vector) < 1e-12:
        return float("nan")
    return float((np.angle(vector) * period_deg / (2.0 * math.pi)) % period_deg)


def primary_metric_name(signal_name: str) -> str:
    return PRIMARY_SIGNAL_TO_METRIC[signal_name]


def primary_axis_field(signal_name: str) -> str:
    return PRIMARY_SIGNAL_TO_AXIS[signal_name]


def primary_period_deg(signal_name: str) -> float:
    return PRIMARY_SIGNAL_TO_PERIOD[signal_name]


def validate_no_preview_paths(manifest: dict[str, Any]) -> None:
    banned_tokens = ("\\RGB\\", "/RGB/", "\\ROI_selection\\", "/ROI_selection/")
    for sample in manifest["samples"]:
        for key in ("sample_dir", "blank_dir", "empty_dir", "dark_dir"):
            raw = str(sample[key])
            if any(token in raw for token in banned_tokens):
                raise ValueError(f"Preview path detected in manifest field {key}: {raw}")


def load_manifest(path: Path) -> tuple[dict[str, Any], list[SampleSpec]]:
    manifest = load_json(path)
    validate_no_preview_paths(manifest)
    samples = [
        SampleSpec(
            sample_id=entry["sample_id"],
            sample_type=entry["sample_type"],
            sample_dir=resolve_repo_path(entry["sample_dir"]),
            blank_dir=resolve_repo_path(entry["blank_dir"]),
            empty_dir=resolve_repo_path(entry["empty_dir"]),
            dark_dir=resolve_repo_path(entry["dark_dir"]),
            roi_size_fullres=int(entry["roi_size_fullres"]),
        )
        for entry in manifest["samples"]
    ]
    manifest["roi_preset_source"] = str(resolve_repo_path(manifest["roi_preset_source"]))
    manifest["rotation_geometry"] = str(resolve_repo_path(manifest["rotation_geometry"]))
    manifest["rotation_valid_mask"] = str(resolve_repo_path(manifest["rotation_valid_mask"]))
    return manifest, samples


def select_samples(samples: list[SampleSpec], sample_names: list[str]) -> list[SampleSpec]:
    if not sample_names or sample_names == ["all"]:
        return samples
    requested = set(sample_names)
    selected = [sample for sample in samples if sample.sample_id in requested]
    missing = requested - {sample.sample_id for sample in selected}
    if missing:
        raise KeyError(f"Unknown sample ids requested: {sorted(missing)}")
    return selected


def load_dataset_records(dataset_dir: Path) -> dict[str, dict[int, DatasetFrame]]:
    dataset_dir = dataset_dir.resolve()
    cached = DATASET_RECORD_CACHE.get(dataset_dir)
    if cached is not None:
        return cached

    metadata_paths = sorted(dataset_dir.glob("*_metadata.json"))
    if len(metadata_paths) != 1:
        raise FileNotFoundError(f"Expected one metadata JSON in {dataset_dir}, found {len(metadata_paths)}")

    metadata = load_json(metadata_paths[0])
    records: dict[str, dict[int, DatasetFrame]] = {mode: {} for mode in MODE_SEQUENCE}
    for image in metadata["images"]:
        dataset_mode = image["mode"]
        mode_alias = "PPL" if dataset_mode == "normal" else "XPL"
        angle_deg = int(image["sample_angle_deg"])
        path = dataset_dir / dataset_mode / image["filename"]
        records[mode_alias][angle_deg] = DatasetFrame(
            angle_deg=angle_deg,
            exposure_us=float(image["exposure_us"]),
            path=path,
        )

    for mode in MODE_SEQUENCE:
        missing = sorted(set(ANGLES_DEG) - set(records[mode]))
        if missing:
            raise ValueError(f"Missing {mode} angles in {dataset_dir}: {missing}")

    DATASET_RECORD_CACHE[dataset_dir] = records
    return records


def load_raw16(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32)


def split_bayer_gbrg_block_grid(raw: np.ndarray) -> dict[str, np.ndarray]:
    height = (raw.shape[0] // 2) * 2
    width = (raw.shape[1] // 2) * 2
    trimmed = raw[:height, :width]
    return {
        "G1": trimmed[0::2, 0::2].astype(np.float32, copy=False),
        "B": trimmed[0::2, 1::2].astype(np.float32, copy=False),
        "R": trimmed[1::2, 0::2].astype(np.float32, copy=False),
        "G2": trimmed[1::2, 1::2].astype(np.float32, copy=False),
    }


def get_raw_split(path: Path) -> dict[str, np.ndarray]:
    path = path.resolve()
    cached = RAW_SPLIT_CACHE.get(path)
    if cached is not None:
        return cached
    split = split_bayer_gbrg_block_grid(load_raw16(path))
    RAW_SPLIT_CACHE[path] = split
    return split


def get_dark_split(
    dark_records: dict[str, dict[int, DatasetFrame]],
    mode: str,
    angle_deg: int,
    exposure_us: float,
) -> dict[str, np.ndarray]:
    direct = dark_records[mode].get(angle_deg)
    if direct is not None and math.isclose(direct.exposure_us, exposure_us):
        return get_raw_split(direct.path)

    cache_key = (dark_records[mode][ANGLES_DEG[0]].path.parents[1].resolve(), mode, float(exposure_us))
    cached = DARK_MEDIAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    matches = [frame for frame in dark_records[mode].values() if math.isclose(frame.exposure_us, exposure_us)]
    if not matches:
        raise ValueError(f"No dark frame available for mode={mode} exposure_us={exposure_us}")

    channel_stacks: dict[str, list[np.ndarray]] = {channel: [] for channel in RAW_CHANNELS}
    for frame in matches:
        split = get_raw_split(frame.path)
        for channel in RAW_CHANNELS:
            channel_stacks[channel].append(split[channel])
    median_split = {
        channel: np.median(np.stack(channel_stacks[channel], axis=0), axis=0).astype(np.float32)
        for channel in RAW_CHANNELS
    }
    DARK_MEDIAN_CACHE[cache_key] = median_split
    return median_split


def rotate_plane(arr: np.ndarray, angle_deg: float, center_xy: tuple[float, float]) -> np.ndarray:
    """Derotate: CCW rotation to undo CW sample rotation."""
    if abs(angle_deg) < 1e-9:
        return arr.astype(np.float32, copy=True)
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    rotated = image.rotate(
        angle_deg,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        center=center_xy,
        fillcolor=0.0,
    )
    return np.asarray(rotated, dtype=np.float32)


_CORRECTION_CACHE: dict[str, dict[str, Any]] | None = None


def load_correction(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Load solved_correction.json if referenced in manifest. Returns None if not available."""
    global _CORRECTION_CACHE
    corr_path_str = manifest.get("rotation_correction")
    if not corr_path_str:
        return None
    corr_path = Path(corr_path_str)
    if not corr_path.exists():
        return None
    cache_key = str(corr_path)
    if _CORRECTION_CACHE is not None and cache_key in _CORRECTION_CACHE:
        return _CORRECTION_CACHE[cache_key]

    raw = load_json(corr_path)
    cx = float(raw["solved_center"]["x"])
    cy = float(raw["solved_center"]["y"])
    center_half = ((cx - 0.5) / 2.0, (cy - 0.5) / 2.0)

    shift_table: dict[int, tuple[float, float]] = {}
    for entry in raw.get("translation_table", []):
        angle = int(round(entry["angle_deg"]))
        dx = float(entry.get("shift_x_px", 0.0)) / 2.0
        dy = float(entry.get("shift_y_px", 0.0)) / 2.0
        shift_table[angle] = (dx, dy)

    result = {"center_half": center_half, "shift_table": shift_table}
    if _CORRECTION_CACHE is None:
        _CORRECTION_CACHE = {}
    _CORRECTION_CACHE[cache_key] = result
    return result


def get_center_halfres(manifest: dict[str, Any]) -> tuple[float, float]:
    """Get rotation center in half-res coords, preferring solved_correction if available."""
    corr = load_correction(manifest)
    if corr is not None:
        return corr["center_half"]
    return load_rotation_center_halfres(Path(manifest["rotation_geometry"]))


def derotate_and_shift(
    arr: np.ndarray,
    angle_deg: float,
    center_xy: tuple[float, float],
    manifest: dict[str, Any],
) -> np.ndarray:
    """Derotate plane and apply per-angle translational shift from solved_correction."""
    rotated = rotate_plane(arr, angle_deg, center_xy)
    corr = load_correction(manifest)
    if corr is None:
        return rotated
    angle_key = int(round(angle_deg))
    dx, dy = corr["shift_table"].get(angle_key, (0.0, 0.0))
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return rotated
    return ndimage.shift(rotated, (dy, dx), order=1, mode="constant", cval=0.0).astype(np.float32)


def load_halfres_valid_mask(mask_path: Path) -> np.ndarray:
    full_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    height2 = full_mask.shape[0] // 2
    width2 = full_mask.shape[1] // 2
    trimmed = full_mask[: height2 * 2, : width2 * 2]
    return trimmed.reshape(height2, 2, width2, 2).all(axis=(1, 3))


def load_rotation_center_halfres(geometry_path: Path) -> tuple[float, float]:
    geometry = load_json(geometry_path)
    center_x = float(geometry["center_xy_fullres"]["x"])
    center_y = float(geometry["center_xy_fullres"]["y"])
    return ((center_x - 0.5) / 2.0, (center_y - 0.5) / 2.0)


def build_square_rect(x: int, y: int, size: int) -> tuple[int, int, int, int]:
    return (int(x), int(y), int(size), int(size))


def resolve_sample_layout(size_spec: dict[str, Any], sample_id: str) -> dict[str, Any]:
    default = dict(size_spec["default"])
    override = dict(size_spec.get("sample_overrides", {}).get(sample_id, {}))
    if not override:
        return default
    layout_type = override.get("type", default.get("type"))
    if layout_type == "single":
        return {"type": "single", "x": int(override.get("x", default.get("x", default.get("center_x"))))}
    if layout_type == "span":
        return {
            "type": "span",
            "left_x": int(override.get("left_x", default["left_x"])),
            "center_x": int(override.get("center_x", default["center_x"])),
            "right_x": int(override.get("right_x", default["right_x"])),
        }
    raise ValueError(f"Unsupported layout type {layout_type}")


def full_rect_to_half_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    return (x // 2, y // 2, width // 2, height // 2)


def build_roi_entries(sample: SampleSpec, roi_layout: dict[str, Any]) -> list[ROIEntry]:
    size_key = str(sample.roi_size_fullres)
    if size_key not in roi_layout:
        raise KeyError(f"ROI size {size_key} missing from layout")
    size_spec = roi_layout[size_key]
    resolved = resolve_sample_layout(size_spec, sample.sample_id)
    y = int(size_spec["y"])
    size = int(size_spec["size"])
    entries: list[ROIEntry] = []
    if resolved["type"] == "single":
        full_rect = build_square_rect(int(resolved["x"]), y, size)
        entries.append(ROIEntry("ROI_C", full_rect, full_rect_to_half_rect(full_rect)))
    else:
        for label, key in (("ROI_L", "left_x"), ("ROI_C", "center_x"), ("ROI_R", "right_x")):
            full_rect = build_square_rect(int(resolved[key]), y, size)
            entries.append(ROIEntry(label, full_rect, full_rect_to_half_rect(full_rect)))
    return entries


def build_union_mask(shape: tuple[int, int], roi_entries: list[ROIEntry]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for roi in roi_entries:
        x, y, width, height = roi.half_rect
        mask[y : y + height, x : x + width] = True
    return mask


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def enumerate_tiles(
    roi_entries: list[ROIEntry],
    half_valid_mask: np.ndarray,
    tile_size: int,
    stride: int,
    valid_fraction_threshold: float,
) -> tuple[list[TileEntry], list[np.ndarray]]:
    tiles: list[TileEntry] = []
    tile_masks: list[np.ndarray] = []
    tile_id = 0
    for roi_index, roi in enumerate(roi_entries):
        x0, y0, width, height = roi.half_rect
        for y in range(y0, y0 + height - tile_size + 1, stride):
            for x in range(x0, x0 + width - tile_size + 1, stride):
                local_valid = half_valid_mask[y : y + tile_size, x : x + tile_size]
                valid_fraction = float(local_valid.mean())
                if valid_fraction < valid_fraction_threshold:
                    continue
                tiles.append(
                    TileEntry(
                        tile_id=tile_id,
                        roi_index=roi_index,
                        roi_label=roi.label,
                        x=x,
                        y=y,
                        width=tile_size,
                        height=tile_size,
                        valid_fraction=valid_fraction,
                    )
                )
                tile_masks.append(local_valid.copy())
                tile_id += 1
    if not tiles:
        raise ValueError("No valid tiles found for the selected ROIs")
    return tiles, tile_masks


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    values = arr[mask]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def linear_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    xyz = np.array(
        [
            0.4124564 * rgb[0] + 0.3575761 * rgb[1] + 0.1804375 * rgb[2],
            0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.0721750 * rgb[2],
            0.0193339 * rgb[0] + 0.1191920 * rgb[1] + 0.9503041 * rgb[2],
        ],
        dtype=np.float64,
    )
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    scaled = xyz / np.maximum(white, 1e-12)
    delta = 6.0 / 29.0
    cubic = delta**3
    linear = scaled / (3.0 * delta * delta) + 4.0 / 29.0
    f = np.where(scaled > cubic, np.cbrt(scaled), linear)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])], dtype=np.float64)


def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    return float(np.linalg.norm(lab_a - lab_b))


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


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(arr.astype(np.float64))
    return np.hypot(grad_x, grad_y).astype(np.float32)


def fit_harmonic_curve(values: np.ndarray, valid_mask: np.ndarray, min_valid_angles: int = MIN_VALID_ANGLES) -> dict[str, Any]:
    valid = np.isfinite(values) & valid_mask
    n_valid = int(valid.sum())
    result = {
        "fit_valid": False,
        "n_valid_angles": n_valid,
        "a0": float("nan"),
        "a2c": float("nan"),
        "a2s": float("nan"),
        "a4c": float("nan"),
        "a4s": float("nan"),
        "A2": float("nan"),
        "A4": float("nan"),
        "phase2_raw_deg": float("nan"),
        "phase4_raw_deg": float("nan"),
        "axis2_deg": float("nan"),
        "axis4_deg": float("nan"),
        "rmse": float("nan"),
        "nrmse": float("nan"),
        "predicted": np.full(len(values), np.nan, dtype=np.float64),
    }
    if n_valid < min_valid_angles:
        return result

    theta = np.deg2rad(np.asarray(ANGLES_DEG, dtype=np.float64))
    design = np.column_stack(
        [
            np.ones(theta.shape[0], dtype=np.float64),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
            np.cos(4.0 * theta),
            np.sin(4.0 * theta),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    predicted = design @ coefficients
    residual = values[valid] - predicted[valid]

    a0, a2c, a2s, a4c, a4s = coefficients
    phase2_raw = math.atan2(a2s, a2c)
    phase4_raw = math.atan2(a4s, a4c)
    axis2 = 0.5 * phase2_raw
    axis4 = 0.25 * phase4_raw
    result.update(
        {
            "fit_valid": True,
            "a0": float(a0),
            "a2c": float(a2c),
            "a2s": float(a2s),
            "a4c": float(a4c),
            "a4s": float(a4s),
            "A2": float(np.hypot(a2c, a2s)),
            "A4": float(np.hypot(a4c, a4s)),
            "phase2_raw_deg": float(np.degrees(phase2_raw)),
            "phase4_raw_deg": float(np.degrees(phase4_raw)),
            "axis2_deg": float(np.degrees(axis2) % 180.0),
            "axis4_deg": float(np.degrees(axis4) % 90.0),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "nrmse": float(np.sqrt(np.mean(residual**2)) / (abs(a0) + 1e-12)),
            "predicted": predicted.astype(np.float64),
        }
    )
    return result


def compute_blank_flatness_metrics(blank_g_curves: dict[str, list[float]]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passes = []
    for mode in MODE_SEQUENCE:
        values = np.asarray(blank_g_curves[mode], dtype=np.float64)
        fit = fit_harmonic_curve(values, np.isfinite(values), min_valid_angles=10)
        mean_value = abs(nanmean(values))
        cv_value = float(np.std(values) / max(abs(np.mean(values)), 1e-12))
        max_ratio = float(max(fit["A2"], fit["A4"]) / max(mean_value, 1e-12)) if fit["fit_valid"] else float("inf")
        passed = cv_value < 0.05 and max_ratio < 0.03
        passes.append(passed)
        details[mode] = {
            "cv_theta": cv_value,
            "A2": fit["A2"],
            "A4": fit["A4"],
            "max_harmonic_ratio": max_ratio,
            "flag": passed,
        }
    return {"flag": bool(all(passes)), "details": details}


def compute_signal_values_for_scalar(
    sample_ppl: float,
    sample_xpl: float,
    blank_ppl: float,
    blank_xpl: float,
    eps_count: float,
    tau_low: float,
    t_floor: float = 1e-6,
) -> tuple[dict[str, float], dict[str, bool]]:
    values = {signal: float("nan") for signal in SIGNAL_NAMES}
    valid = {signal: False for signal in SIGNAL_NAMES}

    xfilm = sample_xpl - blank_xpl
    values["Xfilm"] = xfilm
    valid["Xfilm"] = np.isfinite(xfilm)

    if blank_ppl >= tau_low:
        transmission = sample_ppl / (blank_ppl + eps_count)
        if transmission > 0.0 and np.isfinite(transmission):
            values["Afilm_PPL"] = -math.log10(max(transmission, t_floor))
            valid["Afilm_PPL"] = True
            values["Xnorm_blank"] = xfilm / (blank_ppl + eps_count)
            valid["Xnorm_blank"] = np.isfinite(values["Xnorm_blank"])

    if sample_ppl >= tau_low:
        values["Xnorm_sample"] = xfilm / (sample_ppl + eps_count)
        valid["Xnorm_sample"] = np.isfinite(values["Xnorm_sample"])

    return values, valid


def build_qc_curves_template() -> dict[str, Any]:
    def mode_template() -> dict[str, dict[str, list[float]]]:
        return {mode: {channel: [float("nan")] * len(ANGLES_DEG) for channel in QC_CHANNELS} for mode in MODE_SEQUENCE}

    return {
        "dark_raw": mode_template(),
        "blank_corr": mode_template(),
        "empty_corr": mode_template(),
        "sample_corr": mode_template(),
    }


def compute_roi_signal_means(
    signal_values: np.ndarray,
    signal_valid: np.ndarray,
    tile_roi_index: np.ndarray,
    roi_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    roi_signal_means = np.full((roi_count, len(ANGLES_DEG), len(SIGNAL_NAMES), len(FIT_CHANNELS)), np.nan, dtype=np.float32)
    roi_signal_valid_counts = np.zeros((roi_count, len(ANGLES_DEG), len(SIGNAL_NAMES), len(FIT_CHANNELS)), dtype=np.int32)
    for roi_index in range(roi_count):
        tile_mask = tile_roi_index == roi_index
        if not np.any(tile_mask):
            continue
        for signal_index in range(len(SIGNAL_NAMES)):
            for channel_index in range(len(FIT_CHANNELS)):
                roi_values = signal_values[tile_mask, :, signal_index, channel_index]
                roi_valid = signal_valid[tile_mask, :, signal_index, channel_index]
                roi_signal_valid_counts[roi_index, :, signal_index, channel_index] = roi_valid.sum(axis=0)
                filled = np.where(roi_valid, roi_values, np.nan)
                roi_signal_means[roi_index, :, signal_index, channel_index] = np.nanmean(filled, axis=0)
    return roi_signal_means, roi_signal_valid_counts


def plot_roundtrip_qc(roundtrip_rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in roundtrip_rows:
        key = f"{row['dataset_id']} {row['mode']}"
        grouped.setdefault(key, []).append(row)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (label, rows) in zip(axes.flat, grouped.items()):
        rows = sorted(rows, key=lambda item: item["roundtrip_angle_deg"])
        x = [row["roundtrip_angle_deg"] for row in rows]
        axis.plot(x, [row["median_abs_diff"] for row in rows], marker="o", label="median abs diff")
        axis.plot(x, [row["local_contrast_attenuation"] for row in rows], marker="s", label="contrast attenuation")
        axis.plot(x, [row["pearson_corr"] for row in rows], marker="^", label="pearson corr")
        axis.set_title(label)
        axis.set_xlabel("Artificial rotate angle (deg)")
        axis.set_ylabel("Metric value")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Interpolation Round-Trip QC", fontsize=14)
    ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_sample_qc(
    sample: SampleSpec,
    angles_deg: np.ndarray,
    qc_curves: dict[str, Any],
    roi_labels: list[str],
    roi_signal_means_g: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    ax = axes[0, 0]
    for channel in ("G", "R", "B"):
        ax.plot(angles_deg, qc_curves["blank_corr"]["PPL"][channel], label=f"blank {channel}")
        ax.plot(angles_deg, qc_curves["empty_corr"]["PPL"][channel], linestyle="--", label=f"empty {channel}")
    ax.plot(angles_deg, qc_curves["dark_raw"]["PPL"]["G"], color="black", linewidth=1.2, label="dark G raw")
    ax.set_title(f"{sample.sample_id} PPL blank/empty/dark")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Corrected mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for channel in ("G", "R", "B"):
        ax.plot(angles_deg, qc_curves["blank_corr"]["XPL"][channel], label=f"blank {channel}")
        ax.plot(angles_deg, qc_curves["empty_corr"]["XPL"][channel], linestyle="--", label=f"empty {channel}")
    ax.plot(angles_deg, qc_curves["dark_raw"]["XPL"]["G"], color="black", linewidth=1.2, label="dark G raw")
    ax.set_title(f"{sample.sample_id} XPL blank/empty/dark")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Corrected mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(angles_deg, qc_curves["blank_corr"]["PPL"]["G1"], marker="o", label="G1 blank PPL")
    ax.plot(angles_deg, qc_curves["blank_corr"]["PPL"]["G2"], marker="s", label="G2 blank PPL")
    ax.set_title(f"{sample.sample_id} G1/G2 blank PPL")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Corrected mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(angles_deg, qc_curves["blank_corr"]["XPL"]["G1"], marker="o", label="G1 blank XPL")
    ax.plot(angles_deg, qc_curves["blank_corr"]["XPL"]["G2"], marker="s", label="G2 blank XPL")
    ax.set_title(f"{sample.sample_id} G1/G2 blank XPL")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Corrected mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for roi_index, roi_label in enumerate(roi_labels):
        ax.plot(angles_deg, roi_signal_means_g[roi_index, :, 0], label=f"{roi_label} Afilm")
        ax.plot(angles_deg, roi_signal_means_g[roi_index, :, 1], linestyle="--", label=f"{roi_label} Xfilm")
    ax.set_title(f"{sample.sample_id} primary G signals")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Signal value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    for roi_index, roi_label in enumerate(roi_labels):
        ax.plot(angles_deg, roi_signal_means_g[roi_index, :, 2], label=f"{roi_label} Xnorm sample")
        ax.plot(angles_deg, roi_signal_means_g[roi_index, :, 3], linestyle="--", label=f"{roi_label} Xnorm blank")
    ax.set_title(f"{sample.sample_id} normalized G signals")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Signal value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_fit_overlays(
    sample: SampleSpec,
    roi_labels: list[str],
    roi_signal_means_g: np.ndarray,
    roi_signal_valid_counts_g: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, signal_name in zip(axes.flat, SIGNAL_NAMES):
        signal_index = SIGNAL_NAMES.index(signal_name)
        for roi_index, roi_label in enumerate(roi_labels):
            values = roi_signal_means_g[roi_index, :, signal_index]
            valid_mask = roi_signal_valid_counts_g[roi_index, :, signal_index] > 0
            fit = fit_harmonic_curve(values, valid_mask)
            axis.plot(ANGLES_DEG, values, marker="o", label=f"{roi_label} raw")
            if fit["fit_valid"]:
                axis.plot(ANGLES_DEG, fit["predicted"], linestyle="--", label=f"{roi_label} fit")
        axis.set_title(f"{sample.sample_id} {signal_name}")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Signal")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def compute_roundtrip_metrics(
    plane: np.ndarray,
    valid_mask: np.ndarray,
    tile_size: int,
    stride: int,
    center_xy: tuple[float, float],
) -> list[dict[str, float]]:
    rows = []
    valid_tiles: list[tuple[int, int, np.ndarray]] = []
    for y in range(0, plane.shape[0] - tile_size + 1, stride):
        for x in range(0, plane.shape[1] - tile_size + 1, stride):
            local_mask = valid_mask[y : y + tile_size, x : x + tile_size]
            if float(local_mask.mean()) < 0.9:
                continue
            valid_tiles.append((x, y, local_mask.copy()))
    for rotate_angle in ROUNDTRIP_ANGLES:
        rotated = rotate_plane(plane, -rotate_angle, center_xy)
        roundtrip = rotate_plane(rotated, rotate_angle, center_xy)
        diff = np.abs(roundtrip - plane)
        ref_values = plane[valid_mask]
        rt_values = roundtrip[valid_mask]
        tile_mean_diffs = []
        tile_std_diffs = []
        contrast_ratios = []
        for x, y, local_mask in valid_tiles:
            original_tile = plane[y : y + tile_size, x : x + tile_size][local_mask]
            roundtrip_tile = roundtrip[y : y + tile_size, x : x + tile_size][local_mask]
            tile_mean_diffs.append(abs(float(roundtrip_tile.mean() - original_tile.mean())))
            tile_std_diffs.append(abs(float(roundtrip_tile.std() - original_tile.std())))
            contrast_ratios.append(float(roundtrip_tile.std() / max(original_tile.std(), 1e-12)))
        rows.append(
            {
                "roundtrip_angle_deg": rotate_angle,
                "tile_mean_difference": float(np.median(tile_mean_diffs)),
                "tile_std_difference": float(np.median(tile_std_diffs)),
                "median_abs_diff": masked_mean(diff, valid_mask),
                "pearson_corr": float(np.corrcoef(ref_values, rt_values)[0, 1]),
                "local_contrast_attenuation": float(np.median(contrast_ratios)),
            }
        )
    return rows


def extract_sample_bundle(
    sample: SampleSpec,
    manifest: dict[str, Any],
    output_root: Path,
    force: bool,
) -> tuple[Path, Path]:
    signals_dir = ensure_dir(output_root / "signals")
    qc_dir = ensure_dir(output_root / "qc")
    bundle_path = signals_dir / f"{sample.sample_id}_extraction_bundle.npz"
    metadata_path = signals_dir / f"{sample.sample_id}_extraction_metadata.json"
    qc_plot_path = qc_dir / f"{sample.sample_id}_qc_curves.png"
    qc_json_path = qc_dir / f"{sample.sample_id}_qc_metrics.json"
    if bundle_path.exists() and metadata_path.exists() and qc_plot_path.exists() and qc_json_path.exists() and not force:
        return bundle_path, metadata_path

    roi_layout = load_json(Path(manifest["roi_preset_source"]))
    half_valid_mask = load_halfres_valid_mask(Path(manifest["rotation_valid_mask"]))
    center_half = get_center_halfres(manifest)
    roi_entries = build_roi_entries(sample, roi_layout)
    union_mask = build_union_mask(half_valid_mask.shape, roi_entries)
    union_mask_valid = union_mask & half_valid_mask
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = mask_bbox(union_mask_valid)
    bbox_mask = union_mask_valid[bbox_y0:bbox_y1, bbox_x0:bbox_x1]
    tiles, tile_masks = enumerate_tiles(
        roi_entries,
        half_valid_mask,
        tile_size=int(manifest["tile"]["size"]),
        stride=int(manifest["tile"]["stride"]),
        valid_fraction_threshold=float(manifest["tile"]["valid_fraction_threshold"]),
    )

    sample_records = load_dataset_records(sample.sample_dir)
    blank_records = load_dataset_records(sample.blank_dir)
    empty_records = load_dataset_records(sample.empty_dir)
    dark_records = load_dataset_records(sample.dark_dir)

    tile_count = len(tiles)
    sample_tile_means = np.full((len(MODE_SEQUENCE), len(ANGLES_DEG), tile_count, len(FIT_CHANNELS)), np.nan, dtype=np.float32)
    blank_tile_means = np.full_like(sample_tile_means, np.nan)
    sample_union_means = np.full((len(MODE_SEQUENCE), len(ANGLES_DEG), len(QC_CHANNELS)), np.nan, dtype=np.float32)
    blank_union_means = np.full_like(sample_union_means, np.nan)
    empty_union_means = np.full_like(sample_union_means, np.nan)
    dark_union_raw = np.full_like(sample_union_means, np.nan)
    saturation_fraction = np.full((len(MODE_SEQUENCE), len(ANGLES_DEG), len(RAW_CHANNELS)), np.nan, dtype=np.float32)
    registration_rows: list[dict[str, float]] = []
    reference_patch = None
    qc_curves = build_qc_curves_template()

    for mode_index, mode in enumerate(MODE_SEQUENCE):
        for angle_index, angle_deg in enumerate(ANGLES_DEG):
            sample_frame = sample_records[mode][angle_deg]
            blank_frame = blank_records[mode][angle_deg]
            empty_frame = empty_records[mode][angle_deg]
            dark_split = get_dark_split(dark_records, mode, angle_deg, sample_frame.exposure_us)
            sample_split = get_raw_split(sample_frame.path)
            blank_split = get_raw_split(blank_frame.path)
            empty_split = get_raw_split(empty_frame.path)

            corrected_sample: dict[str, np.ndarray] = {}
            corrected_blank: dict[str, np.ndarray] = {}
            corrected_empty: dict[str, np.ndarray] = {}
            for raw_channel_index, channel in enumerate(RAW_CHANNELS):
                dark_plane = dark_split[channel]
                saturation_fraction[mode_index, angle_index, raw_channel_index] = float(
                    np.mean(sample_split[channel][union_mask_valid] >= SATURATION_LEVEL)
                )
                dark_union_raw[mode_index, angle_index, raw_channel_index] = masked_mean(dark_plane, union_mask_valid)
                corrected_sample[channel] = derotate_and_shift(
                    (sample_split[channel] - dark_plane) / sample_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                    manifest,
                )
                corrected_blank[channel] = derotate_and_shift(
                    (blank_split[channel] - dark_plane) / blank_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                    manifest,
                )
                corrected_empty[channel] = derotate_and_shift(
                    (empty_split[channel] - dark_plane) / empty_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                    manifest,
                )

            corrected_sample["G"] = 0.5 * (corrected_sample["G1"] + corrected_sample["G2"])
            corrected_blank["G"] = 0.5 * (corrected_blank["G1"] + corrected_blank["G2"])
            corrected_empty["G"] = 0.5 * (corrected_empty["G1"] + corrected_empty["G2"])
            dark_union_raw[mode_index, angle_index, QC_CHANNELS.index("G")] = 0.5 * (
                dark_union_raw[mode_index, angle_index, QC_CHANNELS.index("G1")]
                + dark_union_raw[mode_index, angle_index, QC_CHANNELS.index("G2")]
            )

            for channel_index, channel in enumerate(QC_CHANNELS):
                qc_curves["sample_corr"][mode][channel][angle_index] = masked_mean(corrected_sample[channel], union_mask_valid)
                qc_curves["blank_corr"][mode][channel][angle_index] = masked_mean(corrected_blank[channel], union_mask_valid)
                qc_curves["empty_corr"][mode][channel][angle_index] = masked_mean(corrected_empty[channel], union_mask_valid)
                qc_curves["dark_raw"][mode][channel][angle_index] = float(dark_union_raw[mode_index, angle_index, channel_index])
                sample_union_means[mode_index, angle_index, channel_index] = qc_curves["sample_corr"][mode][channel][angle_index]
                blank_union_means[mode_index, angle_index, channel_index] = qc_curves["blank_corr"][mode][channel][angle_index]
                empty_union_means[mode_index, angle_index, channel_index] = qc_curves["empty_corr"][mode][channel][angle_index]

            fit_sample_planes = {channel: corrected_sample[channel] for channel in FIT_CHANNELS}
            fit_blank_planes = {channel: corrected_blank[channel] for channel in FIT_CHANNELS}
            for tile_index, tile in enumerate(tiles):
                mask = tile_masks[tile_index]
                y0 = tile.y
                y1 = tile.y + tile.height
                x0 = tile.x
                x1 = tile.x + tile.width
                for channel_index, channel in enumerate(FIT_CHANNELS):
                    sample_tile_means[mode_index, angle_index, tile_index, channel_index] = masked_mean(
                        fit_sample_planes[channel][y0:y1, x0:x1],
                        mask,
                    )
                    blank_tile_means[mode_index, angle_index, tile_index, channel_index] = masked_mean(
                        fit_blank_planes[channel][y0:y1, x0:x1],
                        mask,
                    )

            if mode == "PPL":
                patch = gradient_magnitude(corrected_sample["G"][bbox_y0:bbox_y1, bbox_x0:bbox_x1])
                if angle_deg == 0:
                    reference_patch = patch.copy()
                else:
                    assert reference_patch is not None
                    shift_x, shift_y, corr_peak = phase_correlation_shift(reference_patch, patch, bbox_mask)
                    registration_rows.append(
                        {
                            "angle_deg": float(angle_deg),
                            "shift_x_px": shift_x,
                            "shift_y_px": shift_y,
                            "shift_magnitude_px": float(np.hypot(shift_x, shift_y)),
                            "corr_peak": corr_peak,
                        }
                    )

    blank_ppl_medians = np.array(
        [nanmedian(blank_union_means[MODE_SEQUENCE.index("PPL"), :, QC_CHANNELS.index(channel)]) for channel in FIT_CHANNELS],
        dtype=np.float64,
    )
    eps_count = 0.005 * blank_ppl_medians
    tau_low = 0.02 * blank_ppl_medians
    signal_values = np.full((tile_count, len(ANGLES_DEG), len(SIGNAL_NAMES), len(FIT_CHANNELS)), np.nan, dtype=np.float32)
    signal_valid = np.zeros((tile_count, len(ANGLES_DEG), len(SIGNAL_NAMES), len(FIT_CHANNELS)), dtype=bool)

    for tile_index in range(tile_count):
        for angle_index in range(len(ANGLES_DEG)):
            for channel_index in range(len(FIT_CHANNELS)):
                sample_ppl = float(sample_tile_means[MODE_SEQUENCE.index("PPL"), angle_index, tile_index, channel_index])
                sample_xpl = float(sample_tile_means[MODE_SEQUENCE.index("XPL"), angle_index, tile_index, channel_index])
                blank_ppl = float(blank_tile_means[MODE_SEQUENCE.index("PPL"), angle_index, tile_index, channel_index])
                blank_xpl = float(blank_tile_means[MODE_SEQUENCE.index("XPL"), angle_index, tile_index, channel_index])
                scalar_values, scalar_valid = compute_signal_values_for_scalar(
                    sample_ppl=sample_ppl,
                    sample_xpl=sample_xpl,
                    blank_ppl=blank_ppl,
                    blank_xpl=blank_xpl,
                    eps_count=float(eps_count[channel_index]),
                    tau_low=float(tau_low[channel_index]),
                )
                for signal_name in SIGNAL_NAMES:
                    signal_index = SIGNAL_NAMES.index(signal_name)
                    signal_values[tile_index, angle_index, signal_index, channel_index] = scalar_values[signal_name]
                    signal_valid[tile_index, angle_index, signal_index, channel_index] = scalar_valid[signal_name]

    tile_roi_index = np.array([tile.roi_index for tile in tiles], dtype=np.int32)
    roi_signal_means, roi_signal_valid_counts = compute_roi_signal_means(
        signal_values=signal_values,
        signal_valid=signal_valid,
        tile_roi_index=tile_roi_index,
        roi_count=len(roi_entries),
    )

    blank_scale = np.maximum(blank_ppl_medians, 1e-12)
    roi_deltae_xpl_vs_ppl = np.full((len(roi_entries), len(ANGLES_DEG)), np.nan, dtype=np.float32)
    roi_deltae_angle_vs_zero = np.full_like(roi_deltae_xpl_vs_ppl, np.nan)
    roi_deltae_sample_vs_blank = np.full_like(roi_deltae_xpl_vs_ppl, np.nan)
    for roi_index in range(len(roi_entries)):
        tile_mask = tile_roi_index == roi_index
        if not np.any(tile_mask):
            continue
        tile_indices = np.flatnonzero(tile_mask)
        sample_ppl_rgb = np.array(
            [
                np.nanmean(sample_tile_means[MODE_SEQUENCE.index("PPL"), angle_index, tile_indices, :], axis=0)
                for angle_index in range(len(ANGLES_DEG))
            ],
            dtype=np.float32,
        )
        sample_xpl_rgb = np.array(
            [
                np.nanmean(sample_tile_means[MODE_SEQUENCE.index("XPL"), angle_index, tile_indices, :], axis=0)
                for angle_index in range(len(ANGLES_DEG))
            ],
            dtype=np.float32,
        )
        blank_ppl_rgb = np.array(
            [
                np.nanmean(blank_tile_means[MODE_SEQUENCE.index("PPL"), angle_index, tile_indices, :], axis=0)
                for angle_index in range(len(ANGLES_DEG))
            ],
            dtype=np.float32,
        )
        reference_lab = linear_rgb_to_lab(sample_ppl_rgb[0] / blank_scale)
        for angle_index in range(len(ANGLES_DEG)):
            ppl_lab = linear_rgb_to_lab(sample_ppl_rgb[angle_index] / blank_scale)
            xpl_lab = linear_rgb_to_lab(sample_xpl_rgb[angle_index] / blank_scale)
            blank_lab = linear_rgb_to_lab(blank_ppl_rgb[angle_index] / blank_scale)
            roi_deltae_xpl_vs_ppl[roi_index, angle_index] = delta_e76(xpl_lab, ppl_lab)
            roi_deltae_angle_vs_zero[roi_index, angle_index] = delta_e76(ppl_lab, reference_lab)
            roi_deltae_sample_vs_blank[roi_index, angle_index] = delta_e76(ppl_lab, blank_lab)

    g1_blank = np.array(qc_curves["blank_corr"]["PPL"]["G1"] + qc_curves["blank_corr"]["XPL"]["G1"], dtype=np.float64)
    g2_blank = np.array(qc_curves["blank_corr"]["PPL"]["G2"] + qc_curves["blank_corr"]["XPL"]["G2"], dtype=np.float64)
    g1_g2_curve_diff = float(np.median(np.abs(g1_blank - g2_blank) / (0.5 * (g1_blank + g2_blank) + eps_count[0])))
    blank_flatness = compute_blank_flatness_metrics(
        {
            "PPL": qc_curves["blank_corr"]["PPL"]["G"],
            "XPL": qc_curves["blank_corr"]["XPL"]["G"],
        }
    )
    registration_magnitudes = np.array([row["shift_magnitude_px"] for row in registration_rows], dtype=np.float64)
    registration_median_shift = nanmedian(registration_magnitudes)
    registration_flag = bool(np.isfinite(registration_median_shift) and registration_median_shift <= 0.5)
    near_black_fraction = {
        channel: float(np.mean(sample_tile_means[:, :, :, channel_index] < tau_low[channel_index]))
        for channel_index, channel in enumerate(FIT_CHANNELS)
    }
    saturation_summary = {
        mode: {
            channel: float(np.nanmedian(saturation_fraction[mode_index, :, channel_index]))
            for channel_index, channel in enumerate(RAW_CHANNELS)
        }
        for mode_index, mode in enumerate(MODE_SEQUENCE)
    }

    qc_metrics = {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "roi_size_fullres": sample.roi_size_fullres,
        "roi_labels": [roi.label for roi in roi_entries],
        "eps_count": {channel: float(eps_count[index]) for index, channel in enumerate(FIT_CHANNELS)},
        "tau_low": {channel: float(tau_low[index]) for index, channel in enumerate(FIT_CHANNELS)},
        "G1_G2_curve_diff": g1_g2_curve_diff,
        "blank_flatness_flag": bool(blank_flatness["flag"]),
        "blank_flatness_details": blank_flatness["details"],
        "registration_stability_flag": registration_flag,
        "registration_stability_median_shift_px": registration_median_shift,
        "registration_rows": registration_rows,
        "near_black_fraction": near_black_fraction,
        "saturation_fraction_median": saturation_summary,
        "qc_curves": qc_curves,
        "tile_count": tile_count,
    }

    plot_sample_qc(
        sample=sample,
        angles_deg=np.asarray(ANGLES_DEG, dtype=np.float64),
        qc_curves=qc_curves,
        roi_labels=[roi.label for roi in roi_entries],
        roi_signal_means_g=roi_signal_means[:, :, :, FIT_CHANNELS.index("G")],
        output_path=qc_plot_path,
    )
    save_json(qc_json_path, qc_metrics)

    roi_rects_fullres = np.array([roi.full_rect for roi in roi_entries], dtype=np.int32)
    roi_rects_halfres = np.array([roi.half_rect for roi in roi_entries], dtype=np.int32)
    tile_bounds_halfres = np.array([[tile.x, tile.y, tile.width, tile.height] for tile in tiles], dtype=np.int32)
    tile_bounds_fullres = tile_bounds_halfres.copy()
    tile_bounds_fullres[:, 0] *= 2
    tile_bounds_fullres[:, 1] *= 2
    tile_bounds_fullres[:, 2] *= 2
    tile_bounds_fullres[:, 3] *= 2

    np.savez_compressed(
        bundle_path,
        angles_deg=np.asarray(ANGLES_DEG, dtype=np.int32),
        signal_values=signal_values.astype(np.float32),
        signal_valid=signal_valid.astype(np.bool_),
        roi_signal_means=roi_signal_means.astype(np.float32),
        roi_signal_valid_counts=roi_signal_valid_counts.astype(np.int32),
        roi_deltae_xpl_vs_ppl=roi_deltae_xpl_vs_ppl.astype(np.float32),
        roi_deltae_angle_vs_zero=roi_deltae_angle_vs_zero.astype(np.float32),
        roi_deltae_sample_vs_blank=roi_deltae_sample_vs_blank.astype(np.float32),
        tile_roi_index=tile_roi_index.astype(np.int32),
        tile_bounds_halfres=tile_bounds_halfres,
        tile_bounds_fullres=tile_bounds_fullres,
        tile_valid_fraction=np.array([tile.valid_fraction for tile in tiles], dtype=np.float32),
        roi_labels=np.asarray([roi.label for roi in roi_entries]),
        roi_rects_fullres=roi_rects_fullres,
        roi_rects_halfres=roi_rects_halfres,
        channel_names=np.asarray(FIT_CHANNELS),
        signal_names=np.asarray(SIGNAL_NAMES),
    )

    metadata = {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "sample_dir": str(sample.sample_dir),
        "blank_dir": str(sample.blank_dir),
        "empty_dir": str(sample.empty_dir),
        "dark_dir": str(sample.dark_dir),
        "roi_size_fullres": sample.roi_size_fullres,
        "roi_labels": [roi.label for roi in roi_entries],
        "bundle_path": str(bundle_path),
        "qc_plot_path": str(qc_plot_path),
        "qc_json_path": str(qc_json_path),
    }
    save_json(metadata_path, metadata)
    return bundle_path, metadata_path


def load_extraction_bundle(bundle_path: Path) -> dict[str, Any]:
    data = np.load(bundle_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def build_primary_metric_row(
    sample: SampleSpec,
    signal_name: str,
    channel_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_rows = [row for row in rows if np.isfinite(float(row["primary_value"]))]
    primary_values = np.array([float(row["primary_value"]) for row in valid_rows], dtype=np.float64)
    axis_values = np.array([float(row["primary_axis_deg"]) for row in valid_rows], dtype=np.float64)
    return {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "roi_size_fullres": sample.roi_size_fullres,
        "signal": signal_name,
        "channel": channel_name,
        "primary_metric_name": primary_metric_name(signal_name),
        "primary_value": nanmedian(primary_values),
        "primary_axis_deg": circular_mean_deg(axis_values, primary_period_deg(signal_name)),
        "mean_value": nanmean(primary_values),
        "p90_value": nanpercentile(primary_values, 90.0),
    }


def run_fit_for_sample(sample: SampleSpec, bundle_path: Path, metadata_path: Path, output_root: Path, force: bool) -> tuple[Path, Path, Path]:
    fits_dir = ensure_dir(output_root / "fits")
    tile_fits_path = fits_dir / f"{sample.sample_id}_tile_fits.csv"
    roi_summary_path = fits_dir / f"{sample.sample_id}_roi_summary.csv"
    sample_summary_path = fits_dir / f"{sample.sample_id}_sample_summary.json"
    overlay_path = fits_dir / f"{sample.sample_id}_fit_overlay.png"
    if tile_fits_path.exists() and roi_summary_path.exists() and sample_summary_path.exists() and overlay_path.exists() and not force:
        return tile_fits_path, roi_summary_path, sample_summary_path

    bundle = load_extraction_bundle(bundle_path)
    signal_values = bundle["signal_values"]
    signal_valid = bundle["signal_valid"].astype(bool)
    roi_signal_means = bundle["roi_signal_means"]
    roi_signal_valid_counts = bundle["roi_signal_valid_counts"]
    tile_roi_index = bundle["tile_roi_index"]
    tile_bounds_halfres = bundle["tile_bounds_halfres"]
    tile_valid_fraction = bundle["tile_valid_fraction"]
    roi_labels = [str(value) for value in bundle["roi_labels"].tolist()]

    fit_rows: list[dict[str, Any]] = []
    for tile_index in range(signal_values.shape[0]):
        roi_label = roi_labels[int(tile_roi_index[tile_index])]
        x, y, width, height = [int(value) for value in tile_bounds_halfres[tile_index]]
        for signal_index, signal_name in enumerate(SIGNAL_NAMES):
            for channel_index, channel_name in enumerate(FIT_CHANNELS):
                fit = fit_harmonic_curve(
                    signal_values[tile_index, :, signal_index, channel_index],
                    signal_valid[tile_index, :, signal_index, channel_index],
                )
                fit_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_type": sample.sample_type,
                        "roi_size_fullres": sample.roi_size_fullres,
                        "tile_id": int(tile_index),
                        "roi_label": roi_label,
                        "tile_x_halfres": x,
                        "tile_y_halfres": y,
                        "tile_width_halfres": width,
                        "tile_height_halfres": height,
                        "tile_valid_fraction": float(tile_valid_fraction[tile_index]),
                        "signal": signal_name,
                        "channel": channel_name,
                        "fit_valid": bool(fit["fit_valid"]),
                        "n_valid_angles": int(fit["n_valid_angles"]),
                        "a0": fit["a0"],
                        "a2c": fit["a2c"],
                        "a2s": fit["a2s"],
                        "a4c": fit["a4c"],
                        "a4s": fit["a4s"],
                        "A2": fit["A2"],
                        "A4": fit["A4"],
                        "phase2_raw_deg": fit["phase2_raw_deg"],
                        "phase4_raw_deg": fit["phase4_raw_deg"],
                        "axis2_deg": fit["axis2_deg"],
                        "axis4_deg": fit["axis4_deg"],
                        "rmse": fit["rmse"],
                        "nrmse": fit["nrmse"],
                    }
                )

    write_csv_rows(tile_fits_path, fit_rows)

    roi_summary_rows: list[dict[str, Any]] = []
    for roi_label in roi_labels:
        for signal_name in SIGNAL_NAMES:
            for channel_name in FIT_CHANNELS:
                rows = [
                    row
                    for row in fit_rows
                    if row["roi_label"] == roi_label and row["signal"] == signal_name and row["channel"] == channel_name
                ]
                valid_rows = [row for row in rows if row["fit_valid"]]
                metric_name = primary_metric_name(signal_name)
                axis_name = primary_axis_field(signal_name)
                primary_values = np.array([row[metric_name] for row in valid_rows], dtype=np.float64)
                primary_axes = np.array([row[axis_name] for row in valid_rows], dtype=np.float64)
                roi_summary_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_type": sample.sample_type,
                        "roi_size_fullres": sample.roi_size_fullres,
                        "roi_label": roi_label,
                        "signal": signal_name,
                        "channel": channel_name,
                        "primary_metric_name": metric_name,
                        "primary_value": nanmedian(primary_values),
                        "primary_axis_deg": circular_mean_deg(primary_axes, primary_period_deg(signal_name)),
                        "mean_value": nanmean(primary_values),
                        "p90_value": nanpercentile(primary_values, 90.0),
                        "valid_tile_fraction": float(len(valid_rows) / max(len(rows), 1)),
                        "valid_tile_count": int(len(valid_rows)),
                        "total_tile_count": int(len(rows)),
                        "median_rmse": nanmedian(np.array([row["rmse"] for row in valid_rows], dtype=np.float64)),
                        "median_nrmse": nanmedian(np.array([row["nrmse"] for row in valid_rows], dtype=np.float64)),
                    }
                )
    write_csv_rows(roi_summary_path, roi_summary_rows)

    sample_summary_rows: list[dict[str, Any]] = []
    for signal_name in SIGNAL_NAMES:
        for channel_name in FIT_CHANNELS:
            rows = [row for row in roi_summary_rows if row["signal"] == signal_name and row["channel"] == channel_name]
            sample_summary_rows.append(build_primary_metric_row(sample, signal_name, channel_name, rows))

    metadata = load_json(metadata_path)
    qc_metrics = load_json(Path(metadata["qc_json_path"]))
    g_tile_valid_all = []
    for tile_id in range(signal_values.shape[0]):
        tile_rows = [
            row
            for row in fit_rows
            if row["tile_id"] == tile_id and row["channel"] == "G" and row["signal"] in SIGNAL_NAMES
        ]
        g_tile_valid_all.append(all(row["fit_valid"] for row in tile_rows))

    final_row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "roi_size_fullres": sample.roi_size_fullres,
        "roi_labels": ";".join(roi_labels),
        "valid_tile_fraction_G": float(np.mean(g_tile_valid_all)),
        "median_RMSE_Afilm_G": nanmedian(
            np.array([row["rmse"] for row in fit_rows if row["signal"] == "Afilm_PPL" and row["channel"] == "G" and row["fit_valid"]], dtype=np.float64)
        ),
        "median_RMSE_Xfilm_G": nanmedian(
            np.array([row["rmse"] for row in fit_rows if row["signal"] == "Xfilm" and row["channel"] == "G" and row["fit_valid"]], dtype=np.float64)
        ),
        "median_NRMSE_Afilm_G": nanmedian(
            np.array([row["nrmse"] for row in fit_rows if row["signal"] == "Afilm_PPL" and row["channel"] == "G" and row["fit_valid"]], dtype=np.float64)
        ),
        "median_NRMSE_Xfilm_G": nanmedian(
            np.array([row["nrmse"] for row in fit_rows if row["signal"] == "Xfilm" and row["channel"] == "G" and row["fit_valid"]], dtype=np.float64)
        ),
        "G1_G2_curve_diff": qc_metrics["G1_G2_curve_diff"],
        "blank_flatness_flag": bool(qc_metrics["blank_flatness_flag"]),
        "registration_stability_flag": bool(qc_metrics["registration_stability_flag"]),
        "registration_stability_median_shift_px": qc_metrics["registration_stability_median_shift_px"],
    }

    key_map = {
        "Afilm_PPL": ("A2_Afilm_G_primary", "axis2_Afilm_G_primary"),
        "Xfilm": ("A4_Xfilm_G_primary", "axis4_Xfilm_G_primary"),
        "Xnorm_sample": ("A4_Xnorm_sample_G_primary", "axis4_Xnorm_sample_G_primary"),
        "Xnorm_blank": ("A4_Xnorm_blank_G_primary", "axis4_Xnorm_blank_G_primary"),
    }
    for row in sample_summary_rows:
        if row["channel"] != "G":
            continue
        signal_name = row["signal"]
        metric_key, axis_key = key_map[signal_name]
        final_row[metric_key] = row["primary_value"]
        final_row[axis_key] = row["primary_axis_deg"]

    summary = {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "roi_size_fullres": sample.roi_size_fullres,
        "roi_labels": roi_labels,
        "tile_fit_csv": str(tile_fits_path),
        "roi_summary_csv": str(roi_summary_path),
        "overlay_plot": str(overlay_path),
        "sample_summary_rows": sample_summary_rows,
        "final_row": final_row,
        "qc_metrics": qc_metrics,
    }
    save_json(sample_summary_path, summary)
    plot_fit_overlays(
        sample=sample,
        roi_labels=roi_labels,
        roi_signal_means_g=roi_signal_means[:, :, :, FIT_CHANNELS.index("G")],
        roi_signal_valid_counts_g=roi_signal_valid_counts[:, :, :, FIT_CHANNELS.index("G")],
        output_path=overlay_path,
    )
    return tile_fits_path, roi_summary_path, sample_summary_path


def run_roundtrip_qc(manifest: dict[str, Any], samples: list[SampleSpec], output_root: Path, force: bool) -> Path:
    qc_dir = ensure_dir(output_root / "qc")
    roundtrip_json_path = qc_dir / "interpolation_roundtrip_qc.json"
    roundtrip_plot_path = qc_dir / "interpolation_roundtrip_qc.png"
    if roundtrip_json_path.exists() and roundtrip_plot_path.exists() and not force:
        return roundtrip_json_path

    half_valid_mask = load_halfres_valid_mask(Path(manifest["rotation_valid_mask"]))
    center_half = get_center_halfres(manifest)
    sample_by_type = {}
    for sample in samples:
        sample_by_type.setdefault(sample.sample_type, sample)

    rows: list[dict[str, Any]] = []
    for sample_type, sample in sorted(sample_by_type.items()):
        blank_records = load_dataset_records(sample.blank_dir)
        dark_records = load_dataset_records(sample.dark_dir)
        for mode in MODE_SEQUENCE:
            blank_frame = blank_records[mode][0]
            dark_split = get_dark_split(dark_records, mode, 0, blank_frame.exposure_us)
            blank_split = get_raw_split(blank_frame.path)
            blank_g = rotate_plane(
                ((blank_split["G1"] - dark_split["G1"]) + (blank_split["G2"] - dark_split["G2"])) / (2.0 * blank_frame.exposure_us),
                0.0,
                center_half,
            )
            for metric in compute_roundtrip_metrics(
                plane=blank_g,
                valid_mask=half_valid_mask,
                tile_size=int(manifest["tile"]["size"]),
                stride=int(manifest["tile"]["stride"]),
                center_xy=center_half,
            ):
                rows.append({"dataset_id": f"blank_{sample_type}", "mode": mode, **metric})

    plot_roundtrip_qc(rows, roundtrip_plot_path)
    save_json(roundtrip_json_path, {"rows": rows, "plot_path": str(roundtrip_plot_path)})
    return roundtrip_json_path


def run_qc_stage(manifest: dict[str, Any], samples: list[SampleSpec], output_root: Path, force: bool) -> None:
    roundtrip_json = run_roundtrip_qc(manifest, samples, output_root, force)
    summary_rows = []
    for sample in samples:
        _, metadata_path = extract_sample_bundle(sample, manifest, output_root, force)
        metadata = load_json(metadata_path)
        qc_metrics = load_json(Path(metadata["qc_json_path"]))
        summary_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_type": sample.sample_type,
                "roi_size_fullres": sample.roi_size_fullres,
                "G1_G2_curve_diff": qc_metrics["G1_G2_curve_diff"],
                "blank_flatness_flag": qc_metrics["blank_flatness_flag"],
                "registration_stability_flag": qc_metrics["registration_stability_flag"],
                "registration_stability_median_shift_px": qc_metrics["registration_stability_median_shift_px"],
            }
        )
    write_csv_rows(output_root / "qc" / "qc_summary.csv", summary_rows)
    print(f"QC outputs written to {output_root / 'qc'}")
    print(f"Round-trip QC: {roundtrip_json}")


def run_extract_stage(manifest: dict[str, Any], samples: list[SampleSpec], output_root: Path, force: bool) -> None:
    summary_rows = []
    for sample in samples:
        bundle_path, metadata_path = extract_sample_bundle(sample, manifest, output_root, force)
        summary_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_type": sample.sample_type,
                "roi_size_fullres": sample.roi_size_fullres,
                "bundle_path": str(bundle_path),
                "metadata_path": str(metadata_path),
            }
        )
    write_csv_rows(output_root / "signals" / "extraction_index.csv", summary_rows)
    print(f"Signal bundles written to {output_root / 'signals'}")


def run_fit_stage(manifest: dict[str, Any], samples: list[SampleSpec], output_root: Path, force: bool) -> None:
    fit_index_rows = []
    for sample in samples:
        bundle_path, metadata_path = extract_sample_bundle(sample, manifest, output_root, force=False)
        tile_csv, roi_csv, summary_json = run_fit_for_sample(sample, bundle_path, metadata_path, output_root, force)
        fit_index_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_type": sample.sample_type,
                "roi_size_fullres": sample.roi_size_fullres,
                "tile_fit_csv": str(tile_csv),
                "roi_summary_csv": str(roi_csv),
                "sample_summary_json": str(summary_json),
            }
        )
    write_csv_rows(output_root / "fits" / "fit_index.csv", fit_index_rows)
    print(f"Fit outputs written to {output_root / 'fits'}")


def run_summarize_stage(manifest: dict[str, Any], samples: list[SampleSpec], output_root: Path, force: bool) -> Path:
    run_fit_stage(manifest, samples, output_root, force=False)
    summary_dir = ensure_dir(output_root / "summary")
    final_summary_path = summary_dir / "final_summary.csv"
    secondary_summary_path = summary_dir / "secondary_summary.csv"
    final_json_path = summary_dir / "final_summary.json"
    primary_rows = []
    secondary_rows = []
    for sample in samples:
        sample_summary_path = output_root / "fits" / f"{sample.sample_id}_sample_summary.json"
        summary = load_json(sample_summary_path)
        primary_rows.append(summary["final_row"])
        secondary_rows.extend(summary["sample_summary_rows"])
    write_csv_rows(final_summary_path, primary_rows)
    write_csv_rows(secondary_summary_path, secondary_rows)
    save_json(final_json_path, {"rows": primary_rows, "secondary_rows": secondary_rows})
    print(f"Final summary written to {final_summary_path}")
    return final_summary_path


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = []
    for _ in range(1000):
        indices = rng.integers(0, len(x), len(x))
        sample_x = x[indices]
        sample_y = y[indices]
        if len(np.unique(sample_x)) < 2 or len(np.unique(sample_y)) < 2:
            continue
        if method == "pearson":
            values.append(float(stats.pearsonr(sample_x, sample_y).statistic))
        else:
            values.append(float(stats.spearmanr(sample_x, sample_y).statistic))
    if not values:
        return float("nan"), float("nan")
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def run_compare_mm_stage(
    manifest: dict[str, Any],
    samples: list[SampleSpec],
    output_root: Path,
    mm_csv: Path | None,
    force: bool,
) -> None:
    if mm_csv is None:
        raise ValueError("--mm-csv is required for compare-mm stage")

    final_summary_path = run_summarize_stage(manifest, samples, output_root, force=False)
    with final_summary_path.open("r", newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with mm_csv.open("r", newline="", encoding="utf-8") as handle:
        mm_rows = list(csv.DictReader(handle))

    mm_by_sample = {row["sample_id"]: row for row in mm_rows}
    comparisons = [
        ("LD_mean", "A2_Afilm_G_primary"),
        ("LB_mean", "A4_Xfilm_G_primary"),
        ("LB_mean", "A4_Xnorm_sample_G_primary"),
        ("LB_mean", "A4_Xnorm_blank_G_primary"),
    ]

    report_rows = []
    for mm_column, analysis_column in comparisons:
        paired = []
        for row in summary_rows:
            mm_row = mm_by_sample.get(row["sample_id"])
            if mm_row is None:
                continue
            if row.get(analysis_column, "") == "" or mm_row.get(mm_column, "") == "":
                continue
            paired.append((float(row[analysis_column]), float(mm_row[mm_column])))
        if len(paired) < 3:
            continue
        x = np.array([item[0] for item in paired], dtype=np.float64)
        y = np.array([item[1] for item in paired], dtype=np.float64)
        pearson = float(stats.pearsonr(x, y).statistic)
        spearman = float(stats.spearmanr(x, y).statistic)
        pearson_ci = bootstrap_corr_ci(x, y, "pearson")
        spearman_ci = bootstrap_corr_ci(x, y, "spearman")
        report_rows.append(
            {
                "mm_column": mm_column,
                "analysis_column": analysis_column,
                "sample_count": int(len(paired)),
                "pearson_r": pearson,
                "pearson_ci_low": pearson_ci[0],
                "pearson_ci_high": pearson_ci[1],
                "spearman_rho": spearman,
                "spearman_ci_low": spearman_ci[0],
                "spearman_ci_high": spearman_ci[1],
                "note": "n=13 scale is small; interpret with caution",
            }
        )

    compare_dir = ensure_dir(output_root / "compare_mm")
    write_csv_rows(compare_dir / "mm_comparison.csv", report_rows)
    save_json(compare_dir / "mm_comparison.json", {"rows": report_rows, "mm_csv": str(mm_csv)})
    print(f"Mueller comparison written to {compare_dir}")


def main() -> None:
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    manifest_path = resolve_repo_path(args.manifest)
    manifest, samples = load_manifest(manifest_path)
    selected_samples = select_samples(samples, args.samples)

    if args.stage == "qc":
        run_qc_stage(manifest, selected_samples, output_root, args.force)
    elif args.stage == "extract":
        run_extract_stage(manifest, selected_samples, output_root, args.force)
    elif args.stage == "fit":
        run_fit_stage(manifest, selected_samples, output_root, args.force)
    elif args.stage == "summarize":
        run_summarize_stage(manifest, selected_samples, output_root, args.force)
    elif args.stage == "compare-mm":
        mm_csv = resolve_repo_path(args.mm_csv) if args.mm_csv is not None else None
        run_compare_mm_stage(manifest, selected_samples, output_root, mm_csv, args.force)
    else:
        raise ValueError(f"Unsupported stage {args.stage}")


if __name__ == "__main__":
    main()
