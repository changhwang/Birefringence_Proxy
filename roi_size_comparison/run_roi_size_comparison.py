"""ROI size sensitivity comparison for shortframe samples (s9-s13).

Compares 150/200/300 px ROIs across UC/CC/LC positions, computing signal
metrics for G/R/B channels and size-dependent QC metrics.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_analysis_pipeline as rp

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

GRID_PRESET = REPO_ROOT / "configs" / "roi_presets" / "refined_grid_3x3_v1.json"
MANIFEST_PATH = SCRIPT_DIR.parent / "shortframe_calibration" / "shortframe_manifest.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

ROI_SIZES = (150, 200, 300)
CHANNELS = ("G", "R", "B")
SIGNALS = ("Afilm_PPL", "Xfilm", "Xnorm_blank")
TILE_SIZE_HALFRES = 32
TILE_STRIDE_HALFRES = 16
TILE_VALID_THRESHOLD = 0.9

DELTA_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# ROI resolution from grid preset
# ---------------------------------------------------------------------------

def resolve_grid_rois(preset: dict[str, Any], sample_id: str, size: int) -> list[rp.ROIEntry]:
    size_spec = preset["sizes"][str(size)]
    default = dict(size_spec["default_short"])
    override = dict(size_spec.get("sample_overrides", {}).get(sample_id, {}))

    center_x = int(override.get("center_x", default["center_x"]))
    upper_y = int(override.get("upper_y", default["upper_y"]))
    center_y = int(override.get("center_y", default["center_y"]))
    lower_y = int(override.get("lower_y", default["lower_y"]))
    all_dx = int(override.get("all_dx", 0))
    center_x += all_dx

    entries: list[rp.ROIEntry] = []
    for label, y_val in (("ROI_UC", upper_y), ("ROI_CC", center_y), ("ROI_LC", lower_y)):
        full_rect = (center_x, y_val, size, size)
        entries.append(rp.ROIEntry(label, full_rect, rp.full_rect_to_half_rect(full_rect)))
    return entries


# ---------------------------------------------------------------------------
# Manifest loader (reuses shortframe manifest)
# ---------------------------------------------------------------------------

def load_manifest() -> tuple[dict[str, Any], list[rp.SampleSpec]]:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    samples = [
        rp.SampleSpec(
            sample_id=entry["sample_id"],
            sample_type=entry["sample_type"],
            sample_dir=rp.resolve_repo_path(entry["sample_dir"]),
            blank_dir=rp.resolve_repo_path(entry["blank_dir"]),
            empty_dir=rp.resolve_repo_path(entry["empty_dir"]),
            dark_dir=rp.resolve_repo_path(entry["dark_dir"]),
            roi_size_fullres=int(entry["roi_size_fullres"]),
        )
        for entry in manifest["samples"]
    ]
    manifest["rotation_geometry"] = str(rp.resolve_repo_path(manifest["rotation_geometry"]))
    manifest["rotation_valid_mask"] = str(rp.resolve_repo_path(manifest["rotation_valid_mask"]))
    if "rotation_correction" in manifest:
        manifest["rotation_correction"] = str(rp.resolve_repo_path(manifest["rotation_correction"]))
    return manifest, samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roi_masked_mean(image: np.ndarray, roi: rp.ROIEntry, valid_mask: np.ndarray) -> float:
    x, y, w, h = roi.half_rect
    patch = image[y : y + h, x : x + w]
    mask = valid_mask[y : y + h, x : x + w]
    vals = patch[mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def compute_valid_tile_fraction(roi: rp.ROIEntry, valid_mask: np.ndarray) -> float:
    x, y, w, h = roi.half_rect
    total = 0
    valid = 0
    for ty in range(y, y + h - TILE_SIZE_HALFRES + 1, TILE_STRIDE_HALFRES):
        for tx in range(x, x + w - TILE_SIZE_HALFRES + 1, TILE_STRIDE_HALFRES):
            total += 1
            local = valid_mask[ty : ty + TILE_SIZE_HALFRES, tx : tx + TILE_SIZE_HALFRES]
            if float(local.mean()) >= TILE_VALID_THRESHOLD:
                valid += 1
    return valid / max(total, 1)


def compute_edge_margin(roi: rp.ROIEntry, img_shape: tuple[int, int]) -> int:
    x, y, w, h = roi.half_rect
    img_h, img_w = img_shape
    return min(x, y, img_w - (x + w), img_h - (y + h))


def compute_roundtrip_distortion(
    plane: np.ndarray, roi: rp.ROIEntry, valid_mask: np.ndarray, center: tuple[float, float]
) -> float:
    """Mean absolute pixel difference after rotate-then-unrotate at 90 deg."""
    test_angle = 90.0
    rotated = rp.rotate_plane(plane, test_angle, center)
    roundtrip = rp.rotate_plane(rotated, -test_angle, center)
    x, y, w, h = roi.half_rect
    mask = valid_mask[y : y + h, x : x + w]
    orig_patch = plane[y : y + h, x : x + w][mask]
    rt_patch = roundtrip[y : y + h, x : x + w][mask]
    if orig_patch.size == 0:
        return float("nan")
    return float(np.mean(np.abs(rt_patch - orig_patch)))


def compute_derot_visual_metrics(
    patches: np.ndarray, roi_mask: np.ndarray
) -> dict[str, float]:
    """MAD vs 0deg (median across angles), corr(165, 0)."""
    ref = patches[0]
    n_angles = patches.shape[0]
    mad_per_angle = np.full(n_angles, np.nan)
    for ai in range(n_angles):
        diff = np.abs(patches[ai] - ref)
        valid = diff[roi_mask]
        if valid.size > 0:
            mad_per_angle[ai] = float(valid.mean())
    idx_165 = list(rp.ANGLES_DEG).index(165)
    ref_flat = ref[roi_mask]
    p165_flat = patches[idx_165][roi_mask]
    if ref_flat.size > 0 and np.std(ref_flat) > 0 and np.std(p165_flat) > 0:
        corr_165_0 = float(np.corrcoef(ref_flat, p165_flat)[0, 1])
    else:
        corr_165_0 = float("nan")
    return {
        "MAD_vs_0deg_median": float(np.nanmedian(mad_per_angle)),
        "corr_165_0": corr_165_0,
    }


def angle_cv(curve: list[float]) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2 or abs(np.mean(finite)) < 1e-12:
        return float("nan")
    return float(np.std(finite) / abs(np.mean(finite)))


# ---------------------------------------------------------------------------
# Sample-level context (ROI-size-independent)
# ---------------------------------------------------------------------------

def compute_sample_context(
    sample: rp.SampleSpec,
    manifest: dict[str, Any],
    valid_mask: np.ndarray,
    center_half: tuple[float, float],
) -> dict[str, Any]:
    sample_records = rp.load_dataset_records(sample.sample_dir)
    blank_records = rp.load_dataset_records(sample.blank_dir)
    empty_records = rp.load_dataset_records(sample.empty_dir)
    dark_records = rp.load_dataset_records(sample.dark_dir)

    blank_ppl_g: list[float] = []
    blank_xpl_g: list[float] = []
    empty_xpl_g: list[float] = []
    g1_blank_ppl: list[float] = []
    g2_blank_ppl: list[float] = []

    ref_patch: dict[str, np.ndarray | None] = {"PPL": None, "XPL": None}
    reg_shifts: dict[str, list[float]] = {"PPL": [], "XPL": []}

    union_mask = valid_mask  # use full valid mask for context

    for mode in rp.MODE_SEQUENCE:
        for angle_deg in rp.ANGLES_DEG:
            blank_frame = blank_records[mode][angle_deg]
            empty_frame = empty_records[mode][angle_deg]
            sample_frame = sample_records[mode][angle_deg]
            dark_split = rp.get_dark_split(dark_records, mode, angle_deg, blank_frame.exposure_us)

            blank_split = rp.get_raw_split(blank_frame.path)
            blank_g1 = rp.derotate_and_shift(
                (blank_split["G1"] - dark_split["G1"]) / blank_frame.exposure_us,
                float(angle_deg), center_half, manifest,
            )
            blank_g2 = rp.derotate_and_shift(
                (blank_split["G2"] - dark_split["G2"]) / blank_frame.exposure_us,
                float(angle_deg), center_half, manifest,
            )
            blank_g = 0.5 * (blank_g1 + blank_g2)

            if mode == "PPL":
                blank_ppl_g.append(rp.masked_mean(blank_g, union_mask))
                g1_blank_ppl.append(rp.masked_mean(blank_g1, union_mask))
                g2_blank_ppl.append(rp.masked_mean(blank_g2, union_mask))
            else:
                blank_xpl_g.append(rp.masked_mean(blank_g, union_mask))

            if mode == "XPL":
                empty_split = rp.get_raw_split(empty_frame.path)
                dark_split_e = rp.get_dark_split(dark_records, mode, angle_deg, empty_frame.exposure_us)
                empty_g = 0.5 * (
                    rp.derotate_and_shift(
                        (empty_split["G1"] - dark_split_e["G1"]) / empty_frame.exposure_us,
                        float(angle_deg), center_half, manifest,
                    )
                    + rp.derotate_and_shift(
                        (empty_split["G2"] - dark_split_e["G2"]) / empty_frame.exposure_us,
                        float(angle_deg), center_half, manifest,
                    )
                )
                empty_xpl_g.append(rp.masked_mean(empty_g, union_mask))

            # registration shift
            patch = rp.gradient_magnitude(blank_g)
            if angle_deg == 0:
                ref_patch[mode] = patch.copy()
            elif ref_patch[mode] is not None:
                _, _, corr_peak = rp.phase_correlation_shift(ref_patch[mode], patch, union_mask)
                # we just record the shift magnitude
                dx, dy, _ = rp.phase_correlation_shift(ref_patch[mode], patch, union_mask)
                reg_shifts[mode].append(float(np.hypot(dx, dy)))

    g1_arr = np.asarray(g1_blank_ppl, dtype=np.float64)
    g2_arr = np.asarray(g2_blank_ppl, dtype=np.float64)
    eps = max(0.005 * rp.nanmedian(np.asarray(blank_ppl_g, dtype=np.float64)), 1e-12)
    g1g2_diff = float(np.median(np.abs(g1_arr - g2_arr) / (0.5 * (g1_arr + g2_arr) + eps)))

    return {
        "ctx_blank_ppl_cv": angle_cv(blank_ppl_g),
        "ctx_blank_xpl_cv": angle_cv(blank_xpl_g),
        "ctx_empty_xpl_cv": angle_cv(empty_xpl_g),
        "ctx_g1g2_curve_diff": g1g2_diff,
        "ctx_reg_median_shift_ppl": float(np.median(reg_shifts["PPL"])) if reg_shifts["PPL"] else float("nan"),
        "ctx_reg_median_shift_xpl": float(np.median(reg_shifts["XPL"])) if reg_shifts["XPL"] else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_sample(
    sample: rp.SampleSpec,
    manifest: dict[str, Any],
    preset: dict[str, Any],
    valid_mask: np.ndarray,
    center_half: tuple[float, float],
) -> list[dict[str, Any]]:
    print(f"  Processing {sample.sample_id} ...")

    sample_records = rp.load_dataset_records(sample.sample_dir)
    blank_records = rp.load_dataset_records(sample.blank_dir)
    dark_records = rp.load_dataset_records(sample.dark_dir)

    # Pre-resolve ROIs for all sizes
    all_rois: dict[int, list[rp.ROIEntry]] = {
        sz: resolve_grid_rois(preset, sample.sample_id, sz) for sz in ROI_SIZES
    }

    # Curve storage: [size][roi_label][mode][channel] -> list of 34 floats
    n_angles = len(rp.ANGLES_DEG)
    CurveDict = dict[str, dict[str, dict[str, list[float]]]]
    sample_curves: dict[int, CurveDict] = {}
    blank_curves: dict[int, CurveDict] = {}
    # PPL G patches for derotation visual QC: [size][roi_label] -> (34, H, W)
    ppl_g_patches: dict[int, dict[str, np.ndarray]] = {}

    for sz in ROI_SIZES:
        sample_curves[sz] = {}
        blank_curves[sz] = {}
        ppl_g_patches[sz] = {}
        for roi in all_rois[sz]:
            _, _, w, h = roi.half_rect
            hw, hh = w, h
            sample_curves[sz][roi.label] = {
                m: {ch: [float("nan")] * n_angles for ch in CHANNELS}
                for m in rp.MODE_SEQUENCE
            }
            blank_curves[sz][roi.label] = {
                m: {ch: [float("nan")] * n_angles for ch in CHANNELS}
                for m in rp.MODE_SEQUENCE
            }
            ppl_g_patches[sz][roi.label] = np.full((n_angles, hh, hw), np.nan, dtype=np.float32)

    # Main angle loop — load images once, extract per-size/roi values
    for mode in rp.MODE_SEQUENCE:
        for ai, angle_deg in enumerate(rp.ANGLES_DEG):
            sample_frame = sample_records[mode][angle_deg]
            blank_frame = blank_records[mode][angle_deg]
            dark_split = rp.get_dark_split(dark_records, mode, angle_deg, sample_frame.exposure_us)
            dark_split_b = rp.get_dark_split(dark_records, mode, angle_deg, blank_frame.exposure_us)

            sample_split = rp.get_raw_split(sample_frame.path)
            blank_split = rp.get_raw_split(blank_frame.path)

            corrected_sample: dict[str, np.ndarray] = {}
            corrected_blank: dict[str, np.ndarray] = {}
            for ch in rp.RAW_CHANNELS:
                corrected_sample[ch] = rp.derotate_and_shift(
                    (sample_split[ch] - dark_split[ch]) / sample_frame.exposure_us,
                    float(angle_deg), center_half, manifest,
                )
                corrected_blank[ch] = rp.derotate_and_shift(
                    (blank_split[ch] - dark_split_b[ch]) / blank_frame.exposure_us,
                    float(angle_deg), center_half, manifest,
                )
            corrected_sample["G"] = 0.5 * (corrected_sample["G1"] + corrected_sample["G2"])
            corrected_blank["G"] = 0.5 * (corrected_blank["G1"] + corrected_blank["G2"])

            for sz in ROI_SIZES:
                for roi in all_rois[sz]:
                    for ch in CHANNELS:
                        sample_curves[sz][roi.label][mode][ch][ai] = roi_masked_mean(
                            corrected_sample[ch], roi, valid_mask,
                        )
                        blank_curves[sz][roi.label][mode][ch][ai] = roi_masked_mean(
                            corrected_blank[ch], roi, valid_mask,
                        )
                    if mode == "PPL":
                        x, y, w, h = roi.half_rect
                        ppl_g_patches[sz][roi.label][ai] = corrected_sample["G"][y : y + h, x : x + w]

    # Compute sample-level context
    context = compute_sample_context(sample, manifest, valid_mask, center_half)

    # Compute roundtrip distortion reference plane (blank G at 0deg PPL)
    blank_frame_0 = blank_records["PPL"][0]
    dark_split_0 = rp.get_dark_split(dark_records, "PPL", 0, blank_frame_0.exposure_us)
    blank_split_0 = rp.get_raw_split(blank_frame_0.path)
    rt_plane = 0.5 * (
        (blank_split_0["G1"] - dark_split_0["G1"]) / blank_frame_0.exposure_us
        + (blank_split_0["G2"] - dark_split_0["G2"]) / blank_frame_0.exposure_us
    )

    # Build detail rows
    detail_rows: list[dict[str, Any]] = []

    for sz in ROI_SIZES:
        for roi in all_rois[sz]:
            x, y, w, h = roi.half_rect
            roi_valid = valid_mask[y : y + h, x : x + w]
            vm_coverage = float(roi_valid.mean())
            edge_margin = compute_edge_margin(roi, valid_mask.shape)
            vtf = compute_valid_tile_fraction(roi, valid_mask)
            rt_dist = compute_roundtrip_distortion(rt_plane, roi, valid_mask, center_half)

            derot_metrics = compute_derot_visual_metrics(
                ppl_g_patches[sz][roi.label], roi_valid,
            )

            for ch in CHANNELS:
                sc = sample_curves[sz][roi.label]
                bc = blank_curves[sz][roi.label]

                blank_ppl_med = rp.nanmedian(np.asarray(bc["PPL"][ch], dtype=np.float64))
                tau_low = 0.02 * blank_ppl_med
                eps_count = 0.005 * blank_ppl_med

                sig_curves = {s: [float("nan")] * n_angles for s in SIGNALS}
                sig_valid = {s: [False] * n_angles for s in SIGNALS}
                for ai2 in range(n_angles):
                    vals, vld = rp.compute_signal_values_for_scalar(
                        sample_ppl=float(sc["PPL"][ch][ai2]),
                        sample_xpl=float(sc["XPL"][ch][ai2]),
                        blank_ppl=float(bc["PPL"][ch][ai2]),
                        blank_xpl=float(bc["XPL"][ch][ai2]),
                        eps_count=eps_count,
                        tau_low=tau_low,
                    )
                    for s in SIGNALS:
                        sig_curves[s][ai2] = float(vals[s])
                        sig_valid[s][ai2] = bool(vld[s])

                sig_fits = {
                    s: rp.fit_harmonic_curve(
                        np.asarray(sig_curves[s], dtype=np.float64),
                        np.asarray(sig_valid[s], dtype=bool),
                    )
                    for s in SIGNALS
                }

                row: dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "roi_label": roi.label,
                    "roi_size": sz,
                    "channel": ch,
                    "valid_mask_coverage": vm_coverage,
                    "edge_margin_px": edge_margin,
                    "valid_tile_fraction": vtf,
                    "roundtrip_distortion": rt_dist,
                    "MAD_vs_0deg_median": derot_metrics["MAD_vs_0deg_median"],
                    "corr_165_0": derot_metrics["corr_165_0"],
                }

                for s in SIGNALS:
                    fit = sig_fits[s]
                    row[f"a0_{s}"] = fit["a0"]
                    row[f"A2_{s}"] = fit["A2"]
                    row[f"A4_{s}"] = fit["A4"]
                    row[f"axis2_{s}_deg"] = fit["axis2_deg"]
                    row[f"axis4_{s}_deg"] = fit["axis4_deg"]
                    row[f"RMSE_{s}"] = fit["rmse"]
                    row[f"NRMSE_{s}"] = fit["nrmse"]
                    if fit["fit_valid"]:
                        pred = fit["predicted"]
                        row[f"peak_to_peak_{s}"] = float(np.nanmax(pred) - np.nanmin(pred))
                    else:
                        row[f"peak_to_peak_{s}"] = float("nan")

                xf = sig_fits["Xfilm"]
                row["A2_A4_ratio_Xfilm"] = (
                    float(xf["A2"] / xf["A4"]) if xf["A4"] > 1e-15 else float("nan")
                )

                row.update(context)
                detail_rows.append(row)

    return detail_rows


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def build_stability_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import defaultdict

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (row["sample_id"], row["roi_label"], row["channel"])
        grouped[key].append(row)

    stability_rows: list[dict[str, Any]] = []
    metric_keys = [
        "A2_Afilm_PPL", "A4_Xfilm", "A4_Xnorm_blank",
        "NRMSE_Afilm_PPL", "NRMSE_Xfilm", "NRMSE_Xnorm_blank",
        "valid_mask_coverage", "valid_tile_fraction",
    ]

    for (sample_id, roi_label, channel), rows in sorted(grouped.items()):
        srow: dict[str, Any] = {
            "sample_id": sample_id,
            "roi_label": roi_label,
            "channel": channel,
        }
        for mk in metric_keys:
            values = [r[mk] for r in rows if np.isfinite(r[mk])]
            if len(values) >= 2:
                med = float(np.median(values))
                spread = max(values) - min(values)
                delta = spread / abs(med) if abs(med) > 1e-15 else float("nan")
                srow[f"{mk}_min"] = min(values)
                srow[f"{mk}_max"] = max(values)
                srow[f"{mk}_median"] = med
                srow[f"{mk}_delta"] = delta
                srow[f"{mk}_pass"] = bool(np.isfinite(delta) and delta <= DELTA_THRESHOLD)
            else:
                for sfx in ("_min", "_max", "_median", "_delta", "_pass"):
                    srow[f"{mk}{sfx}"] = float("nan") if sfx != "_pass" else False
        stability_rows.append(srow)

    return stability_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sample_comparison(
    sample_id: str, detail_rows: list[dict[str, Any]], output_path: Path,
) -> None:
    if plt is None:
        return
    g_rows = [r for r in detail_rows if r["sample_id"] == sample_id and r["channel"] == "G"]
    roi_labels = sorted(set(r["roi_label"] for r in g_rows))
    signal_names = list(SIGNALS)
    n_rois = len(roi_labels)
    n_signals = len(signal_names)

    fig, axes = plt.subplots(n_rois, n_signals, figsize=(5 * n_signals, 4 * n_rois), constrained_layout=True)
    if n_rois == 1:
        axes = axes[None, :]
    if n_signals == 1:
        axes = axes[:, None]

    colors = {150: "#1f77b4", 200: "#ff7f0e", 300: "#2ca02c"}
    angles = np.asarray(rp.ANGLES_DEG, dtype=np.float64)

    for ri, roi_label in enumerate(roi_labels):
        for si, sig in enumerate(signal_names):
            ax = axes[ri, si]
            for sz in ROI_SIZES:
                matching = [
                    r for r in g_rows
                    if r["roi_label"] == roi_label and r["roi_size"] == sz
                ]
                if not matching:
                    continue
                r = matching[0]
                a2 = r[f"A2_{sig}"]
                a4 = r[f"A4_{sig}"]
                nrmse = r[f"NRMSE_{sig}"]
                ptp = r[f"peak_to_peak_{sig}"]
                a0 = r[f"a0_{sig}"]

                # reconstruct predicted curve from fit params
                # we don't store predicted, so reconstruct from a0, a2c, a2s, a4c, a4s
                # but we don't store those individually — use A2, axis2, A4, axis4
                ax2_deg = r[f"axis2_{sig}_deg"]
                ax4_deg = r[f"axis4_{sig}_deg"]
                phase2 = np.deg2rad(ax2_deg * 2.0)
                phase4 = np.deg2rad(ax4_deg * 4.0)
                pred = a0 + a2 * np.cos(2.0 * np.deg2rad(angles) - phase2) + a4 * np.cos(4.0 * np.deg2rad(angles) - phase4)
                ax.plot(angles, pred, color=colors[sz], linewidth=1.5,
                        label=f"{sz}px  A2={a2:.4g} A4={a4:.4g}\n  a0={a0:.4g} p2p={ptp:.4g} NR={nrmse:.3g}")

            ax.set_xlabel("theta (deg)")
            ax.set_ylabel(sig)
            if ri == 0:
                ax.set_title(sig, fontsize=11)
            if si == 0:
                ax.set_ylabel(f"{roi_label}\n{sig}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="best")

    fig.suptitle(f"{sample_id} — ROI Size Comparison (G channel)", fontsize=14)
    rp.ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stability_summary(
    stability_rows: list[dict[str, Any]], output_path: Path,
) -> None:
    if plt is None:
        return
    g_rows = [r for r in stability_rows if r["channel"] == "G"]
    if not g_rows:
        return

    metrics = ["A2_Afilm_PPL", "A4_Xfilm", "A4_Xnorm_blank"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), constrained_layout=True)

    for mi, mk in enumerate(metrics):
        ax = axes[mi]
        labels = []
        deltas = []
        passes = []
        for r in g_rows:
            labels.append(f"{r['sample_id']}\n{r['roi_label']}")
            deltas.append(r[f"{mk}_delta"])
            passes.append(r[f"{mk}_pass"])
        x = np.arange(len(labels))
        colors_bar = ["#2ca02c" if p else "#d62728" for p in passes]
        ax.bar(x, deltas, color=colors_bar, alpha=0.8)
        ax.axhline(DELTA_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"threshold={DELTA_THRESHOLD:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("delta (max-min)/median")
        ax.set_title(mk)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("ROI Size Stability — G channel", fontsize=14)
    rp.ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading manifest and preset ...")
    manifest, samples = load_manifest()
    preset = rp.load_json(GRID_PRESET)
    valid_mask = rp.load_halfres_valid_mask(Path(manifest["rotation_valid_mask"]))
    center_half = rp.get_center_halfres(manifest)

    rp.ensure_dir(OUTPUT_DIR)
    all_detail_rows: list[dict[str, Any]] = []

    for sample in samples:
        rows = process_sample(sample, manifest, preset, valid_mask, center_half)
        all_detail_rows.extend(rows)

    # Write detail CSV
    detail_csv = OUTPUT_DIR / "roi_size_comparison_detail.csv"
    if all_detail_rows:
        fieldnames = list(all_detail_rows[0].keys())
        with detail_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detail_rows)
        print(f"  Detail CSV: {detail_csv.relative_to(REPO_ROOT)}")

    # Stability CSV
    stability_rows = build_stability_rows(all_detail_rows)
    stability_csv = OUTPUT_DIR / "roi_size_comparison_stability.csv"
    if stability_rows:
        fieldnames_s = list(stability_rows[0].keys())
        with stability_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_s)
            writer.writeheader()
            writer.writerows(stability_rows)
        print(f"  Stability CSV: {stability_csv.relative_to(REPO_ROOT)}")

    # Per-sample plots
    sample_ids = sorted(set(r["sample_id"] for r in all_detail_rows))
    for sid in sample_ids:
        plot_path = OUTPUT_DIR / f"{sid}_roi_size_comparison.png"
        plot_sample_comparison(sid, all_detail_rows, plot_path)
        print(f"  Plot: {plot_path.relative_to(REPO_ROOT)}")

    # Stability summary plot
    stability_plot = OUTPUT_DIR / "stability_summary.png"
    plot_stability_summary(stability_rows, stability_plot)
    print(f"  Stability plot: {stability_plot.relative_to(REPO_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
