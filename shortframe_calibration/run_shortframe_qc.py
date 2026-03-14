from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_PATH = REPO_ROOT / "scripts" / "run_analysis_pipeline.py"
DEFAULT_MANIFEST = SCRIPT_DIR / "shortframe_manifest.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
ALL_CHANNELS = ("G1", "G2", "R", "B", "G")
PHASE_B_SIGNALS = ("Afilm_PPL", "Xfilm", "Xnorm_sample", "Xnorm_blank")
VISUAL_STACK_ANGLES = (0, 45, 90, 135, 165)


def load_pipeline_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_analysis_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pipeline helper module from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rp = load_pipeline_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short-frame QC plots for QLB_s9-QLB_s13.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def load_local_manifest(path: Path) -> tuple[dict[str, Any], list[Any]]:
    manifest = rp.load_json(path)
    samples = [
        rp.SampleSpec(
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
    manifest["rotation_geometry"] = str(resolve_repo_path(manifest["rotation_geometry"]))
    manifest["rotation_valid_mask"] = str(resolve_repo_path(manifest["rotation_valid_mask"]))
    manifest["roi_spec"] = str(resolve_repo_path(manifest["roi_spec"]))
    manifest["outputs_dir"] = str(resolve_repo_path(manifest["outputs_dir"]))
    return manifest, samples


def build_roi_entries(sample_id: str, roi_spec: dict[str, Any]) -> list[Any]:
    if sample_id not in roi_spec["samples"]:
        raise KeyError(f"Sample {sample_id} missing from ROI spec")
    sample_regions = roi_spec["samples"][sample_id]["regions"]
    entries = []
    for label in roi_spec["roi_labels"]:
        region = sample_regions[label]
        full_rect = (
            int(region["x"]),
            int(region["y"]),
            int(region["width"]),
            int(region["height"]),
        )
        entries.append(rp.ROIEntry(label, full_rect, rp.full_rect_to_half_rect(full_rect)))
    return entries


def init_mode_channel_lists(channels: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    return {mode: {channel: [float("nan")] * len(rp.ANGLES_DEG) for channel in channels} for mode in rp.MODE_SEQUENCE}


def masked_fraction(condition: np.ndarray, mask: np.ndarray) -> float:
    values = condition[mask]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def median_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def fit_to_json(fit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fit.items() if key != "predicted"}


def finite_percentile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def compute_phase_payload(sample: Any, manifest: dict[str, Any], roi_spec: dict[str, Any]) -> dict[str, Any]:
    roi_entries = build_roi_entries(sample.sample_id, roi_spec)
    patch_width = roi_entries[0].half_rect[2]
    patch_height = roi_entries[0].half_rect[3]
    half_valid_mask = rp.load_halfres_valid_mask(Path(manifest["rotation_valid_mask"]))
    center_half = rp.load_rotation_center_halfres(Path(manifest["rotation_geometry"]))
    union_mask = rp.build_union_mask(half_valid_mask.shape, roi_entries) & half_valid_mask
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = rp.mask_bbox(union_mask)
    bbox_mask = union_mask[bbox_y0:bbox_y1, bbox_x0:bbox_x1]

    sample_records = rp.load_dataset_records(sample.sample_dir)
    blank_records = rp.load_dataset_records(sample.blank_dir)
    empty_records = rp.load_dataset_records(sample.empty_dir)
    dark_records = rp.load_dataset_records(sample.dark_dir)

    curves = {
        "dark_raw": init_mode_channel_lists(ALL_CHANNELS),
        "empty_corr": init_mode_channel_lists(ALL_CHANNELS),
        "blank_corr": init_mode_channel_lists(ALL_CHANNELS),
        "sample_raw_rot": init_mode_channel_lists(ALL_CHANNELS),
        "sample_corr": init_mode_channel_lists(ALL_CHANNELS),
    }
    roi_patch_stacks_ppl_g = np.full(
        (len(roi_entries), len(rp.ANGLES_DEG), patch_height, patch_width),
        np.nan,
        dtype=np.float32,
    )
    saturation_fraction = init_mode_channel_lists(rp.RAW_CHANNELS)
    registration_rows: list[dict[str, float | str]] = []
    reference_patch_by_mode: dict[str, np.ndarray] = {}

    for mode in rp.MODE_SEQUENCE:
        for angle_index, angle_deg in enumerate(rp.ANGLES_DEG):
            sample_frame = sample_records[mode][angle_deg]
            blank_frame = blank_records[mode][angle_deg]
            empty_frame = empty_records[mode][angle_deg]
            dark_split = rp.get_dark_split(dark_records, mode, angle_deg, sample_frame.exposure_us)
            sample_split = rp.get_raw_split(sample_frame.path)
            blank_split = rp.get_raw_split(blank_frame.path)
            empty_split = rp.get_raw_split(empty_frame.path)

            raw_rot_sample: dict[str, np.ndarray] = {}
            corrected_sample: dict[str, np.ndarray] = {}
            corrected_blank: dict[str, np.ndarray] = {}
            corrected_empty: dict[str, np.ndarray] = {}
            for channel in rp.RAW_CHANNELS:
                dark_plane = dark_split[channel]
                raw_rot_sample[channel] = rp.rotate_plane(sample_split[channel], float(angle_deg), center_half)
                corrected_sample[channel] = rp.rotate_plane(
                    (sample_split[channel] - dark_plane) / sample_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                )
                corrected_blank[channel] = rp.rotate_plane(
                    (blank_split[channel] - dark_plane) / blank_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                )
                corrected_empty[channel] = rp.rotate_plane(
                    (empty_split[channel] - dark_plane) / empty_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                )
                curves["dark_raw"][mode][channel][angle_index] = rp.masked_mean(dark_plane, union_mask)
                curves["empty_corr"][mode][channel][angle_index] = rp.masked_mean(corrected_empty[channel], union_mask)
                curves["blank_corr"][mode][channel][angle_index] = rp.masked_mean(corrected_blank[channel], union_mask)
                curves["sample_raw_rot"][mode][channel][angle_index] = rp.masked_mean(raw_rot_sample[channel], union_mask)
                curves["sample_corr"][mode][channel][angle_index] = rp.masked_mean(corrected_sample[channel], union_mask)
                saturation_fraction[mode][channel][angle_index] = masked_fraction(
                    sample_split[channel] >= rp.SATURATION_LEVEL,
                    union_mask,
                )

            curves["dark_raw"][mode]["G"][angle_index] = 0.5 * (
                curves["dark_raw"][mode]["G1"][angle_index] + curves["dark_raw"][mode]["G2"][angle_index]
            )
            raw_rot_sample["G"] = 0.5 * (raw_rot_sample["G1"] + raw_rot_sample["G2"])
            corrected_sample["G"] = 0.5 * (corrected_sample["G1"] + corrected_sample["G2"])
            corrected_blank["G"] = 0.5 * (corrected_blank["G1"] + corrected_blank["G2"])
            corrected_empty["G"] = 0.5 * (corrected_empty["G1"] + corrected_empty["G2"])

            curves["empty_corr"][mode]["G"][angle_index] = rp.masked_mean(corrected_empty["G"], union_mask)
            curves["blank_corr"][mode]["G"][angle_index] = rp.masked_mean(corrected_blank["G"], union_mask)
            curves["sample_raw_rot"][mode]["G"][angle_index] = rp.masked_mean(raw_rot_sample["G"], union_mask)
            curves["sample_corr"][mode]["G"][angle_index] = rp.masked_mean(corrected_sample["G"], union_mask)

            if mode == "PPL":
                for roi_index, roi in enumerate(roi_entries):
                    x0, y0, width, height = roi.half_rect
                    roi_patch_stacks_ppl_g[roi_index, angle_index] = corrected_sample["G"][y0 : y0 + height, x0 : x0 + width]

            patch = rp.gradient_magnitude(corrected_blank["G"][bbox_y0:bbox_y1, bbox_x0:bbox_x1])
            if angle_deg == 0:
                reference_patch_by_mode[mode] = patch.copy()
            else:
                shift_x, shift_y, corr_peak = rp.phase_correlation_shift(
                    reference_patch_by_mode[mode],
                    patch,
                    bbox_mask,
                )
                registration_rows.append(
                    {
                        "mode": mode,
                        "angle_deg": float(angle_deg),
                        "shift_x_px": shift_x,
                        "shift_y_px": shift_y,
                        "shift_magnitude_px": float(np.hypot(shift_x, shift_y)),
                        "corr_peak": corr_peak,
                    }
                )

    tau_low = {}
    eps_count = {}
    for channel in ALL_CHANNELS:
        blank_ppl_curve = np.asarray(curves["blank_corr"]["PPL"][channel], dtype=np.float64)
        blank_ppl_median = rp.nanmedian(blank_ppl_curve)
        tau_low[channel] = 0.02 * blank_ppl_median
        if channel in rp.FIT_CHANNELS:
            eps_count[channel] = 0.005 * blank_ppl_median

    near_black_fraction = init_mode_channel_lists(ALL_CHANNELS)
    for mode in rp.MODE_SEQUENCE:
        for angle_index, angle_deg in enumerate(rp.ANGLES_DEG):
            sample_frame = sample_records[mode][angle_deg]
            dark_split = rp.get_dark_split(dark_records, mode, angle_deg, sample_frame.exposure_us)
            sample_split = rp.get_raw_split(sample_frame.path)
            corrected_sample: dict[str, np.ndarray] = {}
            for channel in rp.RAW_CHANNELS:
                corrected_sample[channel] = rp.rotate_plane(
                    (sample_split[channel] - dark_split[channel]) / sample_frame.exposure_us,
                    float(angle_deg),
                    center_half,
                )
            corrected_sample["G"] = 0.5 * (corrected_sample["G1"] + corrected_sample["G2"])
            for channel in ALL_CHANNELS:
                near_black_fraction[mode][channel][angle_index] = masked_fraction(
                    corrected_sample[channel] < tau_low[channel],
                    union_mask,
                )

    signal_curves = {signal: [float("nan")] * len(rp.ANGLES_DEG) for signal in PHASE_B_SIGNALS}
    signal_valid = {signal: [False] * len(rp.ANGLES_DEG) for signal in PHASE_B_SIGNALS}
    for angle_index, _angle_deg in enumerate(rp.ANGLES_DEG):
        values, valid = rp.compute_signal_values_for_scalar(
            sample_ppl=float(curves["sample_corr"]["PPL"]["G"][angle_index]),
            sample_xpl=float(curves["sample_corr"]["XPL"]["G"][angle_index]),
            blank_ppl=float(curves["blank_corr"]["PPL"]["G"][angle_index]),
            blank_xpl=float(curves["blank_corr"]["XPL"]["G"][angle_index]),
            eps_count=float(eps_count["G"]),
            tau_low=float(tau_low["G"]),
        )
        for signal in PHASE_B_SIGNALS:
            signal_curves[signal][angle_index] = float(values[signal])
            signal_valid[signal][angle_index] = bool(valid[signal])

    signal_fits = {
        signal: rp.fit_harmonic_curve(
            np.asarray(signal_curves[signal], dtype=np.float64),
            np.asarray(signal_valid[signal], dtype=bool),
        )
        for signal in PHASE_B_SIGNALS
    }

    g1_blank = np.asarray(curves["blank_corr"]["PPL"]["G1"] + curves["blank_corr"]["XPL"]["G1"], dtype=np.float64)
    g2_blank = np.asarray(curves["blank_corr"]["PPL"]["G2"] + curves["blank_corr"]["XPL"]["G2"], dtype=np.float64)
    g1_g2_curve_diff = float(
        np.median(np.abs(g1_blank - g2_blank) / (0.5 * (g1_blank + g2_blank) + max(eps_count["G"], 1e-12)))
    )
    blank_flatness = rp.compute_blank_flatness_metrics(
        {
            "PPL": curves["blank_corr"]["PPL"]["G"],
            "XPL": curves["blank_corr"]["XPL"]["G"],
        }
    )

    registration_summary = {}
    registration_flags = []
    for mode in rp.MODE_SEQUENCE:
        magnitudes = [float(row["shift_magnitude_px"]) for row in registration_rows if row["mode"] == mode]
        median_shift = median_or_nan(magnitudes)
        flag = bool(np.isfinite(median_shift) and median_shift <= 0.5)
        registration_flags.append(flag)
        registration_summary[mode] = {
            "median_shift_px": median_shift,
            "flag": flag,
        }

    summary_row = {
        "sample_id": sample.sample_id,
        "roi_labels": ",".join(roi.label for roi in roi_entries),
        "eps_count_G": float(eps_count["G"]),
        "tau_low_G": float(tau_low["G"]),
        "G1_G2_curve_diff": g1_g2_curve_diff,
        "blank_flatness_flag": bool(blank_flatness["flag"]),
        "registration_stability_flag": bool(all(registration_flags)),
        "registration_median_shift_PPL_px": float(registration_summary["PPL"]["median_shift_px"]),
        "registration_median_shift_XPL_px": float(registration_summary["XPL"]["median_shift_px"]),
        "valid_angles_Afilm_PPL_G": int(signal_fits["Afilm_PPL"]["n_valid_angles"]),
        "valid_angles_Xfilm_G": int(signal_fits["Xfilm"]["n_valid_angles"]),
        "A2_Afilm_PPL_G": float(signal_fits["Afilm_PPL"]["A2"]),
        "axis2_Afilm_PPL_G_deg": float(signal_fits["Afilm_PPL"]["axis2_deg"]),
        "RMSE_Afilm_PPL_G": float(signal_fits["Afilm_PPL"]["rmse"]),
        "NRMSE_Afilm_PPL_G": float(signal_fits["Afilm_PPL"]["nrmse"]),
        "A4_Xfilm_G": float(signal_fits["Xfilm"]["A4"]),
        "axis4_Xfilm_G_deg": float(signal_fits["Xfilm"]["axis4_deg"]),
        "RMSE_Xfilm_G": float(signal_fits["Xfilm"]["rmse"]),
        "NRMSE_Xfilm_G": float(signal_fits["Xfilm"]["nrmse"]),
        "A4_Xnorm_sample_G": float(signal_fits["Xnorm_sample"]["A4"]),
        "axis4_Xnorm_sample_G_deg": float(signal_fits["Xnorm_sample"]["axis4_deg"]),
        "RMSE_Xnorm_sample_G": float(signal_fits["Xnorm_sample"]["rmse"]),
        "NRMSE_Xnorm_sample_G": float(signal_fits["Xnorm_sample"]["nrmse"]),
        "A4_Xnorm_blank_G": float(signal_fits["Xnorm_blank"]["A4"]),
        "axis4_Xnorm_blank_G_deg": float(signal_fits["Xnorm_blank"]["axis4_deg"]),
        "RMSE_Xnorm_blank_G": float(signal_fits["Xnorm_blank"]["rmse"]),
        "NRMSE_Xnorm_blank_G": float(signal_fits["Xnorm_blank"]["nrmse"]),
        "near_black_fraction_PPL_G_median": median_or_nan(near_black_fraction["PPL"]["G"]),
        "near_black_fraction_XPL_G_median": median_or_nan(near_black_fraction["XPL"]["G"]),
        "saturation_fraction_PPL_G1_median": median_or_nan(saturation_fraction["PPL"]["G1"]),
        "saturation_fraction_XPL_G1_median": median_or_nan(saturation_fraction["XPL"]["G1"]),
    }

    return {
        "sample": sample,
        "roi_entries": roi_entries,
        "curves": curves,
        "roi_patch_stacks_ppl_g": roi_patch_stacks_ppl_g,
        "saturation_fraction": saturation_fraction,
        "near_black_fraction": near_black_fraction,
        "registration_rows": registration_rows,
        "registration_summary": registration_summary,
        "eps_count": eps_count,
        "tau_low": tau_low,
        "signal_curves": signal_curves,
        "signal_valid": signal_valid,
        "signal_fits": signal_fits,
        "g1_g2_curve_diff": g1_g2_curve_diff,
        "blank_flatness": blank_flatness,
        "summary_row": summary_row,
    }


def plot_phase_a(payload: dict[str, Any], output_path: Path) -> None:
    sample = payload["sample"]
    curves = payload["curves"]
    saturation_fraction = payload["saturation_fraction"]
    near_black_fraction = payload["near_black_fraction"]
    registration_rows = payload["registration_rows"]
    registration_summary = payload["registration_summary"]
    g1_g2_curve_diff = payload["g1_g2_curve_diff"]
    blank_flatness = payload["blank_flatness"]
    eps_count = payload["eps_count"]
    tau_low = payload["tau_low"]
    angles = np.asarray(rp.ANGLES_DEG, dtype=np.float64)

    figure, axes = plt.subplots(6, 2, figsize=(15, 20), constrained_layout=True)

    for axis, mode in zip(axes[0], rp.MODE_SEQUENCE):
        for channel in rp.RAW_CHANNELS:
            axis.plot(angles, curves["dark_raw"][mode][channel], marker="o", label=channel)
        axis.set_title(f"{sample.sample_id} dark {mode}")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Raw mean")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    for axis, mode in zip(axes[1], rp.MODE_SEQUENCE):
        for channel in rp.RAW_CHANNELS:
            axis.plot(angles, curves["empty_corr"][mode][channel], marker="o", label=channel)
        axis.set_title(f"{sample.sample_id} empty {mode}")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Corrected mean")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    for axis, mode in zip(axes[2], rp.MODE_SEQUENCE):
        for channel in rp.RAW_CHANNELS:
            axis.plot(angles, curves["blank_corr"][mode][channel], marker="o", label=channel)
        axis.set_title(f"{sample.sample_id} blank {mode}")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Corrected mean")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    axes[3, 0].plot(angles, curves["blank_corr"]["PPL"]["G1"], marker="o", label="G1")
    axes[3, 0].plot(angles, curves["blank_corr"]["PPL"]["G2"], marker="s", label="G2")
    axes[3, 0].set_title(f"{sample.sample_id} blank PPL G1/G2")
    axes[3, 0].set_xlabel("theta (deg)")
    axes[3, 0].set_ylabel("Corrected mean")
    axes[3, 0].grid(alpha=0.3)
    axes[3, 0].legend(fontsize=8)

    axes[3, 1].plot(angles, curves["blank_corr"]["XPL"]["G1"], marker="o", label="G1")
    axes[3, 1].plot(angles, curves["blank_corr"]["XPL"]["G2"], marker="s", label="G2")
    axes[3, 1].set_title(f"{sample.sample_id} blank XPL G1/G2")
    axes[3, 1].set_xlabel("theta (deg)")
    axes[3, 1].set_ylabel("Corrected mean")
    axes[3, 1].grid(alpha=0.3)
    axes[3, 1].legend(fontsize=8)

    reg_ax = axes[4, 0]
    for mode in rp.MODE_SEQUENCE:
        mode_rows = [row for row in registration_rows if row["mode"] == mode]
        reg_ax.plot(
            [row["angle_deg"] for row in mode_rows],
            [row["shift_magnitude_px"] for row in mode_rows],
            marker="o",
            label=f"{mode} shift",
        )
    reg_ax.set_title(f"{sample.sample_id} registration stability")
    reg_ax.set_xlabel("theta (deg)")
    reg_ax.set_ylabel("Shift magnitude (px)")
    reg_ax.grid(alpha=0.3)
    reg_ax.legend(fontsize=8)
    reg_text = (
        f"PPL median={registration_summary['PPL']['median_shift_px']:.3f}px\n"
        f"XPL median={registration_summary['XPL']['median_shift_px']:.3f}px\n"
        f"PPL flag={registration_summary['PPL']['flag']}\n"
        f"XPL flag={registration_summary['XPL']['flag']}"
    )
    reg_ax.text(
        0.98,
        0.98,
        reg_text,
        transform=reg_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    sat_ax = axes[4, 1]
    for mode, linestyle in (("PPL", "-"), ("XPL", "--")):
        for channel in rp.RAW_CHANNELS:
            sat_ax.plot(angles, saturation_fraction[mode][channel], linestyle=linestyle, label=f"{mode} {channel}")
    sat_ax.set_title(f"{sample.sample_id} saturation fraction")
    sat_ax.set_xlabel("theta (deg)")
    sat_ax.set_ylabel("Fraction")
    sat_ax.grid(alpha=0.3)
    sat_ax.legend(fontsize=7, ncol=2)

    near_ax = axes[5, 0]
    for mode, linestyle in (("PPL", "-"), ("XPL", "--")):
        for channel in rp.RAW_CHANNELS:
            near_ax.plot(angles, near_black_fraction[mode][channel], linestyle=linestyle, label=f"{mode} {channel}")
    near_ax.set_title(f"{sample.sample_id} near-black fraction")
    near_ax.set_xlabel("theta (deg)")
    near_ax.set_ylabel("Fraction")
    near_ax.grid(alpha=0.3)
    near_ax.legend(fontsize=7, ncol=2)

    info_ax = axes[5, 1]
    info_ax.axis("off")
    info_text = (
        f"ROI labels: {', '.join(roi.label for roi in payload['roi_entries'])}\n"
        f"G1/G2 curve diff: {g1_g2_curve_diff:.5f}\n"
        f"Blank flatness flag: {blank_flatness['flag']}\n"
        f"eps_count_G: {eps_count['G']:.6g}\n"
        f"tau_low_G: {tau_low['G']:.6g}"
    )
    info_ax.text(
        0.02,
        0.98,
        info_text,
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9},
    )

    figure.suptitle(f"{sample.sample_id} Phase A Calibration QC", fontsize=16)
    rp.ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_phase_b(payload: dict[str, Any], output_path: Path) -> None:
    sample = payload["sample"]
    curves = payload["curves"]
    signal_curves = payload["signal_curves"]
    signal_valid = payload["signal_valid"]
    signal_fits = payload["signal_fits"]
    angles = np.asarray(rp.ANGLES_DEG, dtype=np.float64)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    def plot_raw_corrected(axis: plt.Axes, mode: str, title: str) -> None:
        raw_values = np.asarray(curves["sample_raw_rot"][mode]["G"], dtype=np.float64)
        corrected_values = np.asarray(curves["sample_corr"][mode]["G"], dtype=np.float64)
        axis.plot(angles, raw_values, marker="o", color="#1f77b4", label="raw derotated G")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Raw mean", color="#1f77b4")
        axis.tick_params(axis="y", labelcolor="#1f77b4")
        twin = axis.twinx()
        twin.plot(angles, corrected_values, marker="s", color="#d62728", label="corrected G")
        twin.set_ylabel("Corrected mean", color="#d62728")
        twin.tick_params(axis="y", labelcolor="#d62728")
        lines = axis.get_lines() + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
        axis.set_title(title)
        axis.grid(alpha=0.3)

    plot_raw_corrected(axes[0, 0], "PPL", f"{sample.sample_id} sample_PPL_G")
    plot_raw_corrected(axes[0, 1], "XPL", f"{sample.sample_id} sample_XPL_G")

    signal_titles = {
        "Afilm_PPL": "Afilm_PPL_G",
        "Xfilm": "Xfilm_G",
        "Xnorm_sample": "Xnorm_sample_G",
        "Xnorm_blank": "Xnorm_blank_G",
    }
    for axis, signal_name in zip(axes[1:].flat, PHASE_B_SIGNALS):
        values = np.asarray(signal_curves[signal_name], dtype=np.float64)
        valid_mask = np.asarray(signal_valid[signal_name], dtype=bool)
        fit = signal_fits[signal_name]
        axis.plot(angles, values, marker="o", label="raw curve")
        if fit["fit_valid"]:
            axis.plot(angles, fit["predicted"], linestyle="--", linewidth=1.6, label="2theta + 4theta fit")
            residual = np.where(valid_mask, values - fit["predicted"], np.nan)
            text = (
                f"n={fit['n_valid_angles']}\n"
                f"A2={fit['A2']:.4g}  A4={fit['A4']:.4g}\n"
                f"RMSE={fit['rmse']:.4g}\n"
                f"NRMSE={fit['nrmse']:.4g}\n"
                f"axis2={fit['axis2_deg']:.2f}deg\n"
                f"axis4={fit['axis4_deg']:.2f}deg\n"
                f"median resid={rp.nanmedian(residual):.4g}"
            )
        else:
            text = f"fit invalid\nn={fit['n_valid_angles']}"
        axis.set_title(f"{sample.sample_id} {signal_titles[signal_name]}")
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("Signal")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        axis.text(
            0.98,
            0.98,
            text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85},
        )

    figure.suptitle(f"{sample.sample_id} Phase B Sanity QC", fontsize=16)
    rp.ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_derotation_visual_qc(payload: dict[str, Any], output_path: Path) -> None:
    sample = payload["sample"]
    roi_entries = payload["roi_entries"]
    roi_patch_stacks = payload["roi_patch_stacks_ppl_g"]
    selected_indices = [rp.ANGLES_DEG.index(angle) for angle in VISUAL_STACK_ANGLES]
    column_titles = [f"{angle} deg" for angle in VISUAL_STACK_ANGLES] + ["Mean", "Std", "MAD vs 0deg"]
    figure, axes = plt.subplots(len(roi_entries), len(column_titles), figsize=(18, 3.8 * len(roi_entries)), constrained_layout=True)
    if len(roi_entries) == 1:
        axes = np.asarray([axes])

    for roi_index, roi in enumerate(roi_entries):
        stack = roi_patch_stacks[roi_index]
        ref = stack[0]
        mean_img = np.nanmean(stack, axis=0)
        std_img = np.nanstd(stack, axis=0)
        mad_img = np.nanmean(np.abs(stack - ref[None, :, :]), axis=0)

        intensity_vmin = finite_percentile(stack, 1.0)
        intensity_vmax = finite_percentile(stack, 99.0)
        std_vmax = finite_percentile(std_img, 99.0)
        mad_vmax = finite_percentile(mad_img, 99.0)

        image_list = [stack[index] for index in selected_indices] + [mean_img, std_img, mad_img]
        cmaps = ["gray"] * len(VISUAL_STACK_ANGLES) + ["gray", "magma", "magma"]
        vmax_list = [intensity_vmax] * len(VISUAL_STACK_ANGLES) + [intensity_vmax, std_vmax, mad_vmax]
        vmin_list = [intensity_vmin] * len(VISUAL_STACK_ANGLES) + [intensity_vmin, 0.0, 0.0]

        for col_index, (image, title, cmap, vmin, vmax) in enumerate(zip(image_list, column_titles, cmaps, vmin_list, vmax_list)):
            axis = axes[roi_index, col_index]
            axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if roi_index == 0:
                axis.set_title(title, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(roi.label, rotation=90, fontsize=11)

        mean_abs_diff_curve = np.nanmean(np.abs(stack - ref[None, :, :]), axis=(1, 2))
        per_angle_corr = []
        ref_flat = ref.ravel()
        for angle_index in range(stack.shape[0]):
            mov_flat = stack[angle_index].ravel()
            if np.allclose(np.nanstd(mov_flat), 0.0) or np.allclose(np.nanstd(ref_flat), 0.0):
                per_angle_corr.append(float("nan"))
            else:
                per_angle_corr.append(float(np.corrcoef(ref_flat, mov_flat)[0, 1]))
        text = (
            f"mean std={np.nanmean(std_img):.4g}\n"
            f"median std={np.nanmedian(std_img):.4g}\n"
            f"mean MAD={np.nanmean(mad_img):.4g}\n"
            f"corr(165,0)={per_angle_corr[-1]:.4f}"
        )
        axes[roi_index, -1].text(
            0.98,
            0.02,
            text,
            transform=axes[roi_index, -1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8},
        )

    figure.suptitle(f"{sample.sample_id} Derotation Visual QC (PPL G)", fontsize=16)
    rp.ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def build_derotation_visual_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roi_entries = payload["roi_entries"]
    roi_patch_stacks = payload["roi_patch_stacks_ppl_g"]
    sample = payload["sample"]
    for roi_index, roi in enumerate(roi_entries):
        stack = roi_patch_stacks[roi_index]
        ref = stack[0]
        ref_flat = ref.ravel()
        for angle_index, angle_deg in enumerate(rp.ANGLES_DEG):
            patch = stack[angle_index]
            patch_flat = patch.ravel()
            corr = float(np.corrcoef(ref_flat, patch_flat)[0, 1]) if np.nanstd(ref_flat) > 0 and np.nanstd(patch_flat) > 0 else float("nan")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "roi_label": roi.label,
                    "angle_deg": angle_deg,
                    "mean_intensity": float(np.nanmean(patch)),
                    "std_intensity": float(np.nanstd(patch)),
                    "mean_abs_diff_vs_0deg": float(np.nanmean(np.abs(patch - ref))),
                    "corr_vs_0deg": corr,
                }
            )
    return rows


def build_derotation_visual_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    roi_entries = payload["roi_entries"]
    roi_patch_stacks = payload["roi_patch_stacks_ppl_g"]
    sample = payload["sample"]
    per_roi = {}
    for roi_index, roi in enumerate(roi_entries):
        stack = roi_patch_stacks[roi_index]
        ref = stack[0]
        mean_img = np.nanmean(stack, axis=0)
        std_img = np.nanstd(stack, axis=0)
        mad_img = np.nanmean(np.abs(stack - ref[None, :, :]), axis=0)
        ref_flat = ref.ravel()
        corr_values = []
        mad_curve = []
        for angle_index in range(stack.shape[0]):
            patch_flat = stack[angle_index].ravel()
            corr = float(np.corrcoef(ref_flat, patch_flat)[0, 1]) if np.nanstd(ref_flat) > 0 and np.nanstd(patch_flat) > 0 else float("nan")
            corr_values.append(corr)
            mad_curve.append(float(np.nanmean(np.abs(stack[angle_index] - ref))))
        per_roi[roi.label] = {
            "mean_std_image": float(np.nanmean(std_img)),
            "median_std_image": float(np.nanmedian(std_img)),
            "mean_mad_image": float(np.nanmean(mad_img)),
            "median_mad_curve": median_or_nan(mad_curve),
            "corr_165deg_vs_0deg": float(corr_values[-1]),
            "selected_angles_deg": list(VISUAL_STACK_ANGLES),
        }
    return {
        "sample_id": sample.sample_id,
        "channel": "G",
        "mode": "PPL",
        "roi_metrics": per_roi,
    }


def build_phase_a_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sample = payload["sample"]
    curves = payload["curves"]
    rows: list[dict[str, Any]] = []
    for series_name in ("dark_raw", "empty_corr", "blank_corr", "sample_raw_rot", "sample_corr"):
        for mode in rp.MODE_SEQUENCE:
            for channel in ALL_CHANNELS:
                for angle_deg, value in zip(rp.ANGLES_DEG, curves[series_name][mode][channel]):
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "series": series_name,
                            "mode": mode,
                            "channel": channel,
                            "angle_deg": angle_deg,
                            "value": value,
                        }
                    )
    for series_name, store in (
        ("saturation_fraction", payload["saturation_fraction"]),
        ("near_black_fraction", payload["near_black_fraction"]),
    ):
        channels = rp.RAW_CHANNELS if series_name == "saturation_fraction" else ALL_CHANNELS
        for mode in rp.MODE_SEQUENCE:
            for channel in channels:
                for angle_deg, value in zip(rp.ANGLES_DEG, store[mode][channel]):
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "series": series_name,
                            "mode": mode,
                            "channel": channel,
                            "angle_deg": angle_deg,
                            "value": value,
                        }
                    )
    for row in payload["registration_rows"]:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "series": "registration_shift",
                "mode": row["mode"],
                "channel": "G",
                "angle_deg": int(row["angle_deg"]),
                "value": row["shift_magnitude_px"],
            }
        )
    return rows


def build_phase_b_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sample = payload["sample"]
    curves = payload["curves"]
    signal_curves = payload["signal_curves"]
    signal_valid = payload["signal_valid"]
    signal_fits = payload["signal_fits"]
    rows = []
    for angle_index, angle_deg in enumerate(rp.ANGLES_DEG):
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "angle_deg": angle_deg,
            "sample_PPL_G_raw": float(curves["sample_raw_rot"]["PPL"]["G"][angle_index]),
            "sample_PPL_G_corrected": float(curves["sample_corr"]["PPL"]["G"][angle_index]),
            "sample_XPL_G_raw": float(curves["sample_raw_rot"]["XPL"]["G"][angle_index]),
            "sample_XPL_G_corrected": float(curves["sample_corr"]["XPL"]["G"][angle_index]),
            "blank_PPL_G_corrected": float(curves["blank_corr"]["PPL"]["G"][angle_index]),
            "blank_XPL_G_corrected": float(curves["blank_corr"]["XPL"]["G"][angle_index]),
            "empty_PPL_G_corrected": float(curves["empty_corr"]["PPL"]["G"][angle_index]),
            "empty_XPL_G_corrected": float(curves["empty_corr"]["XPL"]["G"][angle_index]),
        }
        for signal_name in PHASE_B_SIGNALS:
            fit = signal_fits[signal_name]
            value = float(signal_curves[signal_name][angle_index])
            fit_value = float(fit["predicted"][angle_index]) if np.isfinite(fit["predicted"][angle_index]) else float("nan")
            row[f"{signal_name}_G"] = value
            row[f"{signal_name}_G_valid"] = bool(signal_valid[signal_name][angle_index])
            row[f"{signal_name}_G_fit"] = fit_value
            row[f"{signal_name}_G_residual"] = value - fit_value if np.isfinite(value) and np.isfinite(fit_value) else float("nan")
        rows.append(row)
    return rows


def build_phase_a_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload["sample"]
    return {
        "sample_id": sample.sample_id,
        "roi_entries": [
            {
                "label": roi.label,
                "full_rect": list(roi.full_rect),
                "half_rect": list(roi.half_rect),
            }
            for roi in payload["roi_entries"]
        ],
        "eps_count": {key: float(value) for key, value in payload["eps_count"].items()},
        "tau_low": {key: float(value) for key, value in payload["tau_low"].items()},
        "G1_G2_curve_diff": float(payload["g1_g2_curve_diff"]),
        "blank_flatness_flag": bool(payload["blank_flatness"]["flag"]),
        "blank_flatness_details": payload["blank_flatness"]["details"],
        "registration_stability_flag": bool(
            payload["registration_summary"]["PPL"]["flag"] and payload["registration_summary"]["XPL"]["flag"]
        ),
        "registration_summary": payload["registration_summary"],
        "registration_rows": payload["registration_rows"],
        "near_black_fraction_median": {
            mode: {channel: median_or_nan(payload["near_black_fraction"][mode][channel]) for channel in ALL_CHANNELS}
            for mode in rp.MODE_SEQUENCE
        },
        "saturation_fraction_median": {
            mode: {channel: median_or_nan(payload["saturation_fraction"][mode][channel]) for channel in rp.RAW_CHANNELS}
            for mode in rp.MODE_SEQUENCE
        },
    }


def build_phase_b_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload["sample"]
    return {
        "sample_id": sample.sample_id,
        "eps_count_G": float(payload["eps_count"]["G"]),
        "tau_low_G": float(payload["tau_low"]["G"]),
        "signal_fits": {signal_name: fit_to_json(fit) for signal_name, fit in payload["signal_fits"].items()},
    }


def write_sample_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    sample = payload["sample"]
    sample_dir = rp.ensure_dir(output_dir / sample.sample_id)
    phase_a_plot = sample_dir / "phaseA_calibration_qc.png"
    phase_b_plot = sample_dir / "phaseB_sanity_qc.png"
    phase_a5_plot = sample_dir / "phaseA5_derotation_visual_qc.png"
    phase_a_csv = sample_dir / "phaseA_calibration_curves.csv"
    phase_b_csv = sample_dir / "phaseB_signal_curves.csv"
    phase_a5_csv = sample_dir / "phaseA5_derotation_visual_metrics.csv"
    phase_a_json = sample_dir / "phaseA_calibration_metrics.json"
    phase_b_json = sample_dir / "phaseB_signal_metrics.json"
    phase_a5_json = sample_dir / "phaseA5_derotation_visual_metrics.json"

    plot_phase_a(payload, phase_a_plot)
    plot_phase_b(payload, phase_b_plot)
    plot_derotation_visual_qc(payload, phase_a5_plot)
    rp.write_csv_rows(phase_a_csv, build_phase_a_rows(payload))
    rp.write_csv_rows(phase_b_csv, build_phase_b_rows(payload))
    rp.write_csv_rows(phase_a5_csv, build_derotation_visual_rows(payload))
    rp.save_json(phase_a_json, build_phase_a_metrics(payload))
    rp.save_json(phase_b_json, build_phase_b_metrics(payload))
    rp.save_json(phase_a5_json, build_derotation_visual_metrics(payload))

    return {
        "phaseA_plot": str(phase_a_plot.relative_to(REPO_ROOT)),
        "phaseA_csv": str(phase_a_csv.relative_to(REPO_ROOT)),
        "phaseA_json": str(phase_a_json.relative_to(REPO_ROOT)),
        "phaseB_plot": str(phase_b_plot.relative_to(REPO_ROOT)),
        "phaseB_csv": str(phase_b_csv.relative_to(REPO_ROOT)),
        "phaseB_json": str(phase_b_json.relative_to(REPO_ROOT)),
        "phaseA5_plot": str(phase_a5_plot.relative_to(REPO_ROOT)),
        "phaseA5_csv": str(phase_a5_csv.relative_to(REPO_ROOT)),
        "phaseA5_json": str(phase_a5_json.relative_to(REPO_ROOT)),
    }


def main() -> None:
    args = parse_args()
    manifest, samples = load_local_manifest(args.manifest)
    roi_spec = rp.load_json(Path(manifest["roi_spec"]))
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    rp.ensure_dir(output_dir)

    summary_rows = []
    index_entries = []
    for sample in samples:
        payload = compute_phase_payload(sample, manifest, roi_spec)
        file_map = write_sample_outputs(payload, output_dir)
        summary_rows.append(payload["summary_row"])
        index_entries.append(
            {
                "sample_id": sample.sample_id,
                "files": file_map,
            }
        )

    summary_path = output_dir / "shortframe_qc_summary.csv"
    index_path = output_dir / "index.json"
    rp.write_csv_rows(summary_path, summary_rows)
    rp.save_json(
        index_path,
        {
            "manifest": str(args.manifest.relative_to(REPO_ROOT) if args.manifest.is_absolute() else args.manifest),
            "roi_spec": str(Path(manifest["roi_spec"]).relative_to(REPO_ROOT)),
            "samples": index_entries,
            "summary_csv": str(summary_path.relative_to(REPO_ROOT)),
        },
    )


if __name__ == "__main__":
    main()
