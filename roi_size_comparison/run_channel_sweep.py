"""Multi-channel signal sweep for shortframe samples (s9-s13).

Evaluates harmonic fits across 55 channel x signal combinations per ROI,
with channel-type-appropriate signal derivations:
  A. Intensity-like  (8 ch x 5 sig = 40)
  B. Opponent/chroma  (3 ch x 3 sig =  9)
  C. Hue              (1 ch x 3 sig =  3)  circular difference
  D. Pseudo-deltaE76  (1 ch x 3 sig =  3)  direct harmonic fit
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_analysis_pipeline as rp  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = REPO_ROOT / "shortframe_calibration" / "shortframe_manifest.json"
ROI_JSON_PATH = REPO_ROOT / "shortframe_calibration" / "shortframe_roi_200.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

TS = 32
STRIDE = 16
TILE_VLD_TH = 0.9

INTENSITY_CHS = ("G1", "G2", "R", "B", "G", "Gray_sp", "L", "V")
OPPONENT_CHS = ("a_lab", "b_lab", "S")
HUE_CHS = ("H",)
DELTAE_CH = "pseudo_dE76"
ALL_PIXEL_CHS = INTENSITY_CHS + OPPONENT_CHS + HUE_CHS

INTENSITY_SIGS = ("Tfilm", "Afilm", "Xfilm", "Xnorm_blank", "Xnorm_sample")
OPPONENT_SIGS = ("Dblank_PPL", "Dblank_XPL", "Dxp_pp")
HUE_SIGS = ("dH_xpl_ppl", "dH_angle_0", "dH_sample_blank")
DELTAE_SIGS = ("pdE_xpl_ppl", "pdE_angle_0", "pdE_sample_blank")

SOURCES = ("sample_PPL", "sample_XPL", "blank_PPL", "blank_XPL")

EPS = 1e-10

_M_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3,
                    np.cbrt(np.maximum(t, 0.0)),
                    t / (3 * delta ** 2) + 4.0 / 29.0)


def rgb_to_lab(rgb: np.ndarray, white_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) linear camera-RGB -> pseudo-CIE-Lab.  white_rgb: (3,) scalar."""
    norm = rgb.astype(np.float64) / (white_rgb.reshape(1, 1, 3) + EPS)
    xyz = np.einsum("ij,hwj->hwi", _M_XYZ, norm) / _D65.reshape(1, 1, 3)
    f = _lab_f(xyz)
    lab_L = 116.0 * f[..., 1] - 16.0
    lab_a = 500.0 * (f[..., 0] - f[..., 1])
    lab_b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lab_L, lab_a, lab_b], axis=-1).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) non-negative linear RGB -> HSV.  H in [0,360), S in [0,1], V=max."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v_max = np.maximum(np.maximum(r, g), b)
    v_min = np.minimum(np.minimum(r, g), b)
    chroma = v_max - v_min
    sat = np.where(v_max > EPS, chroma / (v_max + EPS), 0.0)
    hue = np.zeros_like(v_max)
    mc = chroma > EPS
    mr = mc & (r >= g) & (r >= b)
    mg = mc & ~mr & (g >= b)
    mb = mc & ~mr & ~mg
    hue[mr] = (60.0 * ((g[mr] - b[mr]) / (chroma[mr] + EPS))) % 360
    hue[mg] = (60.0 * ((b[mg] - r[mg]) / (chroma[mg] + EPS)) + 120.0) % 360
    hue[mb] = (60.0 * ((r[mb] - g[mb]) / (chroma[mb] + EPS)) + 240.0) % 360
    return np.stack([hue, sat, v_max], axis=-1).astype(np.float32)


def deltae76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Pixel-wise pseudo-dE76.  (H,W,3) -> (H,W)."""
    d = lab1.astype(np.float64) - lab2.astype(np.float64)
    return np.sqrt((d ** 2).sum(axis=-1)).astype(np.float32)


def circ_mean_deg(vals: np.ndarray) -> float:
    if vals.size == 0:
        return float("nan")
    r = np.deg2rad(vals.astype(np.float64))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def circ_diff_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise (a - b) wrapped to [-180, 180)."""
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) % 360
    return np.where(d >= 180, d - 360, d)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_manifest() -> tuple[dict[str, Any], list[rp.SampleSpec]]:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    samples = [
        rp.SampleSpec(
            sample_id=e["sample_id"],
            sample_type=e["sample_type"],
            sample_dir=rp.resolve_repo_path(e["sample_dir"]),
            blank_dir=rp.resolve_repo_path(e["blank_dir"]),
            empty_dir=rp.resolve_repo_path(e["empty_dir"]),
            dark_dir=rp.resolve_repo_path(e["dark_dir"]),
            roi_size_fullres=int(e["roi_size_fullres"]),
        )
        for e in manifest["samples"]
    ]
    for key in ("rotation_geometry", "rotation_valid_mask"):
        manifest[key] = str(rp.resolve_repo_path(manifest[key]))
    if "rotation_correction" in manifest:
        manifest["rotation_correction"] = str(rp.resolve_repo_path(manifest["rotation_correction"]))
    return manifest, samples


def load_rois() -> dict[str, list[rp.ROIEntry]]:
    data = rp.load_json(ROI_JSON_PATH)
    result: dict[str, list[rp.ROIEntry]] = {}
    for sid, sd in data["samples"].items():
        entries = []
        for label in data["roi_labels"]:
            reg = sd["regions"][label]
            fr = (reg["x"], reg["y"], reg["width"], reg["height"])
            entries.append(rp.ROIEntry(label, fr, rp.full_rect_to_half_rect(fr)))
        result[sid] = entries
    return result


# ---------------------------------------------------------------------------
# Channel derivation & white reference
# ---------------------------------------------------------------------------


def derive_channels(bayer: dict[str, np.ndarray],
                    white_ref: np.ndarray) -> dict[str, np.ndarray]:
    """From corrected Bayer {G1,G2,R,B} compute all 12 pixel-level channels."""
    ch: dict[str, np.ndarray] = dict(bayer)
    g = 0.5 * (ch["G1"] + ch["G2"])
    ch["G"] = g
    ch["Gray_sp"] = (ch["R"] + 2.0 * g + ch["B"]) / 4.0

    rgb = np.stack([ch["R"], g, ch["B"]], axis=-1)
    lab = rgb_to_lab(rgb, white_ref)
    ch["L"] = lab[..., 0]
    ch["a_lab"] = lab[..., 1]
    ch["b_lab"] = lab[..., 2]

    hsv = rgb_to_hsv(np.maximum(rgb, 0.0))
    ch["H"] = hsv[..., 0]
    ch["S"] = hsv[..., 1]
    ch["V"] = hsv[..., 2]
    return ch


def compute_white_ref(blank_rec: dict, dark_rec: dict, manifest: dict,
                      center_half: tuple[float, float],
                      valid_mask: np.ndarray,
                      rois: list[rp.ROIEntry]) -> np.ndarray:
    """Median of per-angle blank-PPL (R,G,B) means within ROI union -> (3,)."""
    roi_mask = np.zeros(valid_mask.shape, dtype=bool)
    for roi in rois:
        x, y, w, h = roi.half_rect
        roi_mask[y:y + h, x:x + w] |= valid_mask[y:y + h, x:x + w]

    n = len(rp.ANGLES_DEG)
    rgb_arr = np.empty((n, 3), dtype=np.float64)
    for ai, ang in enumerate(rp.ANGLES_DEG):
        fr = blank_rec["PPL"][ang]
        dk = rp.get_dark_split(dark_rec, "PPL", ang, fr.exposure_us)
        raw = rp.get_raw_split(fr.path)
        corr = {}
        for c in rp.RAW_CHANNELS:
            corr[c] = rp.derotate_and_shift(
                (raw[c] - dk[c]) / fr.exposure_us,
                float(ang), center_half, manifest,
            )
        g = 0.5 * (corr["G1"] + corr["G2"])
        rgb_arr[ai] = [float(corr["R"][roi_mask].mean()),
                       float(g[roi_mask].mean()),
                       float(corr["B"][roi_mask].mean())]
    return np.median(rgb_arr, axis=0)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

TileInfo = tuple[str, int, int]


def enumerate_tiles(rois: list[rp.ROIEntry],
                    valid_mask: np.ndarray) -> list[TileInfo]:
    tiles: list[TileInfo] = []
    for roi in rois:
        x, y, w, h = roi.half_rect
        for ty in range(y, y + h - TS + 1, STRIDE):
            for tx in range(x, x + w - TS + 1, STRIDE):
                if valid_mask[ty:ty + TS, tx:tx + TS].mean() >= TILE_VLD_TH:
                    tiles.append((roi.label, tx, ty))
    return tiles


def _tmean(img: np.ndarray, tx: int, ty: int,
           mask: np.ndarray) -> float:
    v = img[ty:ty + TS, tx:tx + TS][mask[ty:ty + TS, tx:tx + TS]]
    return float(v.mean()) if v.size else float("nan")


def _tcmean(img: np.ndarray, tx: int, ty: int,
            mask: np.ndarray) -> float:
    v = img[ty:ty + TS, tx:tx + TS][mask[ty:ty + TS, tx:tx + TS]]
    return circ_mean_deg(v)


def _tlab(channels: dict[str, np.ndarray], tx: int, ty: int) -> np.ndarray:
    return np.stack([channels["L"][ty:ty + TS, tx:tx + TS],
                     channels["a_lab"][ty:ty + TS, tx:tx + TS],
                     channels["b_lab"][ty:ty + TS, tx:tx + TS]], axis=-1)


def _de_mean(lab1: np.ndarray, lab2: np.ndarray,
             mask: np.ndarray) -> float:
    de = deltae76(lab1, lab2)
    v = de[mask]
    return float(v.mean()) if v.size else float("nan")


# ---------------------------------------------------------------------------
# Harmonic fit wrapper
# ---------------------------------------------------------------------------


def _fit_row(values: np.ndarray) -> dict[str, Any]:
    valid = np.ones(len(values), dtype=bool)
    fit = rp.fit_harmonic_curve(values.astype(np.float64), valid)
    a2 = fit["A2"] if np.isfinite(fit["A2"]) else 0.0
    a4 = fit["A4"] if np.isfinite(fit["A4"]) else 0.0
    denom = a2 + a4 + 1e-15
    ptp = float("nan")
    if fit["fit_valid"]:
        p = fit["predicted"]
        ptp = float(np.nanmax(p) - np.nanmin(p))
    return {
        "a0": fit["a0"],
        "A2": fit["A2"],
        "A4": fit["A4"],
        "axis2_deg": fit["axis2_deg"],
        "axis4_deg": fit["axis4_deg"],
        "RMSE": fit["rmse"],
        "NRMSE": fit["nrmse"],
        "peak_to_peak": ptp,
        "valid_angle_count": fit["n_valid_angles"],
        "LD_purity": a2 / denom,
        "LB_purity": a4 / denom,
        "A4_A2_ratio": a4 / (a2 + 1e-15),
        "A2_A4_logratio": math.log10((a2 + 1e-15) / (a4 + 1e-15)),
    }


# ---------------------------------------------------------------------------
# process_sample
# ---------------------------------------------------------------------------


def process_sample(sample: rp.SampleSpec,
                   manifest: dict,
                   valid_mask: np.ndarray,
                   center_half: tuple[float, float],
                   rois: list[rp.ROIEntry]) -> list[dict[str, Any]]:
    print(f"  {sample.sample_id} ...")
    n_ang = len(rp.ANGLES_DEG)

    sample_rec = rp.load_dataset_records(sample.sample_dir)
    blank_rec = rp.load_dataset_records(sample.blank_dir)
    dark_rec = rp.load_dataset_records(sample.dark_dir)

    white_ref = compute_white_ref(blank_rec, dark_rec, manifest,
                                  center_half, valid_mask, rois)
    print(f"    white ref  R={white_ref[0]:.4f}  G={white_ref[1]:.4f}  B={white_ref[2]:.4f}")

    tiles = enumerate_tiles(rois, valid_mask)
    n_tiles = len(tiles)
    print(f"    {n_tiles} valid tiles across {len(rois)} ROIs")

    td: dict[str, dict[str, np.ndarray]] = {
        ch: {src: np.full((n_tiles, n_ang), np.nan, dtype=np.float32)
             for src in SOURCES}
        for ch in ALL_PIXEL_CHS
    }
    de_data: dict[str, np.ndarray] = {
        sig: np.full((n_tiles, n_ang), np.nan, dtype=np.float32)
        for sig in DELTAE_SIGS
    }
    ref_lab: dict[int, np.ndarray] = {}

    # ── angle loop ────────────────────────────────────────────────────
    for ai, ang in enumerate(rp.ANGLES_DEG):
        imgs: dict[str, dict[str, np.ndarray]] = {}
        for mode in rp.MODE_SEQUENCE:
            s_fr = sample_rec[mode][ang]
            b_fr = blank_rec[mode][ang]
            s_dk = rp.get_dark_split(dark_rec, mode, ang, s_fr.exposure_us)
            b_dk = rp.get_dark_split(dark_rec, mode, ang, b_fr.exposure_us)
            s_raw = rp.get_raw_split(s_fr.path)
            b_raw = rp.get_raw_split(b_fr.path)

            s_corr: dict[str, np.ndarray] = {}
            b_corr: dict[str, np.ndarray] = {}
            for c in rp.RAW_CHANNELS:
                s_corr[c] = rp.derotate_and_shift(
                    (s_raw[c] - s_dk[c]) / s_fr.exposure_us,
                    float(ang), center_half, manifest)
                b_corr[c] = rp.derotate_and_shift(
                    (b_raw[c] - b_dk[c]) / b_fr.exposure_us,
                    float(ang), center_half, manifest)

            imgs[f"sample_{mode}"] = derive_channels(s_corr, white_ref)
            imgs[f"blank_{mode}"] = derive_channels(b_corr, white_ref)

        for ti, (_, tx, ty) in enumerate(tiles):
            tm = valid_mask[ty:ty + TS, tx:tx + TS]

            for ch in ALL_PIXEL_CHS:
                for src in SOURCES:
                    if ch == "H":
                        td[ch][src][ti, ai] = _tcmean(imgs[src][ch], tx, ty, valid_mask)
                    else:
                        td[ch][src][ti, ai] = _tmean(imgs[src][ch], tx, ty, valid_mask)

            lab_s_xpl = _tlab(imgs["sample_XPL"], tx, ty)
            lab_s_ppl = _tlab(imgs["sample_PPL"], tx, ty)
            lab_b_xpl = _tlab(imgs["blank_XPL"], tx, ty)

            de_data["pdE_xpl_ppl"][ti, ai] = _de_mean(lab_s_xpl, lab_s_ppl, tm)
            de_data["pdE_sample_blank"][ti, ai] = _de_mean(lab_s_xpl, lab_b_xpl, tm)

            if ai == 0:
                ref_lab[ti] = lab_s_ppl.copy()
            de_data["pdE_angle_0"][ti, ai] = _de_mean(lab_s_ppl, ref_lab[ti], tm)

        if (ai + 1) % 10 == 0:
            print(f"    angle {ang} deg done")

    # ── signal computation + harmonic fit ─────────────────────────────
    rows: list[dict[str, Any]] = []

    for ti, (roi_label, tx, ty) in enumerate(tiles):
        base = {"sample_id": sample.sample_id,
                "roi_label": roi_label, "tile_x": tx, "tile_y": ty}

        # A. Intensity channels ────────────────────────────────────────
        for ch in INTENSITY_CHS:
            sp = td[ch]["sample_PPL"][ti]
            sx = td[ch]["sample_XPL"][ti]
            bp = td[ch]["blank_PPL"][ti]
            bx = td[ch]["blank_XPL"][ti]

            bmed = float(np.nanmedian(bp))
            tau = 0.02 * max(bmed, EPS)
            ec = 0.005 * max(bmed, EPS)

            xfilm = sx - bx
            tfilm = np.where(bp >= tau, sp / (bp + ec), np.nan)
            afilm = np.where(np.isfinite(tfilm) & (tfilm > 0),
                             -np.log10(np.maximum(tfilm, 1e-6)), np.nan)
            xnb = np.where(bp >= tau, xfilm / (bp + ec), np.nan)
            xns = np.where(sp >= tau, xfilm / (sp + ec), np.nan)

            for sn, sv in [("Tfilm", tfilm), ("Afilm", afilm),
                           ("Xfilm", xfilm), ("Xnorm_blank", xnb),
                           ("Xnorm_sample", xns)]:
                rows.append({**base, "channel": ch, "channel_type": "intensity",
                             "signal": sn, **_fit_row(sv)})

        # B. Opponent channels ─────────────────────────────────────────
        for ch in OPPONENT_CHS:
            sp = td[ch]["sample_PPL"][ti]
            sx = td[ch]["sample_XPL"][ti]
            bp = td[ch]["blank_PPL"][ti]
            bx = td[ch]["blank_XPL"][ti]

            for sn, sv in [("Dblank_PPL", sp - bp),
                           ("Dblank_XPL", sx - bx),
                           ("Dxp_pp", sx - sp)]:
                rows.append({**base, "channel": ch, "channel_type": "opponent",
                             "signal": sn, **_fit_row(sv)})

        # C. Hue ───────────────────────────────────────────────────────
        hp = td["H"]["sample_PPL"][ti]
        hx = td["H"]["sample_XPL"][ti]
        hbx = td["H"]["blank_XPL"][ti]

        dh_xpp = circ_diff_vec(hx, hp)
        dh_a0 = circ_diff_vec(hp, np.full_like(hp, hp[0]))
        dh_sb = circ_diff_vec(hx, hbx)

        for sn, sv in [("dH_xpl_ppl", dh_xpp),
                       ("dH_angle_0", dh_a0),
                       ("dH_sample_blank", dh_sb)]:
            rows.append({**base, "channel": "H", "channel_type": "hue",
                         "signal": sn, **_fit_row(sv)})

        # D. DeltaE ────────────────────────────────────────────────────
        for sn in DELTAE_SIGS:
            rows.append({**base, "channel": DELTAE_CH, "channel_type": "deltaE",
                         "signal": sn, **_fit_row(de_data[sn][ti])})

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(tile_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in tile_rows:
        groups[(r["sample_id"], r["roi_label"],
                r["channel"], r["channel_type"], r["signal"])].append(r)

    out: list[dict] = []
    for key, rows in sorted(groups.items()):
        sid, rl, ch, ct, sig = key
        n = len(rows)

        def _s(field: str) -> tuple[float, float, float, float]:
            v = np.array([r[field] for r in rows], dtype=np.float64)
            f = v[np.isfinite(v)]
            if f.size == 0:
                return (float("nan"),) * 4  # type: ignore[return-value]
            return (float(np.median(f)), float(np.mean(f)),
                    float(np.percentile(f, 90)), float(np.std(f)))

        a2m, a2mn, a2p, a2s = _s("A2")
        a4m, a4mn, a4p, a4s = _s("A4")
        nm, nmn, _, _ = _s("NRMSE")
        pm, pmn, _, _ = _s("peak_to_peak")
        a0m = float(np.nanmedian([r["a0"] for r in rows]))
        ldm = float(np.nanmedian([r["LD_purity"] for r in rows]))
        lbm = float(np.nanmedian([r["LB_purity"] for r in rows]))
        a4a2m = float(np.nanmedian([r["A4_A2_ratio"] for r in rows]))
        nv = sum(1 for r in rows if np.isfinite(r["A2"]))

        out.append({
            "sample_id": sid, "roi_label": rl,
            "channel": ch, "channel_type": ct, "signal": sig,
            "a0_median": a0m,
            "A2_median": a2m, "A2_mean": a2mn, "A2_p90": a2p, "A2_std": a2s,
            "A4_median": a4m, "A4_mean": a4mn, "A4_p90": a4p, "A4_std": a4s,
            "NRMSE_median": nm, "NRMSE_mean": nmn,
            "peak_to_peak_median": pm, "peak_to_peak_mean": pmn,
            "LD_purity_median": ldm, "LB_purity_median": lbm,
            "A4_A2_ratio_median": a4a2m,
            "valid_tile_count": nv, "total_tile_count": n,
            "valid_tile_fraction": nv / max(n, 1),
        })
    return out


# ---------------------------------------------------------------------------
# CSV / plot output
# ---------------------------------------------------------------------------

TILE_FIELDS = [
    "sample_id", "roi_label", "tile_x", "tile_y",
    "channel", "channel_type", "signal",
    "a0", "A2", "A4", "axis2_deg", "axis4_deg",
    "RMSE", "NRMSE", "peak_to_peak", "valid_angle_count",
    "LD_purity", "LB_purity", "A4_A2_ratio", "A2_A4_logratio",
]

ROI_FIELDS = [
    "sample_id", "roi_label", "channel", "channel_type", "signal",
    "a0_median",
    "A2_median", "A2_mean", "A2_p90", "A2_std",
    "A4_median", "A4_mean", "A4_p90", "A4_std",
    "NRMSE_median", "NRMSE_mean",
    "peak_to_peak_median", "peak_to_peak_mean",
    "LD_purity_median", "LB_purity_median", "A4_A2_ratio_median",
    "valid_tile_count", "total_tile_count", "valid_tile_fraction",
]


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.name}  ({len(rows)} rows)")


def plot_heatmap(roi_summary: list[dict], path: Path) -> None:
    if plt is None:
        print("  matplotlib unavailable, skipping heatmap")
        return

    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in roi_summary:
        if np.isfinite(r["NRMSE_median"]):
            agg[(r["channel"], r["signal"])].append(r["NRMSE_median"])

    ch_order = list(INTENSITY_CHS) + list(OPPONENT_CHS) + list(HUE_CHS) + [DELTAE_CH]
    sig_order = list(INTENSITY_SIGS) + list(OPPONENT_SIGS) + list(HUE_SIGS) + list(DELTAE_SIGS)
    used_ch = [c for c in ch_order if any((c, s) in agg for s in sig_order)]
    used_sig = [s for s in sig_order if any((c, s) in agg for c in ch_order)]
    if not used_ch or not used_sig:
        return

    mat = np.full((len(used_ch), len(used_sig)), np.nan)
    for i, c in enumerate(used_ch):
        for j, s in enumerate(used_sig):
            vs = agg.get((c, s), [])
            if vs:
                mat[i, j] = float(np.median(vs))

    fig, ax = plt.subplots(figsize=(max(14, len(used_sig) * 1.0),
                                    max(5, len(used_ch) * 0.45)))
    vmax = min(0.5, float(np.nanmax(mat))) if np.any(np.isfinite(mat)) else 0.5
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(used_sig)))
    ax.set_xticklabels(used_sig, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(used_ch)))
    ax.set_yticklabels(used_ch, fontsize=8)

    for i in range(len(used_ch)):
        for j in range(len(used_sig)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=5.5, color="white" if v > vmax * 0.55 else "black")

    # type-boundary lines
    cumul = 0
    for grp in [INTENSITY_CHS, OPPONENT_CHS, HUE_CHS]:
        cnt = sum(1 for c in grp if c in used_ch)
        cumul += cnt
        if 0 < cumul < len(used_ch):
            ax.axhline(cumul - 0.5, color="gray", lw=0.5, ls="--")
    cumul = 0
    for grp in [INTENSITY_SIGS, OPPONENT_SIGS, HUE_SIGS]:
        cnt = sum(1 for s in grp if s in used_sig)
        cumul += cnt
        if 0 < cumul < len(used_sig):
            ax.axvline(cumul - 0.5, color="gray", lw=0.5, ls="--")

    ax.set_title("Median NRMSE  (channel x signal)  —  lower = better fit", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="NRMSE")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 55)
    print("  Multi-Channel Signal Sweep  (55 ch x sig combos)")
    print("=" * 55)

    manifest, samples = load_manifest()
    all_rois = load_rois()

    valid_mask = rp.load_halfres_valid_mask(Path(manifest["rotation_valid_mask"]))
    center_half = rp.get_center_halfres(manifest)
    print(f"Center (half-res): {center_half}")

    tile_rows: list[dict] = []
    for sample in samples:
        tile_rows.extend(
            process_sample(sample, manifest, valid_mask,
                           center_half, all_rois[sample.sample_id])
        )

    write_csv(tile_rows, OUTPUT_DIR / "channel_sweep_tile_fits.csv", TILE_FIELDS)

    roi_summary = aggregate(tile_rows)
    write_csv(roi_summary, OUTPUT_DIR / "channel_sweep_roi_summary.csv", ROI_FIELDS)

    plot_heatmap(roi_summary, OUTPUT_DIR / "channel_sweep_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
