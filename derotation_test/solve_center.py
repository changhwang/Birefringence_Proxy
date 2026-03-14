# Solve rotation center from landmark points observed at multiple angles.
# Then refine with per-angle sub-pixel shift via phase correlation.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

REPO_ROOT = Path(__file__).resolve().parents[1]
ANGLES = (0, 45, 70, 90, 135, 160, 165)
ALL_ANGLES = tuple(range(0, 166, 5))
FALLBACK_OLD_CENTER = (1204.0, 528.0)

DEFAULT_RGB_DIR = REPO_ROOT / "data" / "Calibration" / "rotation_calibration_new" / "rotation_calibration_sample_rgb" / "normal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "derotation_test"

# Landmark points (full-res) — user-provided
# crop origin = (524, 44) for 1000x1000 crop centered on 2048x1088
DEFAULT_LANDMARKS = {
    0: (732, 373),
    45: (985, 139),
    90: (1296, 183),
    70: (1174, 120),
    135: (1496, 424),
    160: (1495, 578),
    165: (1492, 619),
}
ROI_HALF = 100  # half-size of ROI around landmark for phase correlation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve rotation center from landmark points.")
    p.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--crop-size", type=int, default=1000)
    p.add_argument("--landmarks", type=str, default=None,
                   help="Override landmarks: '0:x,y;45:x,y;90:x,y;135:x,y'")
    return p.parse_args()


def parse_landmarks(s: str) -> dict[int, tuple[float, float]]:
    result = {}
    for pair in s.split(";"):
        angle_s, coords = pair.strip().split(":")
        x, y = coords.strip().split(",")
        result[int(angle_s)] = (float(x), float(y))
    return result


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_rgb_path(rgb_dir: Path, angle_deg: int) -> Path | None:
    exact = rgb_dir / f"rotation_calibration_sample_normal_{angle_deg:03d}_rgb.tif"
    if exact.exists():
        return exact
    for f in rgb_dir.glob(f"*_{angle_deg:03d}_*rgb*.tif"):
        return f
    return None


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def rot_cw_matrix(theta_deg: float) -> np.ndarray:
    """2x2 CW rotation in image coords (y-down). R(90)*(1,0)=(0,1)=down=CW visually."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)],
                     [math.sin(t), math.cos(t)]], dtype=np.float64)


# ---- Phase 1: Solve center from landmarks ----

def predicted_position(center: np.ndarray, p0: np.ndarray, theta_deg: float) -> np.ndarray:
    """Where P(0) ends up at angle theta if rotated CW around center."""
    R = rot_cw_matrix(theta_deg)
    return center + R @ (p0 - center)


def residuals(center_flat: np.ndarray,
              p0: np.ndarray,
              observations: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Residual vector for least-squares: predicted - observed for each angle."""
    center = center_flat.reshape(2)
    res = []
    for theta, p_obs in observations:
        p_pred = predicted_position(center, p0, theta)
        res.append(p_pred[0] - p_obs[0])
        res.append(p_pred[1] - p_obs[1])
    return np.array(res, dtype=np.float64)


def solve_center(landmarks: dict[int, tuple[float, float]]) -> tuple[tuple[float, float], dict]:
    p0 = np.array(landmarks[0], dtype=np.float64)
    observations = []
    for angle in sorted(landmarks):
        if angle == 0:
            continue
        observations.append((float(angle), np.array(landmarks[angle], dtype=np.float64)))

    # Initial guess: image center
    x0 = np.array([1024.0, 544.0], dtype=np.float64)

    result = optimize.least_squares(residuals, x0, args=(p0, observations), method="lm")
    center = (float(result.x[0]), float(result.x[1]))

    # Compute per-angle residuals
    per_angle = {}
    for theta, p_obs in observations:
        p_pred = predicted_position(result.x, p0, theta)
        err = np.linalg.norm(p_pred - p_obs)
        per_angle[int(theta)] = {
            "predicted": (float(p_pred[0]), float(p_pred[1])),
            "observed": (float(p_obs[0]), float(p_obs[1])),
            "error_px": float(err),
        }

    info = {
        "center": center,
        "cost": float(result.cost),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "per_angle_residuals": per_angle,
    }
    return center, info


# ---- Image operations ----

def rotate_gray(arr: np.ndarray, angle_deg: float, center_xy: tuple[float, float]) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    rotated = image.rotate(angle_deg, resample=Image.Resampling.BILINEAR,
                           expand=False, center=center_xy, fillcolor=0.0)
    return np.asarray(rotated, dtype=np.float32)


def rotate_rgb(arr: np.ndarray, angle_deg: float, center_xy: tuple[float, float]) -> np.ndarray:
    image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")
    rotated = image.rotate(angle_deg, resample=Image.Resampling.BILINEAR,
                           expand=False, center=center_xy, fillcolor=(0, 0, 0))
    return np.asarray(rotated, dtype=np.float32) / 255.0


def shift_gray(arr: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    sx, sy = shift_xy
    return ndimage.shift(arr.astype(np.float32), (float(sy), float(sx)),
                         order=1, mode="constant", cval=0.0).astype(np.float32)


def extract_roi(arr: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(w, x0 + 2 * half)
    y1 = min(h, y0 + 2 * half)
    return arr[y0:y1, x0:x1].copy()


def phase_correlation_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    if ref.shape != mov.shape or ref.size == 0:
        return 0.0, 0.0, 0.0
    ref64 = ref.astype(np.float64) - ref.mean()
    mov64 = mov.astype(np.float64) - mov.mean()
    window = np.outer(np.hanning(ref64.shape[0]), np.hanning(ref64.shape[1]))
    ref64 *= window
    mov64 *= window
    ref_fft = np.fft.fft2(ref64)
    mov_fft = np.fft.fft2(mov64)
    cross = ref_fft * np.conj(mov_fft)
    mag = np.abs(cross)
    if not np.any(mag > 0):
        return 0.0, 0.0, 0.0
    cross /= np.maximum(mag, 1e-12)
    corr = np.abs(np.fft.ifft2(cross))
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    shifts = np.array(peak_idx, dtype=np.float64)
    shape = np.array(corr.shape, dtype=np.float64)
    shifts = np.where(shifts > shape / 2.0, shifts - shape, shifts)
    dy, dx = shifts
    return float(dx), float(dy), float(corr[peak_idx])


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    af -= af.mean()
    bf -= bf.mean()
    d = np.sqrt(np.sum(af**2)) * np.sqrt(np.sum(bf**2))
    if d < 1e-12:
        return 0.0
    return float(np.sum(af * bf) / d)


def crop_center(arr: np.ndarray, size: int, cxy: tuple[float, float]) -> np.ndarray:
    h, w = arr.shape[0], arr.shape[1]
    half = size / 2.0
    x0 = int(max(0, cxy[0] - half))
    y0 = int(max(0, cxy[1] - half))
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1].copy()
    return arr[y0:y1, x0:x1, :].copy()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.landmarks:
        landmarks = parse_landmarks(args.landmarks)
    else:
        landmarks = dict(DEFAULT_LANDMARKS)

    print("=== Phase 1: Solve rotation center from landmarks ===")
    print("Landmarks (full-res):")
    for angle in sorted(landmarks):
        print(f"  {angle:>3d} deg: ({landmarks[angle][0]:.0f}, {landmarks[angle][1]:.0f})")

    center, info = solve_center(landmarks)
    print(f"\nSolved center: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"Residual norm: {info['residual_norm']:.4f} px")
    for angle, res in info["per_angle_residuals"].items():
        print(f"  {angle:>3d} deg: predicted ({res['predicted'][0]:.1f}, {res['predicted'][1]:.1f})"
              f"  observed ({res['observed'][0]:.1f}, {res['observed'][1]:.1f})"
              f"  error={res['error_px']:.2f} px")

    # Load images — all 34 angles
    rgb_dir = args.rgb_dir
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")

    gray_stack: dict[int, np.ndarray] = {}
    rgb_stack: dict[int, np.ndarray] = {}
    for angle in ALL_ANGLES:
        path = find_rgb_path(rgb_dir, angle)
        if path is None:
            print(f"  Warning: no RGB for {angle} deg, skipping")
            continue
        rgb_stack[angle] = load_rgb(path)
        gray_stack[angle] = rgb_to_gray(rgb_stack[angle])

    available_angles = sorted(gray_stack.keys())
    print(f"Loaded {len(available_angles)} angles: {available_angles[0]}..{available_angles[-1]} deg")

    # ---- Phase 1.5: Auto-refine landmarks via template matching ----
    lm0_ref = landmarks[0]
    ref_gray_0 = rotate_gray(gray_stack[0], 0.0, center)
    ref_template = extract_roi(ref_gray_0, lm0_ref[0], lm0_ref[1], ROI_HALF)

    REFINE_RADIUS = 20  # search ±20px
    REFINE_STEP = 2
    refined_any = False

    for angle in sorted(landmarks):
        if angle == 0 or angle not in gray_stack:
            continue
        derot = rotate_gray(gray_stack[angle], float(angle), center)

        # Current NCC at lm0 position
        current_patch = extract_roi(derot, lm0_ref[0], lm0_ref[1], ROI_HALF)
        current_ncc = ncc(ref_template, current_patch)

        if current_ncc > 0.95:
            continue

        # Grid search around lm0 in the derotated image
        best_ncc = current_ncc
        best_offset = (0, 0)
        for dy in range(-REFINE_RADIUS, REFINE_RADIUS + 1, REFINE_STEP):
            for dx in range(-REFINE_RADIUS, REFINE_RADIUS + 1, REFINE_STEP):
                cx = lm0_ref[0] + dx
                cy = lm0_ref[1] + dy
                patch = extract_roi(derot, cx, cy, ROI_HALF)
                if patch.shape != ref_template.shape:
                    continue
                score = ncc(ref_template, patch)
                if score > best_ncc:
                    best_ncc = score
                    best_offset = (dx, dy)

        if best_offset != (0, 0):
            # Back-project: the offset in derotated space corresponds to where
            # the feature actually ended up. We adjust the original landmark
            # by applying the inverse rotation to the offset.
            t = math.radians(angle)
            cos_t, sin_t = math.cos(t), math.sin(t)
            # Inverse of CW rotation (= CCW) applied to offset
            odx, ody = best_offset
            orig_dx = cos_t * odx + sin_t * ody
            orig_dy = -sin_t * odx + cos_t * ody
            old_lm = landmarks[angle]
            new_lm = (old_lm[0] + orig_dx, old_lm[1] + orig_dy)
            landmarks[angle] = new_lm
            refined_any = True
            print(f"  Refined {angle:>3d} deg landmark: ({old_lm[0]:.0f},{old_lm[1]:.0f}) -> "
                  f"({new_lm[0]:.1f},{new_lm[1]:.1f})  NCC {current_ncc:.4f} -> {best_ncc:.4f}")

    if refined_any:
        print("\nRe-solving center with refined landmarks...")
        center, info = solve_center(landmarks)
        print(f"Refined center: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"Residual norm: {info['residual_norm']:.4f} px")
        for angle, res in info["per_angle_residuals"].items():
            print(f"  {angle:>3d} deg: error={res['error_px']:.2f} px")
    else:
        print("\nAll landmarks NCC > 0.95, no refinement needed.")

    # ---- Phase 2: Per-angle shift for ALL available angles ----
    print(f"\n=== Phase 2: Per-angle shift (all {len(available_angles)} angles) ===")

    NCC_WARN_THRESHOLD = 0.80
    lm0 = landmarks[0]
    ref_derot = rotate_gray(gray_stack[0], 0.0, center)
    ref_roi = extract_roi(ref_derot, lm0[0], lm0[1], ROI_HALF)

    translation_table: dict[int, dict] = {
        0: {"angle_deg": 0.0, "shift_x_px": 0.0, "shift_y_px": 0.0,
            "phase_corr_peak": 1.0, "ncc_before": 1.0, "ncc_after": 1.0, "quality": "ok"}
    }
    suspect_angles: list[int] = []

    # Save suspect crops here
    suspect_dir = args.output_dir / "suspect_crops"
    ensure_dir(suspect_dir)

    for angle in available_angles:
        if angle == 0:
            continue
        derot = rotate_gray(gray_stack[angle], float(angle), center)
        patch = extract_roi(derot, lm0[0], lm0[1], ROI_HALF)
        ncc_before = ncc(ref_roi, patch)
        dx, dy, peak = phase_correlation_shift(ref_roi, patch)

        mag = math.hypot(dx, dy)
        if mag > 50.0:
            scale = 50.0 / mag
            dx *= scale
            dy *= scale

        shifted = shift_gray(derot, (dx, dy))
        patch_after = extract_roi(shifted, lm0[0], lm0[1], ROI_HALF)
        ncc_after = ncc(ref_roi, patch_after)

        if ncc_after <= ncc_before:
            dx, dy = 0.0, 0.0
            ncc_after = ncc_before

        quality = "ok" if ncc_after >= NCC_WARN_THRESHOLD else "suspect"
        if quality == "suspect":
            suspect_angles.append(angle)

        translation_table[angle] = {
            "angle_deg": float(angle),
            "shift_x_px": float(dx),
            "shift_y_px": float(dy),
            "phase_corr_peak": float(peak),
            "ncc_before": float(ncc_before),
            "ncc_after": float(ncc_after),
            "quality": quality,
        }

        flag = " *** SUSPECT" if quality == "suspect" else ""
        print(f"  {angle:>3d} deg: dx={dx:+.2f} dy={dy:+.2f}  "
              f"NCC {ncc_before:.4f} -> {ncc_after:.4f}{flag}")

    # Save suspect angle crops (ref ROI + derotated ROI side by side)
    if suspect_angles:
        print(f"\n{len(suspect_angles)} suspect angles (NCC < {NCC_WARN_THRESHOLD}): {suspect_angles}")
        for angle in suspect_angles:
            derot = rotate_gray(gray_stack[angle], float(angle), center)
            t = translation_table[angle]
            shifted = shift_gray(derot, (t["shift_x_px"], t["shift_y_px"]))
            patch = extract_roi(shifted, lm0[0], lm0[1], ROI_HALF)
            # Normalize and save side-by-side: ref | derotated
            ref_n = ((ref_roi - ref_roi.min()) / max(ref_roi.max() - ref_roi.min(), 1e-6) * 255).astype(np.uint8)
            pat_n = ((patch - patch.min()) / max(patch.max() - patch.min(), 1e-6) * 255).astype(np.uint8)
            combined = np.hstack([ref_n, np.full((ref_n.shape[0], 4), 128, dtype=np.uint8), pat_n])
            Image.fromarray(combined, mode="L").save(suspect_dir / f"suspect_{angle:03d}deg_ref_vs_derot.png")
        # Also save full-frame derotated crops for suspect angles
        size = args.crop_size
        for angle in suspect_angles:
            derot_full = rotate_gray(gray_stack[angle], float(angle), center)
            t = translation_table[angle]
            shifted_full = shift_gray(derot_full, (t["shift_x_px"], t["shift_y_px"]))
            crop_full = crop_center(shifted_full, size, center)
            lo, hi = np.percentile(crop_full, [1, 99])
            norm = np.clip((crop_full - lo) / max(hi - lo, 1e-6), 0, 1)
            Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(
                suspect_dir / f"suspect_{angle:03d}deg_full_crop.png")
        print(f"Saved suspect crops to: {suspect_dir}")
    else:
        print(f"\nAll angles above NCC threshold ({NCC_WARN_THRESHOLD})")

    # ---- Save results ----
    result = {
        "landmarks_fullres": {str(a): list(landmarks[a]) for a in sorted(landmarks)},
        "solved_center": {"x": center[0], "y": center[1]},
        "old_center": {"x": FALLBACK_OLD_CENTER[0], "y": FALLBACK_OLD_CENTER[1]},
        "solve_info": info,
        "ncc_threshold": NCC_WARN_THRESHOLD,
        "suspect_angles": suspect_angles,
        "translation_table": [translation_table.get(a, {"angle_deg": float(a), "shift_x_px": 0.0, "shift_y_px": 0.0, "quality": "missing"})
                               for a in ALL_ANGLES],
    }
    result_path = args.output_dir / "solved_correction.json"
    save_json(result_path, result)
    print(f"\nSaved: {result_path}")

    # ---- Figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    size = args.crop_size

    def normalize_gray(g: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(g, [1, 99])
        return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    # --- 1. Mean overlay of all 34 angles ---
    all_derot_crops: list[np.ndarray] = []
    for angle in available_angles:
        derot = rotate_gray(gray_stack[angle], float(angle), center)
        t = translation_table.get(angle, {})
        shifted = shift_gray(derot, (t.get("shift_x_px", 0.0), t.get("shift_y_px", 0.0)))
        crop = crop_center(shifted, size, center)
        all_derot_crops.append(normalize_gray(crop))

    mean_overlay = np.mean(all_derot_crops, axis=0)
    Image.fromarray((mean_overlay * 255).astype(np.uint8), mode="L").save(
        args.output_dir / "overlay_all_angles.png")
    print(f"Saved: overlay_all_angles.png")

    # --- 2. NCC per angle ---
    ncc_angles = sorted(a for a in translation_table if a in available_angles)
    ncc_before_vals = [translation_table[a]["ncc_before"] for a in ncc_angles]
    ncc_after_vals = [translation_table[a]["ncc_after"] for a in ncc_angles]

    fig_ncc, ax_ncc = plt.subplots(1, 1, figsize=(12, 5))
    ax_ncc.plot(ncc_angles, ncc_before_vals, "o-", label="NCC before shift", markersize=4, alpha=0.7)
    ax_ncc.plot(ncc_angles, ncc_after_vals, "s-", label="NCC after shift", markersize=4)
    ax_ncc.axhline(y=NCC_WARN_THRESHOLD, color="red", linestyle="--", alpha=0.5,
                   label=f"Threshold ({NCC_WARN_THRESHOLD})")
    for a in suspect_angles:
        ax_ncc.axvline(x=a, color="red", alpha=0.2, linewidth=3)
    ax_ncc.set_xlabel("Angle (deg)")
    ax_ncc.set_ylabel("NCC vs 0 deg")
    ax_ncc.set_title(f"Alignment quality per angle -- center ({center[0]:.1f}, {center[1]:.1f})")
    ax_ncc.legend()
    ax_ncc.set_xlim(-5, 170)
    ax_ncc.set_ylim(-0.1, 1.05)
    ax_ncc.grid(True, alpha=0.3)
    fig_ncc.tight_layout()
    fig_ncc.savefig(args.output_dir / "ncc_per_angle.png", dpi=150)
    plt.close(fig_ncc)
    print(f"Saved: ncc_per_angle.png")

    # --- 3. Original vs Derotated comparison (0/45/90/135, 2 rows) ---
    compare_angles = (0, 45, 90, 135)
    img_center = (gray_stack[0].shape[1] / 2.0, gray_stack[0].shape[0] / 2.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, angle in enumerate(compare_angles):
        # Row 0: Original
        orig_crop = normalize_gray(crop_center(gray_stack[angle], size, img_center))
        axes[0, i].imshow(orig_crop, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"{angle} deg (original)", fontsize=11)
        axes[0, i].axis("off")

        # Row 1: Derotated + shift
        derot = rotate_gray(gray_stack[angle], float(angle), center)
        t = translation_table.get(angle, {})
        shifted = shift_gray(derot, (t.get("shift_x_px", 0.0), t.get("shift_y_px", 0.0)))
        derot_crop = normalize_gray(crop_center(shifted, size, center))
        ncc_val = translation_table.get(angle, {}).get("ncc_after", 1.0)
        axes[1, i].imshow(derot_crop, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"{angle} -> 0 deg  (NCC {ncc_val:.3f})", fontsize=11)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Derotated", fontsize=12, rotation=90, labelpad=10)
    fig.suptitle(f"Original vs Derotated -- center ({center[0]:.1f}, {center[1]:.1f}), {size}x{size} crop",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output_dir / "compare_original_vs_derotated.png", dpi=150)
    plt.close(fig)
    print(f"Saved: compare_original_vs_derotated.png")


if __name__ == "__main__":
    main()
