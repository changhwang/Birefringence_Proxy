from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "run_analysis_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_analysis_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RunAnalysisPipelineTests(unittest.TestCase):
    def test_split_bayer_gbrg_block_grid(self) -> None:
        raw = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
        split = MODULE.split_bayer_gbrg_block_grid(raw)
        np.testing.assert_array_equal(split["G1"], raw[0::2, 0::2])
        np.testing.assert_array_equal(split["B"], raw[0::2, 1::2])
        np.testing.assert_array_equal(split["R"], raw[1::2, 0::2])
        np.testing.assert_array_equal(split["G2"], raw[1::2, 1::2])

    def test_fit_harmonic_curve_recovers_expected_amplitudes(self) -> None:
        theta = np.deg2rad(np.asarray(MODULE.ANGLES_DEG, dtype=np.float64))
        values = 2.0 + 0.6 * np.cos(2.0 * theta) + 0.8 * np.sin(4.0 * theta)
        fit = MODULE.fit_harmonic_curve(values, np.ones_like(values, dtype=bool), min_valid_angles=10)
        self.assertTrue(fit["fit_valid"])
        self.assertAlmostEqual(fit["A2"], 0.6, places=6)
        self.assertAlmostEqual(fit["A4"], 0.8, places=6)

    def test_compute_signal_values_preserves_signed_xfilm(self) -> None:
        values, valid = MODULE.compute_signal_values_for_scalar(
            sample_ppl=5.0,
            sample_xpl=2.0,
            blank_ppl=4.0,
            blank_xpl=3.5,
            eps_count=0.1,
            tau_low=0.5,
        )
        self.assertTrue(valid["Xfilm"])
        self.assertLess(values["Xfilm"], 0.0)
        self.assertTrue(valid["Afilm_PPL"])

    def test_compute_signal_values_invalidates_nonpositive_transmission(self) -> None:
        values, valid = MODULE.compute_signal_values_for_scalar(
            sample_ppl=-1.0,
            sample_xpl=1.0,
            blank_ppl=4.0,
            blank_xpl=0.5,
            eps_count=0.1,
            tau_low=0.5,
        )
        self.assertFalse(valid["Afilm_PPL"])
        self.assertTrue(valid["Xfilm"])
        self.assertTrue(math.isnan(values["Afilm_PPL"]))

    def test_validate_no_preview_paths_rejects_preview_dirs(self) -> None:
        manifest = {
            "samples": [
                {
                    "sample_dir": "data/QLB/RGB/QLB_s1_rgb",
                    "blank_dir": "data/Calibration/blank_longframe",
                    "empty_dir": "data/Calibration/empty_longframe",
                    "dark_dir": "data/Calibration/dark",
                }
            ]
        }
        with self.assertRaises(ValueError):
            MODULE.validate_no_preview_paths(manifest)

    def test_circular_mean_deg_respects_axis_period(self) -> None:
        values = np.array([179.0, 1.0], dtype=np.float64)
        mean_deg = MODULE.circular_mean_deg(values, 180.0)
        self.assertTrue(mean_deg < 5.0 or mean_deg > 175.0)


if __name__ == "__main__":
    unittest.main()
