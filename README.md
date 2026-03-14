# Birefringence Proxy

Working repository for raw-image-based birefringence proxy analysis on printed organic conjugated thin films.

## What Is In This Repo

- documentation for measurement, calibration, ROI design, and analysis planning
- raw-analysis scripts for Bayer TIFF processing
- ROI preset JSON files
- short-frame QC utilities
- tests for the main analysis pipeline

## Start Here

- [docs/README.md](docs/README.md)
- [docs/progress_summary.md](docs/progress_summary.md)

## Main Entrypoints

- analysis pipeline: [scripts/run_analysis_pipeline.py](scripts/run_analysis_pipeline.py)
- derotation correction: [derotation_test/solve_center.py](derotation_test/solve_center.py)
- short-frame QC: [shortframe_calibration/run_shortframe_qc.py](shortframe_calibration/run_shortframe_qc.py)

## Repository Scope

Raw TIFF data and generated outputs are intentionally excluded from version control. This repo is meant to track code, configuration, and documentation.
