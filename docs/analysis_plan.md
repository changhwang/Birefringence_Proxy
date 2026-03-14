# Analysis Pipeline V1

This document describes the implemented raw birefringence analysis pipeline entrypoint and its default configuration.

The upstream geometry and acquisition conventions should be read first:

- [measurement_and_raw_data_format.md](../docs/measurement_and_raw_data_format.md)
- [rotation_calibration.md](../docs/rotation_calibration.md)
- [roi_selection_and_custom_presets.md](../docs/roi_selection_and_custom_presets.md)

## 1. Entrypoint

The pipeline is implemented in:

- [run_analysis_pipeline.py](../scripts/run_analysis_pipeline.py)

The default manifest is:

- [analysis_manifest.json](../configs/datasets/analysis_manifest.json)

## 2. Default sample mapping

The current manifest freezes these defaults:

- `QLB_s1` to `QLB_s8`
  - sample type: `long`
  - blank reference: `blank_longframe`
  - empty QC reference: `empty_longframe`
  - primary ROI size: `500`
- `QLB_s9` to `QLB_s13`
  - sample type: `short`
  - blank reference: `blank_shortframe`
  - empty QC reference: `empty_shortframe`
  - primary ROI size: `400`
- all samples
  - dark reference: `dark`

If a later pass needs different primary ROI sizes, that should be changed in the manifest, not hard-coded inside the script.

## 3. Stage usage

Run all QC outputs:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage qc
```

Extract tile-level signal bundles:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage extract
```

Fit tile-level harmonic models:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage fit
```

Build final sample summary CSV:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage summarize
```

Compare against Mueller-matrix CSV:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage compare-mm --mm-csv path\to\mm_summary.csv
```

Run only a subset of samples:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_analysis_pipeline.py --stage fit --samples QLB_s1 QLB_s5 QLB_s9
```

## 4. Output layout

The pipeline writes by default to:

- `analysis_outputs/pipeline_runs/current/qc`
- `analysis_outputs/pipeline_runs/current/signals`
- `analysis_outputs/pipeline_runs/current/fits`
- `analysis_outputs/pipeline_runs/current/summary`
- `analysis_outputs/pipeline_runs/current/compare_mm`

The important v1 outputs are:

- per-sample QC JSON and QC plots
- per-sample extraction bundles (`.npz` + metadata JSON)
- per-sample tile fit CSV
- per-sample ROI summary CSV
- per-sample summary JSON
- final summary CSV keyed by `sample_id`
- optional Mueller-matrix comparison report

## 5. Implemented defaults

The script implements these v1 defaults:

- `PPL = normal`
- `XPL = crosspol`
- `GBRG` Bayer split to `G1`, `G2`, `R`, `B`
- raw mosaic direct derotation disabled
- derotation interpolation = `bilinear`
- tile size = `64 x 64`
- tile stride = `32`
- valid tile fraction threshold = `0.9`
- default fit = simultaneous unweighted `2theta + 4theta` linear least squares
- primary channel = `G`
- secondary confirmation channels = `R`, `B`

## 6. Test entrypoint

Basic unit tests are provided in:

- [test_run_analysis_pipeline.py](../tests/test_run_analysis_pipeline.py)

Run them with:

```powershell
.\.venv\Scripts\python.exe -m unittest .\tests\test_run_analysis_pipeline.py
```

