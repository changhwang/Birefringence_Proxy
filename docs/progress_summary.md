# Progress Summary

This document is a single-entry summary of what has been done so far in the birefringence proxy project, what code and presets were created, and what the current technical status is.

## 1. Measurement And Data Characterization

We first froze the acquisition interpretation:

- `PPL = normal`
- `XPL = crosspol`
- camera raw format is `XI_RAW16`
- actual pixel storage is `10-bit Bayer GBRG in uint16 TIFF`
- preview RGB is only for visual inspection and figure generation
- quantitative analysis uses raw Bayer data, not preview RGB

Detailed reference:

- [Measurement And Raw Data Format](measurement_and_raw_data_format.md)
- [Preview RGB For Figures](preview_rgb_for_figures.md)

## 2. Initial Rotation Calibration

The original `rotation_calibration` dataset was used to estimate a common derotation center and all-angle valid region.

Main result:

- rotation center: `(1204.0, 528.0)`
- all-angle valid fraction: about `41.1%`
- largest conservative safe rectangle: `x=799, y=145, w=765, h=766`

This geometry became the first foundation for ROI work.

Detailed reference:

- [Rotation Calibration](rotation_calibration.md)
- script: [../scripts/derive_derotation_geometry.py](../scripts/derive_derotation_geometry.py)

## 3. ROI Review And Preset Building

ROI design proceeded in two layers.

### 3.1 Horizontal Center 3-Split

We first built the horizontal split preset for global sample comparison.

- long samples `s1-s8`: `ROI_L`, `ROI_C`, `ROI_R`
- short samples `s9-s13`: one centered ROI per size
- preset file: [../configs/roi_presets/horizontal_center_3split.json](../configs/roi_presets/horizontal_center_3split.json)

### 3.2 Refined 3x3 Grid

We then built a more granular preset for ROI refinement.

- long samples `s1-s8`: `ROI_UL`, `ROI_UC`, `ROI_UR`, `ROI_CL`, `ROI_CC`, `ROI_CR`, `ROI_LL`, `ROI_LC`, `ROI_LR`
- short samples `s9-s13`: `ROI_UC`, `ROI_CC`, `ROI_LC`
- sizes reviewed: `150`, `200`, `300`
- preset file: [../configs/roi_presets/refined_grid_3x3_v1.json](../configs/roi_presets/refined_grid_3x3_v1.json)
- middle row is now computed as column-wise midpoint between top and bottom rows

This refinement was especially used to manually tune short-frame ROIs and to update sample `s7` after the source preview image changed.

Detailed reference:

- [ROI Selection And Custom Presets](roi_selection_and_custom_presets.md)
- script: [../scripts/build_roi_review.py](../scripts/build_roi_review.py)
- script: [../scripts/build_custom_roi_review.py](../scripts/build_custom_roi_review.py)
- script: [../scripts/build_refined_roi_grid_review.py](../scripts/build_refined_roi_grid_review.py)

## 4. Raw Analysis Pipeline V1

The raw analysis pipeline was then implemented around the following order:

`raw load -> Bayer split -> dark subtraction -> exposure normalization -> derotation -> ROI/tile extraction -> harmonic fit -> ROI summary -> sample summary`

Key decisions:

- raw Bayer direct derotation is disabled
- channel planes are `G1`, `G2`, `R`, `B`
- primary analysis channel is `G`
- harmonic fit uses simultaneous `2theta + 4theta`
- ROI primary summary is median of valid tile primary metrics
- sample primary summary is median of ROI primary summaries

Implemented files:

- entrypoint: [../scripts/run_analysis_pipeline.py](../scripts/run_analysis_pipeline.py)
- manifest: [../configs/datasets/analysis_manifest.json](../configs/datasets/analysis_manifest.json)
- tests: [../tests/test_run_analysis_pipeline.py](../tests/test_run_analysis_pipeline.py)

Detailed reference:

- [Analysis Plan](analysis_plan.md)

## 5. Short-Frame QC Package

Because short samples `s9-s13` needed more focused inspection, a dedicated QC package was created.

Scope:

- short-frame only
- `200 x 200` refined ROI preset
- calibration QC, derotation visual QC, and signal sanity QC

Implemented files:

- package note: [../shortframe_calibration/README.md](../shortframe_calibration/README.md)
- script: [../shortframe_calibration/run_shortframe_qc.py](../shortframe_calibration/run_shortframe_qc.py)
- local manifest: [../shortframe_calibration/shortframe_manifest.json](../shortframe_calibration/shortframe_manifest.json)
- ROI spec: [../shortframe_calibration/shortframe_roi_200.json](../shortframe_calibration/shortframe_roi_200.json)

The package currently generates:

- Phase A calibration QC
- Phase A.5 derotation visual QC
- Phase B raw/corrected signal sanity QC

## 6. New Registration Target Calibration

The original rotation center was not sufficiently convincing for short-frame alignment checks, so a new printed registration target was acquired.

This new dataset was used to estimate a better center candidate:

- new center candidate: `(953.5, 613.0)`

We then implemented angle-wise post-rotation translation estimation on top of that center.

Implemented file:

- [../scripts/analyze_rotation_calibration_new.py](../scripts/analyze_rotation_calibration_new.py)

Generated reusable correction spec:

- [../analysis_outputs/rotation_calibration_new/rotation_translation_correction.json](../analysis_outputs/rotation_calibration_new/rotation_translation_correction.json)

Important status note:

- the implementation for `rotation + translation` exists
- but on the current target QC metrics it does **not yet outperform** `rotation-only`
- so it should be treated as an experimental correction candidate, not yet as the default production geometry

## 7. Current Technical Status

What is working:

- data interpretation and acquisition conventions are documented
- original rotation calibration is implemented and reproducible
- ROI presets exist for both broad and refined review
- raw analysis pipeline v1 is implemented
- short-frame QC package is implemented
- new target-based rotation-plus-translation correction prototype is implemented

What is still open:

- choose whether to keep the original derotation geometry or switch to a revised one after better target-based validation
- decide whether registration QC should be redefined with a more stable metric
- continue ROI refinement where sample-specific adjustment is still needed
- compare final proxy outputs against Mueller-matrix `LD` / `LB` summaries

## 8. Suggested Next Reading

If the goal is to continue the project from a fresh clone, the most practical order is:

1. [Measurement And Raw Data Format](measurement_and_raw_data_format.md)
2. [Rotation Calibration](rotation_calibration.md)
3. [ROI Selection And Custom Presets](roi_selection_and_custom_presets.md)
4. [Analysis Plan](analysis_plan.md)
5. [../shortframe_calibration/README.md](../shortframe_calibration/README.md)
6. [Project Structure](project_structure.md)
