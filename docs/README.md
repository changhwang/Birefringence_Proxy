# Birefringence Proxy Docs

This folder collects the working documentation for the birefringence proxy workflow.

## Recommended Reading Order

1. [Progress Summary](progress_summary.md)
2. [Measurement And Raw Data Format](measurement_and_raw_data_format.md)
3. [Rotation Calibration](rotation_calibration.md)
4. [ROI Selection And Custom Presets](roi_selection_and_custom_presets.md)
5. [Analysis Plan](analysis_plan.md)
6. [Preview RGB For Figures](preview_rgb_for_figures.md)
7. [Project Structure](project_structure.md)

## Related Code And Config

- Pipeline entrypoint: [../scripts/run_analysis_pipeline.py](../scripts/run_analysis_pipeline.py)
- New target registration analysis: [../scripts/analyze_rotation_calibration_new.py](../scripts/analyze_rotation_calibration_new.py)
- Short-frame QC package: [../shortframe_calibration/README.md](../shortframe_calibration/README.md)
- Dataset manifest: [../configs/datasets/analysis_manifest.json](../configs/datasets/analysis_manifest.json)
- Horizontal ROI preset: [../configs/roi_presets/horizontal_center_3split.json](../configs/roi_presets/horizontal_center_3split.json)
- Refined ROI preset: [../configs/roi_presets/refined_grid_3x3_v1.json](../configs/roi_presets/refined_grid_3x3_v1.json)

## Note About Data

Raw TIFF data and generated analysis outputs are not versioned in the repository. The repo contains code, configuration, and documentation only.

Some detailed documents still reference local `data/` and `analysis_outputs/` paths because those files are part of the working analysis environment. Those links are expected to resolve only when the corresponding local datasets and generated outputs are present.
