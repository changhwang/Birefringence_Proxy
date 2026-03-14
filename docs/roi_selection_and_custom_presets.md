# ROI Selection And Custom Presets Workflow

This document records how the sample ROI design step was carried out after the rotation calibration step had already established the usable geometric region.

Before reading this document, the acquisition and raw-format conventions should be read first:

- [measurement_and_raw_data_format.md](../docs/measurement_and_raw_data_format.md)

The purpose of this stage was not to compute birefringence values yet. The purpose was to define practical, reproducible ROIs for later angle-series analysis.

This stage answered these questions:

1. How should the safe all-angle geometry from rotation calibration be visualized on the sample images?
2. What centered square ROIs are reasonable starting points?
3. For a fixed `y` center, how far can a square ROI move left or right while still surviving the full angle sequence?
4. Which samples should use three span ROIs (`ROI_L`, `ROI_C`, `ROI_R`) and which samples should use a single custom ROI?

This document should be read after [rotation_calibration.md](../docs/rotation_calibration.md), because the ROI work depends directly on the rotation center and valid mask defined there.

## 1. Upstream geometry inherited from rotation calibration

The ROI step reused the following outputs from the rotation calibration stage:

- [derotation_geometry.json](../analysis_outputs/rotation_calibration/derotation_geometry.json)
- [rotation_calibration_valid_mask.png](../analysis_outputs/rotation_calibration/rotation_calibration_valid_mask.png)

These values were treated as the geometric foundation:

- rotation center: `(1204.0, 528.0)`
- largest conservative safe rectangle:
  - `x = 799`
  - `y = 145`
  - `width = 765`
  - `height = 766`
- full valid mask from the all-angle intersection:
  - valid fraction about `41.1%`

Important interpretation:

The green safe rectangle was used as a conservative visual anchor, but it was **not** treated as the absolute final boundary of all future ROIs. Later custom ROIs were allowed to extend outside the green rectangle if they still remained inside the full valid mask.

## 2. Data used in this step

The input images for ROI review were the sample-specific `0 deg` RGB images copied into `ROI_selection`.

Input folder:

- [data/ROI_selection](../data/ROI_selection)

Contents:

- `QLB_s1_normal_000_rgb.tif`
- `QLB_s2_normal_000_rgb.tif`
- `...`
- `QLB_s13_normal_000_rgb.tif`

These files were used as human-readable review images because:

- they are already demosaiced RGB,
- they are all at `0 deg`,
- they make it much easier to judge where the printed film sits inside the allowed region.

## 3. Scripts used

Two scripts were used for this stage:

- [build_roi_review.py](../scripts/build_roi_review.py)
- [build_custom_roi_review.py](../scripts/build_custom_roi_review.py)

Configuration file for the current custom ROI decisions:

- [horizontal_center_3split.json](../configs/roi_presets/horizontal_center_3split.json)

Main output folders:

- [safe_region_baseline](../analysis_outputs/roi_review/safe_region_baseline)
- [horizontal_center_3split](../analysis_outputs/roi_review/horizontal_center_3split)

## 4. Stage 1: First-pass overlay on all sample images

The first pass simply overlaid the rotation-calibration geometry onto the `0 deg` RGB sample images.

This initial output was meant to answer:

- where the green safe rectangle sits on each sample,
- how much of the printed film falls comfortably inside it,
- whether a global square ROI might be reasonable.

Base outputs:

- [roi_review_contact_sheet.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet.png)
- [roi_review_summary.json](../analysis_outputs/roi_review/safe_region_baseline/roi_review_summary.json)

Meaning of the overlays:

- green rectangle: largest conservative all-angle safe rectangle
- red-tinted outside region: pixels that do not survive the full angle range
- yellow cross: rotation center inherited from calibration

This pass was useful for orienting the later discussion, but it was immediately clear that some samples were small enough that a single large common ROI would not be optimal.

## 5. Stage 2: Centered square ROI candidates

The next step was to generate simple centered square candidates inside the safe rectangle.

Three square sizes were tested:

- `500 x 500`
- `400 x 400`
- `300 x 300`

The centered squares were:

- `500`: `x = 931`, `y = 278`
- `400`: `x = 981`, `y = 328`
- `300`: `x = 1031`, `y = 378`

Outputs:

- [roi_review_contact_sheet_sq500.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq500.png)
- [roi_review_contact_sheet_sq400.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq400.png)
- [roi_review_contact_sheet_sq300.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq300.png)

Why these centered squares were useful:

- they gave a quick sense of how much of the film is captured by a globally centered ROI,
- they showed that `500` can be attractive for coverage but sometimes touches less ideal regions,
- they showed that `400` and `300` can be more conservative for smaller or less centered samples.

## 6. Stage 3: Fixed-y span analysis

After looking at the centered squares, the next question was:

> if the `y` center is held fixed, how far can the square move left or right without losing any pixels over the full rotation sequence?

This led to the fixed-`y` span analysis.

### 6.1 Method

For each square size:

1. Fix the square height and width.
2. Fix the `y` center implied by the centered square.
3. Slide the square along `x`.
4. Keep only positions where the entire square stays inside the all-angle valid mask.
5. Record:
   - the leftmost valid square,
   - the centered square,
   - the rightmost valid square.

### 6.2 Span results

The fixed-`y` span results were:

For `500 x 500`:

- `y_top = 278`
- valid `x` range from `706` to `1166`
- centered square `x = 931`

For `400 x 400`:

- `y_top = 328`
- valid `x` range from `685` to `1289`
- centered square `x = 981`

For `300 x 300`:

- `y_top = 378`
- valid `x` range from `668` to `1407`
- centered square `x = 1031`

Span outputs:

- [roi_review_contact_sheet_sq500_span.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq500_span.png)
- [roi_review_contact_sheet_sq400_span.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq400_span.png)
- [roi_review_contact_sheet_sq300_span.png](../analysis_outputs/roi_review/safe_region_baseline/roi_review_contact_sheet_sq300_span.png)

Meaning of the span overlays:

- orange: leftmost valid square
- cyan: centered square
- pink: rightmost valid square
- yellow connecting line: the allowed `x` span at fixed `y`
- green rectangle: conservative safe rectangle

This step showed the full movement budget available for each square size and made it possible to design per-sample ROIs without accidentally leaving the valid region.

## 7. Stage 4: Sample-specific custom ROI design

After reviewing the centered and span sheets, custom ROI presets were created.

The key design decision was:

- `QLB_s1` through `QLB_s8` should generally keep three span positions.
- `QLB_s9` through `QLB_s13` are small enough that a single custom ROI is more appropriate.

This is the point where the naming convention `ROI_L`, `ROI_C`, `ROI_R` was adopted for the three-ROI span case.

Meaning:

- `ROI_L`: left ROI
- `ROI_C`: center ROI
- `ROI_R`: right ROI

For small samples that do not use three ROIs, the layout type is simply `single`.

## 8. Final current custom preset outputs

Current rendered outputs:

- [roi_review_contact_sheet_sq500_custom.png](../analysis_outputs/roi_review/horizontal_center_3split/roi_review_contact_sheet_sq500_custom.png)
- [roi_review_contact_sheet_sq400_custom.png](../analysis_outputs/roi_review/horizontal_center_3split/roi_review_contact_sheet_sq400_custom.png)
- [roi_review_contact_sheet_sq300_custom.png](../analysis_outputs/roi_review/horizontal_center_3split/roi_review_contact_sheet_sq300_custom.png)

Current summary:

- [custom_roi_review_summary.json](../analysis_outputs/roi_review/horizontal_center_3split/custom_roi_review_summary.json)

Current editable source of truth:

- [horizontal_center_3split.json](../configs/roi_presets/horizontal_center_3split.json)

## 9. Final current custom ROI coordinates

This section records the current working ROI presets exactly as stored in the config and output summary at the time this document was written.

### 9.1 `500 x 500` custom layout

Common geometry for this size:

- `y = 278`
- default span:
  - `ROI_L x = 706`
  - `ROI_C x = 931`
  - `ROI_R x = 1166`

Current per-sample decisions:

| Sample | Layout | Coordinates |
| --- | --- | --- |
| `QLB_s1` | span | `ROI_L x=706`, `ROI_C x=931`, `ROI_R x=1136` |
| `QLB_s2` | span | `ROI_L x=706`, `ROI_C x=931`, `ROI_R x=1136` |
| `QLB_s3` | span | default span |
| `QLB_s4` | span | default span |
| `QLB_s5` | span | `ROI_L x=706`, `ROI_C x=771`, `ROI_R x=1166` |
| `QLB_s6` | span | default span |
| `QLB_s7` | span | default span |
| `QLB_s8` | span | `ROI_L x=706`, `ROI_C x=931`, `ROI_R x=1136` |
| `QLB_s9` | single | `x=821`, `y=278` |
| `QLB_s10` | single | `x=931`, `y=278` |
| `QLB_s11` | single | `x=881`, `y=278` |
| `QLB_s12` | single | `x=881`, `y=278` |
| `QLB_s13` | single | `x=799`, `y=278` |

### 9.2 `400 x 400` custom layout

Common geometry for this size:

- `y = 328`
- default span:
  - `ROI_L x = 685`
  - `ROI_C x = 981`
  - `ROI_R x = 1289`

Current per-sample decisions:

| Sample | Layout | Coordinates |
| --- | --- | --- |
| `QLB_s1` | span | `ROI_L x=685`, `ROI_C x=981`, `ROI_R x=1259` |
| `QLB_s2` | span | `ROI_L x=685`, `ROI_C x=981`, `ROI_R x=1259` |
| `QLB_s3` | span | default span |
| `QLB_s4` | span | default span |
| `QLB_s5` | span | `ROI_L x=685`, `ROI_C x=831`, `ROI_R x=1289` |
| `QLB_s6` | span | default span |
| `QLB_s7` | span | default span |
| `QLB_s8` | span | `ROI_L x=685`, `ROI_C x=981`, `ROI_R x=1259` |
| `QLB_s9` | single | `x=881`, `y=328` |
| `QLB_s10` | single | `x=981`, `y=328` |
| `QLB_s11` | single | `x=931`, `y=328` |
| `QLB_s12` | single | `x=931`, `y=328` |
| `QLB_s13` | single | `x=799`, `y=328` |

### 9.3 `300 x 300` custom layout

Common geometry for this size:

- `y = 378`
- default span:
  - `ROI_L x = 668`
  - `ROI_C x = 1031`
  - `ROI_R x = 1407`

Current per-sample decisions:

| Sample | Layout | Coordinates |
| --- | --- | --- |
| `QLB_s1` | span | `ROI_L x=668`, `ROI_C x=1031`, `ROI_R x=1347` |
| `QLB_s2` | span | `ROI_L x=668`, `ROI_C x=1031`, `ROI_R x=1347` |
| `QLB_s3` | span | default span |
| `QLB_s4` | span | default span |
| `QLB_s5` | span | `ROI_L x=668`, `ROI_C x=901`, `ROI_R x=1407` |
| `QLB_s6` | span | default span |
| `QLB_s7` | span | default span |
| `QLB_s8` | span | `ROI_L x=668`, `ROI_C x=1031`, `ROI_R x=1377` |
| `QLB_s9` | single | `x=911`, `y=378` |
| `QLB_s10` | single | `x=1031`, `y=378` |
| `QLB_s11` | single | `x=1031`, `y=378` |
| `QLB_s12` | single | `x=1011`, `y=378` |
| `QLB_s13` | single | `x=799`, `y=378` |

## 10. Interpretation of the final ROI strategy

The current ROI strategy is intentionally mixed rather than forcing one rule on every sample.

The logic is:

- larger or broader samples can support three spatially separated ROIs,
- smaller or more localized samples should use one carefully placed ROI,
- all ROIs still inherit their validity from the all-angle mask defined in rotation calibration.

This is a pragmatic compromise between:

- comparability across samples,
- retaining enough film area,
- avoiding edge artifacts and empty background,
- keeping the ROI definitions simple enough to reproduce.

## 11. Why the custom step was necessary

A purely global ROI would have been convenient, but the sample images showed that this would be too rigid for several samples.

Main reasons the custom step was necessary:

- some films are not centered the same way,
- some films are smaller and do not justify three ROIs,
- some samples benefit from shifting only the right or center ROI,
- using the full span blindly would sometimes include less useful edge regions.

The custom preset approach keeps the workflow explicit and reproducible instead of relying on ad hoc manual choices each time analysis is run.

## 12. How to regenerate the ROI review outputs

### 12.1 Base ROI review and span sheets

```powershell
.\.venv\Scripts\python.exe .\scripts\build_roi_review.py
```

This regenerates:

- base safe-rectangle overlays,
- centered square sheets,
- fixed-`y` span sheets,
- [roi_review_summary.json](../analysis_outputs/roi_review/safe_region_baseline/roi_review_summary.json)

### 12.2 Current custom presets

```powershell
.\.venv\Scripts\python.exe .\scripts\build_custom_roi_review.py
```

This regenerates:

- the custom contact sheets for `500`, `400`, and `300`,
- [custom_roi_review_summary.json](../analysis_outputs/roi_review/horizontal_center_3split/custom_roi_review_summary.json)

## 13. Practical status at the end of this step

At the end of this ROI stage, the project now has:

- a fixed rotation center and all-angle valid mask,
- a conservative safe rectangle,
- centered square baselines,
- left/center/right span limits at fixed `y`,
- sample-specific custom presets,
- a formal naming convention for multi-ROI samples: `ROI_L`, `ROI_C`, `ROI_R`.

This means the next logical step is no longer geometric setup. The next logical step is signal extraction:

- load the raw or processed sample stack,
- apply derotation with the calibrated center,
- apply the chosen ROI preset,
- compute angle-dependent traces from `normal` and `crosspol`,
- compare those traces with Mueller-matrix-derived quantities such as LD or LB.

