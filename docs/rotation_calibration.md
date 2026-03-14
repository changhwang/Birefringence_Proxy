# Rotation Calibration Workflow

This document records, in detail, how the rotation calibration step was performed for the birefringence image analysis workflow.

Before reading this document, the acquisition and file-format conventions should be read first:

- [measurement_and_raw_data_format.md](../docs/measurement_and_raw_data_format.md)

The goal of this step is to answer these questions before any sample ROI work begins:

1. What is the effective rotation center of the image stack?
2. After derotation, which pixels survive across the full `0-165 deg` sequence without leaving the frame?
3. What is the largest conservative rectangular area that is guaranteed to survive all rotations?

This geometry is then reused for downstream ROI work on the sample images.

## 1. Why this step exists

The sample is intentionally rotated during acquisition, but for pixel-wise or ROI-wise comparison across angle we do not want the sample to drift inside the image plane.

That means we need a geometric reference that tells us:

- where the true rotation center is in image coordinates,
- how much of the field of view remains valid after inverse rotation,
- what part of the image can be used safely for every angle.

Without this step, later angle-series plots would mix physical sample changes with simple geometric motion.

## 2. Data used in this step

Only the dedicated `rotation_calibration` dataset was used for the geometry estimation.

Input folder:

- [data/Calibration/rotation_calibration/normal](../data/Calibration/rotation_calibration/normal)

Metadata source:

- [rotation_calibration_metadata.json](../data/Calibration/rotation_calibration/rotation_calibration_metadata.json)
- [rotation_calibration_log.csv](../data/Calibration/rotation_calibration/rotation_calibration_log.csv)

Key metadata values from this dataset:

- `sample_id`: `rotation_calibration`
- `mode`: `normal` only
- `angles`: `0, 5, 10, ..., 165 deg` for a total of `34` images
- `exposure_us`: `18000`
- `resolution`: `2048 x 1088`
- `imgdataformat`: `XI_RAW16`
- `sensor / image bit depth`: `10-bit` data stored in a `16-bit` TIFF container
- `cfa_pattern`: `XI_CFA_BAYER_GBRG`

Important note:

The raw TIFF files are single-channel sensor data but the camera is a Bayer color camera. For geometry estimation we do not want the Bayer mosaic pattern itself to dominate the registration signal, so the search step intentionally suppresses that pattern first.

## 3. Files created for this step

Script used:

- [derive_derotation_geometry.py](../scripts/derive_derotation_geometry.py)

Primary output folder:

- [analysis_outputs/rotation_calibration](../analysis_outputs/rotation_calibration)

Generated outputs:

- [derotation_geometry.json](../analysis_outputs/rotation_calibration/derotation_geometry.json)
- [rotation_calibration_geometry_overlay.png](../analysis_outputs/rotation_calibration/rotation_calibration_geometry_overlay.png)
- [rotation_calibration_valid_mask.png](../analysis_outputs/rotation_calibration/rotation_calibration_valid_mask.png)
- [rotation_calibration_derotated_preview.png](../analysis_outputs/rotation_calibration/rotation_calibration_derotated_preview.png)

## 4. Overall procedure

The rotation calibration was done in four stages:

1. Load the normal-mode rotation calibration TIFF sequence.
2. Estimate the rotation center with a coarse-to-fine search on feature images.
3. Using the chosen center, compute the full valid pixel mask across all 34 angles.
4. Compute the largest axis-aligned rectangle fully contained in that valid mask.

Each stage is described below.

## 5. Stage 1: Load and preprocess the images

### 5.1 Raw image loading

Each TIFF was loaded as `float32` using `Pillow`.

The source data are:

- `2048 x 1088`
- `Gray16` TIFF representation
- `10-bit` raw sensor values stored in a `16-bit` container

### 5.2 Why preprocessing was necessary

Because the camera is Bayer (`GBRG`), raw neighboring pixels do not correspond to the same spectral channel. If we tried to estimate geometry directly from the untouched mosaic, the periodic Bayer pattern could interfere with the rotation-center search.

### 5.3 Preprocessing steps used for center search

For center estimation only, the script does the following:

1. Apply `2x2` binning.
2. Normalize intensities by the maximum value.
3. Apply a Gaussian filter (`sigma = 1.0`).
4. Compute Sobel gradients in `x` and `y`.
5. Convert those gradients into a gradient-magnitude feature image.
6. Downsample again by a factor of `4` for faster search.
7. Standardize the feature image to zero mean and unit variance.

This produces a stable structural image for registration-like comparison while keeping the search inexpensive.

## 6. Stage 2: Estimate the rotation center

### 6.1 Angles used for the search

The center search does not use all 34 angles initially. Instead, it uses this subset:

- `0`
- `30`
- `60`
- `90`
- `120`
- `150`

This is a good compromise between angular coverage and runtime.

### 6.2 Search space and coarse-to-fine strategy

The center search starts from the geometric image center of the downsampled search image and refines the estimate in several passes.

Search settings used:

- `bayer_bin = 2`
- `search_downsample = 4`
- effective full-resolution scale per search pixel = `8`

Search passes:

1. step `8.0`, radius `32.0`
2. step `4.0`, radius `16.0`
3. step `2.0`, radius `8.0`
4. step `1.0`, radius `4.0`
5. step `0.5`, radius `2.0`

At each pass, candidate centers are scored and the best candidate becomes the center of the next pass.

### 6.3 How each candidate center is scored

For each candidate center:

1. Each feature image is inverse-rotated by its acquisition angle.
2. A validity mask is built by rotating an all-ones mask with the same transform.
3. These masks are intersected across the angle subset.
4. The intersected mask is eroded by `2 px` to avoid interpolation-edge artifacts.
5. The derotated feature stacks are standardized frame-wise.
6. The mean per-pixel variance across the stack is computed on the surviving region.

The score used in the script is:

```text
score = -mean_variance + 0.5 * area_fraction
```

This favors centers that both:

- align the structure well after derotation,
- and preserve a reasonably large valid area.

### 6.4 Final center result

The final chosen center was:

- `x = 1204.0`
- `y = 528.0`

This is not identical to the simple image midpoint (`1024, 544`), which is exactly why the calibration step is necessary.

Search history from the output JSON:

1. best search-space center after coarse pass: `(152.0, 68.0)`
2. next pass: `(152.0, 72.0)`
3. next pass: `(150.0, 66.0)`
4. unchanged at `1 px` step
5. final refined search-space center: `(150.5, 66.0)`

Converted back to full resolution:

- `x = 150.5 * 8 = 1204.0`
- `y = 66.0 * 8 = 528.0`

## 7. Stage 3: Compute the common valid mask

Once the center is fixed, the script computes the mask of pixels that remain valid across the **entire** `0-165 deg` sequence.

### 7.1 Method

1. Start with an all-ones mask the size of the original image.
2. Inverse-rotate that mask by every angle from `0` to `165 deg` in `5 deg` steps.
3. Intersect all masks.
4. Apply an additional erosion of `4 px` to remove edge ambiguity.

This produces a conservative mask of pixels that survive the full sequence.

### 7.2 Result

From [derotation_geometry.json](../analysis_outputs/rotation_calibration/derotation_geometry.json):

- valid area fraction: `0.41076211368336396`
- valid area in pixels: `915270`

So approximately `41.1%` of the original frame is guaranteed to survive derotation across all 34 angles.

The bounding box of the full valid mask is:

- `x_min = 648`
- `x_max = 1727`
- `y_min = 4`
- `y_max = 1081`
- `width = 1080`
- `height = 1078`

Important interpretation:

This bounding box is **not** itself guaranteed to be valid everywhere inside. It only describes the outer extent of the valid-mask shape. The actual valid region is more circular / lens-shaped after all rotations are intersected.

## 8. Stage 4: Compute the largest conservative rectangle

The next question is not just "what pixels survive," but:

> what is the largest axis-aligned rectangular ROI we can use safely without touching invalid pixels?

To answer that, the script finds the largest all-valid rectangle inside the binary valid mask using a histogram-based maximal-rectangle search.

### 8.1 Result

The largest conservative axis-aligned rectangle was:

- `x = 799`
- `y = 145`
- `width = 765`
- `height = 766`
- `area = 585990`

This rectangle became the first practical "safe rectangle" used for downstream ROI exploration.

### 8.2 Why this rectangle matters

This rectangle is a conservative, easy-to-understand working region because:

- it is fully valid across all angles,
- it is axis-aligned,
- it is simple to overlay on sample review images,
- it provides an easy starting point for global ROI definitions.

Important caveat:

The rectangle is **not** the only usable region. Because the full valid mask is not rectangular, later custom ROIs can extend outside the green rectangle as long as they remain inside the full valid mask. This is exactly what later happened for some sample-specific ROIs.

## 9. Preview outputs and how they are used

### 9.1 Geometry overlay

- [rotation_calibration_geometry_overlay.png](../analysis_outputs/rotation_calibration/rotation_calibration_geometry_overlay.png)

This image shows:

- the reference `0 deg` rotation calibration image,
- the invalid outside region tinted,
- the full valid mask boundary,
- the largest safe rectangle,
- the chosen rotation center.

This was the main visual check used before moving on to the sample ROI stage.

### 9.2 Valid mask image

- [rotation_calibration_valid_mask.png](../analysis_outputs/rotation_calibration/rotation_calibration_valid_mask.png)

This is the binary all-angle mask itself.

### 9.3 Derotated preview

- [rotation_calibration_derotated_preview.png](../analysis_outputs/rotation_calibration/rotation_calibration_derotated_preview.png)

This preview shows selected angles after inverse rotation using the estimated center. It is intended as a sanity check of the geometric correction.

## 10. Interpretation notes

### 10.1 Why the alignment correlation is not uniformly high

The output JSON contains `corr_to_0deg` and `rmse_to_0deg` values for every angle. These numbers are useful as a rough sanity check, but they should not be overinterpreted as a pure geometric-quality metric.

Reason:

The rotation calibration sample can change apparent intensity pattern with angle, so lower correlation to `0 deg` does not necessarily mean the center is wrong. The center search itself is feature-based and area-aware, which is the more relevant criterion here.

### 10.2 Why the raw Bayer data were not used directly as-is

Using raw Bayer pixels directly for registration is risky because:

- adjacent pixels belong to different color filters,
- the mosaic itself creates periodic structure,
- that pattern can bias a geometric search.

The `2x2` binning step was therefore deliberate and important.

### 10.3 Why the safe rectangle was only a starting point

Later, when working on `ROI_selection`, the safe rectangle was intentionally treated as:

- a conservative global reference region,
- not a hard scientific requirement that all final ROIs must stay inside it.

That distinction matters. The later custom ROIs were allowed to move outside the green rectangle if they still stayed inside the full valid mask.

## 11. How this step connects to the later sample ROI work

The entire purpose of this rotation calibration step was to unlock the next steps safely:

1. Overlay the safe geometry on the `ROI_selection` sample images.
2. Explore centered `500 / 400 / 300` square ROIs.
3. Explore left / center / right span positions at fixed `y`.
4. Build sample-specific ROI presets while staying inside the all-angle valid region.

In other words:

- `rotation_calibration` established the geometry,
- `ROI_selection` used that geometry for practical ROI design.

## 12. Re-running the calibration

From the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\derive_derotation_geometry.py
```

This will regenerate:

- the geometry JSON,
- the valid mask,
- the geometry overlay,
- the derotated preview.

## 13. Final numeric summary

Rotation calibration summary:

- input dataset: `rotation_calibration`, normal mode only, `34` frames
- full-resolution image size: `2048 x 1088`
- estimated rotation center: `(1204.0, 528.0)`
- all-angle valid fraction: `41.076%`
- all-angle valid pixel count: `915270`
- valid-mask bbox: `x=648..1727`, `y=4..1081`
- largest safe rectangle: `x=799`, `y=145`, `w=765`, `h=766`

This was the geometry foundation used for the later ROI review and custom ROI preset work.

