# Measurement And Raw Data Format

This document describes how the birefringence image data were acquired and how the saved image files should be interpreted.

This is the first document that should be read before the later analysis documents:

- [rotation_calibration.md](../docs/rotation_calibration.md)
- [roi_selection_and_custom_presets.md](../docs/roi_selection_and_custom_presets.md)
- [preview_rgb_for_figures.md](../docs/preview_rgb_for_figures.md)

The purpose of this document is to make the measurement setup and the saved file semantics explicit, so that later analysis steps do not accidentally mix:

- raw sensor data,
- preview RGB images,
- polarizer configuration,
- sample rotation metadata.

## 1. Optical measurement setup

The current imaging setup uses:

- a backlight,
- a polarizer in front of the backlight,
- the printed organic conjugated thin-film sample,
- a second polarizer in front of the imaging lens,
- a XIMEA color camera.

The two polarizer states used in the dataset are:

- `normal`
  - the two polarizers are parallel
  - in the current metadata this is stored with `polarizer_angle_deg = 0`
- `crosspol`
  - the two polarizers are crossed by `90 deg`
  - in the current metadata this is stored with `polarizer_angle_deg = 90`

In practical terms:

- `normal` is the parallel-polarizer imaging condition
- `crosspol` is the crossed-polarizer imaging condition

## 2. Camera and sensor settings

The camera recorded in the current datasets is:

- camera model: `XIMEA MQ022CG-CM`
- backend: `ximea`

Representative metadata file:

- [QLB_s1_metadata.json](../data/QLB/RAW/QLB_s1/QLB_s1_metadata.json)

Metadata fields consistently seen in the saved datasets:

- `camera_model = MQ022CG-CM`
- `imgdataformat = XI_RAW16`
- `sensor_bit_depth = XI_BPP_10`
- `image_data_bit_depth = XI_BPP_10`
- `output_bit_depth = XI_BPP_10`
- `cfa_pattern = XI_CFA_BAYER_GBRG`
- `resolution = [2048, 1088]`
- `roi = {x: 0, y: 0, w: 2048, h: 1088}`
- `gain_db = 0.0`

The metadata also records the white-balance and gamma values associated with the camera configuration:

- `wb_r = 1.4`
- `wb_g = 1.0`
- `wb_b = 1.2`
- `gammaY = 1.0`
- `gammaC = 1.0`

Important:

The raw analysis data are still raw Bayer data. These metadata values are useful for preview generation and documentation, but the raw TIFFs themselves are not already white-balanced RGB images.

## 3. Acquisition protocol per sample

For each sample, the basic acquisition sequence is:

1. image the sample in `crosspol`
2. rotate the sample from `0 deg` to `165 deg` in `5 deg` increments
3. image the sample in `normal`
4. rotate the sample from `0 deg` to `165 deg` in `5 deg` increments

This gives:

- `34` images for `crosspol`
- `34` images for `normal`
- `68` images total per full sample sequence

Example metadata:

- [QLB_s1_metadata.json](../data/QLB/RAW/QLB_s1/QLB_s1_metadata.json)
- [QLB_s1_log.csv](../data/QLB/RAW/QLB_s1/QLB_s1_log.csv)

Example per-mode exposure values observed in the metadata:

- `crosspol`: `50000 us`
- `normal`: `18000 us`

The metadata stores, per frame:

- filename
- mode
- polarizer angle
- sample angle
- exposure
- timestamp

So each frame is not just an image; it is an image tied to a physical sample rotation and a polarizer configuration.

## 4. Calibration datasets acquired alongside the samples

In addition to the `QLB_s1` to `QLB_s13` sample sequences, the project also contains dedicated calibration datasets:

- `dark`
- `empty`
- `empty_shortframe`
- `empty_longframe`
- `blank_shortframe`
- `blank_longframe`
- `rotation_calibration`

These are stored under:

- [data/Calibration](../data/Calibration)

Their roles are different:

- `dark`: camera/dark current baseline
- `empty` / `empty_*`: optical-path reference without the sample
- `blank_*`: substrate or blank reference
- `rotation_calibration`: dedicated geometric reference used to estimate the rotation center and all-angle valid region

## 5. What `RAW16` means in this project

This point is critical.

The saved raw TIFF files are **not** 16-bit RGB images.

They should be interpreted as:

- `uint16`
- 2D grayscale array
- Bayer raw
- `10-bit` sensor data stored inside a `16-bit` container

So `XI_RAW16` here means:

> raw Bayer sensor values written to a 16-bit integer container

It does **not** mean:

> every pixel already contains full 16-bit RGB color

## 6. Bayer pattern used by the camera

The saved CFA pattern is:

- `XI_CFA_BAYER_GBRG`

That means the pixel pattern is:

```text
G B G B ...
R G R G ...
G B G B ...
R G R G ...
...
```

Each sensor pixel directly measures only one color component:

- either `R`
- or `G`
- or `B`

So a single raw pixel does not contain full RGB information by itself.

This is why raw Bayer images can look checkerboard-like or strange if opened as if they were already normal grayscale or RGB images.

## 7. Meaning of the saved raw values

Each raw TIFF pixel stores one measured sensor value for one color filter location.

Because the sensor bit depth is `10-bit`, the physically meaningful range is approximately:

- `0` to `1023`

The project stores this in a `uint16` TIFF container, so the file type is 16-bit but the sensor information content is 10-bit.

One-line summary:

- `10-bit sensor data in a uint16 container`

## 8. Raw TIFF vs preview RGB TIFF

The project currently contains two conceptually different image products:

### 8.1 Analysis-oriented raw data

Stored under paths such as:

- [data/QLB/RAW](../data/QLB/RAW)

Characteristics:

- Bayer raw
- single-channel sensor data
- no demosaic in the saved pixel array
- no gamma correction in the analysis image
- no final RGB color representation

These are the files that should be treated as the scientific source for quantitative analysis.

### 8.2 Human-readable preview RGB data

Stored under paths such as:

- [data/QLB/RGB](../data/QLB/RGB)
- [data/ROI_selection](../data/ROI_selection)

Characteristics:

- demosaiced
- white-balanced
- gamma-corrected
- converted to 8-bit RGB for visualization

These are for visual inspection and ROI design, not for final quantitative extraction.

The detailed RGB conversion pipeline is documented separately in:

- [preview_rgb_for_figures.md](../docs/preview_rgb_for_figures.md)

That separation is intentional, because the preview images are gamma-corrected and therefore should not be confused with analysis-ready intensity data.

## 9. Analysis-use vs visualization-use distinction

This distinction should remain strict throughout the project.

### 9.1 For analysis

Use:

- raw TIFF
- Bayer structure preserved unless a specific analysis step intentionally transforms it
- no gamma correction
- no preview white balance
- no casual conversion to RGB just because it looks nicer

Typical examples:

- intensity extraction
- background correction
- dark subtraction
- angle-series analysis
- later comparison to Mueller-matrix-derived quantities

### 9.2 For visualization

Use:

- demosaiced RGB
- white balance
- gamma correction
- contact sheets
- ROI review overlays

Typical examples:

- `ROI_selection`
- sample overview figures
- review contact sheets

The exact figure/preview conversion details are intentionally kept out of this document and described separately in [preview_rgb_for_figures.md](../docs/preview_rgb_for_figures.md).

## 10. Relationship to later analysis steps

This raw/preview distinction is already reflected in the later project workflow:

- rotation calibration geometry was estimated from the raw normal-mode images, not from final preview RGB screenshots
- ROI placement was reviewed on RGB preview images because humans need interpretable images
- the intended future scientific extraction step should go back to the raw TIFF stacks

So the project workflow is intentionally split:

1. acquire raw measurement data
2. generate preview RGB for human inspection
3. use preview RGB to define ROIs
4. apply those ROIs back to derotated raw data for actual analysis

## 11. Example directory structure

Representative sample folders:

- raw sample stack:
  - [data/QLB/RAW/QLB_s1](../data/QLB/RAW/QLB_s1)
- preview RGB sample stack:
  - [data/QLB/RGB/QLB_s1_rgb](../data/QLB/RGB/QLB_s1_rgb)
- ROI review copies:
  - [data/ROI_selection](../data/ROI_selection)

Representative calibration folders:

- [data/Calibration/dark](../data/Calibration/dark)
- [data/Calibration/empty](../data/Calibration/empty)
- [data/Calibration/blank_longframe](../data/Calibration/blank_longframe)
- [data/Calibration/rotation_calibration](../data/Calibration/rotation_calibration)

## 12. Practical one-paragraph summary

The current system saves `GBRG` Bayer raw from a `XIMEA MQ022CG-CM` camera as `XI_RAW16`, meaning `10-bit` sensor data are stored inside a `uint16` grayscale TIFF container. The raw files are not RGB images. Separate RGB preview files may be generated later for inspection and figure preparation, but those are not analysis-safe intensity data. Quantitative analysis should always start from the raw TIFF data, while RGB previews are intended for inspection and ROI selection only.

## 13. Short bullet summary

- camera: `XIMEA MQ022CG-CM`
- raw save mode: `XI_RAW16`
- raw layout: `uint16`, 2D, Bayer raw
- CFA pattern: `XI_CFA_BAYER_GBRG`
- sensor data depth: `10-bit`
- physical raw value meaning: one color-filter sample per pixel
- `normal`: parallel polarizers
- `crosspol`: crossed polarizers (`90 deg`)
- angle sweep: `0-165 deg` in `5 deg` steps
- preview RGB exists for figure/inspection use only
- analysis should use raw, not preview RGB

