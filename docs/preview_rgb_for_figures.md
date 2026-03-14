# Preview RGB For Figures

This document describes the current RGB preview conversion used for human inspection, ROI review, and figure preparation.

This is not an analysis document.

The reason it is separated from the measurement and raw-format document is simple:

- the preview images are demosaiced,
- white-balanced,
- gamma-corrected,
- converted to `uint8` RGB,
- and therefore should not be treated as quantitative intensity data.

If the goal is numerical analysis, start from the raw TIFF files described in:

- [measurement_and_raw_data_format.md](../docs/measurement_and_raw_data_format.md)

## 1. Purpose of the preview RGB images

The preview RGB images exist to support:

- fast visual checking of acquisition results,
- easier human interpretation of the sample appearance,
- ROI placement and review,
- figure preparation for reports or manuscripts.

They do not exist to replace the raw Bayer measurement.

## 2. Input data

The preview pipeline starts from:

- `XI_RAW16` TIFF files
- `uint16` container
- `10-bit` sensor values
- Bayer `GBRG` pattern

Representative folders:

- [data/QLB/RAW](../data/QLB/RAW)
- [data/QLB/RGB](../data/QLB/RGB)
- [data/ROI_selection](../data/ROI_selection)

## 3. Current preview conversion pipeline

The current preview conversion workflow is:

1. read `RAW16` TIFF
2. normalize using the raw sensor range `0..1023`
3. demosaic the Bayer `GBRG` pattern
4. apply fixed white balance
5. apply gamma correction
6. save as `uint8` RGB

The fixed white-balance values currently used are:

- `R = 1.4`
- `G = 1.0`
- `B = 1.2`

The current gamma is:

- `1 / 2.2`

In the present preview path, the `GBRG` raw data were interpreted through OpenCV's `GB2BGR` demosaic route because that produced the visually correct color appearance for the current dataset.

## 4. Why this should not be used for analysis

The preview RGB images are not appropriate as quantitative input because they include nonlinear and display-oriented steps:

- demosaic interpolation,
- fixed white balance,
- gamma correction,
- `uint8` output conversion.

The gamma step is especially important here. Once the data are gamma-corrected for visual appearance, the resulting pixel values are no longer suitable as direct intensity measurements.

So for any step such as:

- transmittance-like normalization,
- crosspol vs normal comparison,
- angle-series fitting,
- harmonic decomposition,
- comparison against Mueller-matrix-derived quantities,

the correct source remains the raw TIFF stack, not the preview RGB stack.

## 5. Where the preview images are useful

The preview RGB images are still useful for:

- checking whether the sample is centered,
- checking whether the field of view is acceptable,
- identifying film edges and obvious defects,
- designing ROIs,
- creating overlays and contact sheets,
- building presentation or manuscript figures.

That is exactly how they were used in the current workflow:

- rotation calibration was computed from raw data,
- ROI review was done on preview RGB images.

## 6. Practical summary

The RGB files in this project are figure/review images, not scientific source data. They are derived from raw Bayer TIFFs by demosaic, fixed white balance, gamma correction, and `uint8` conversion. They are appropriate for visualization and ROI inspection, but not for quantitative birefringence analysis.

