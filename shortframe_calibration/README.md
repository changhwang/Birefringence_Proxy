# Short-Frame QC Package

이 폴더는 `QLB_s9`-`QLB_s13` short-frame 샘플을 대상으로, `200 x 200` refined ROI (`ROI_UC`, `ROI_CC`, `ROI_LC`) 기준의 QC를 따로 확인하기 위한 전용 패키지다.

## 구성

- `shortframe_manifest.json`
  - short sample, calibration, rotation geometry 경로
- `shortframe_roi_200.json`
  - short sample 전용 `200 sq` ROI 좌표
- `run_shortframe_qc.py`
  - Phase A / Phase B QC 생성 스크립트
- `outputs/`
  - 실행 결과물

## Phase A

Calibration QC:

- `theta vs dark_PPL`, `theta vs dark_XPL`
- `theta vs empty_PPL`, `theta vs empty_XPL`
- `theta vs blank_PPL`, `theta vs blank_XPL`
- `theta vs G1`, `theta vs G2` (blank 기준)
- registration stability (blank corrected `G`)
- saturation fraction (sample raw)
- near-black fraction (sample corrected)

## Phase B

Sanity QC:

- `theta vs sample_PPL_G`
- `theta vs sample_XPL_G`
- `theta vs Afilm_PPL_G`
- `theta vs Xfilm_G`
- `theta vs Xnorm_sample_G`
- `theta vs Xnorm_blank_G`

Signal panel에는 아래를 같이 표시한다.

- raw curve
- simultaneous `2theta + 4theta` fit
- residual summary

## Phase A.5

Derotation visual QC:

- `PPL corrected G` 기준 derotated patch stack
- `0, 45, 90, 135, 165 deg` patch
- all-angle mean image
- all-angle std image
- `0 deg` 대비 mean-absolute-difference image

## 실행

```powershell
.\.venv\Scripts\python.exe .\shortframe_calibration\run_shortframe_qc.py
```

## 출력

- `outputs/<sample_id>/phaseA_calibration_qc.png`
- `outputs/<sample_id>/phaseA_calibration_curves.csv`
- `outputs/<sample_id>/phaseA_calibration_metrics.json`
- `outputs/<sample_id>/phaseA5_derotation_visual_qc.png`
- `outputs/<sample_id>/phaseA5_derotation_visual_metrics.csv`
- `outputs/<sample_id>/phaseA5_derotation_visual_metrics.json`
- `outputs/<sample_id>/phaseB_sanity_qc.png`
- `outputs/<sample_id>/phaseB_signal_curves.csv`
- `outputs/<sample_id>/phaseB_signal_metrics.json`
- `outputs/shortframe_qc_summary.csv`
- `outputs/index.json`

## 메모

- 분석 입력은 raw Bayer TIFF만 사용한다.
- preview RGB, gamma, white balance, uint8 경로는 사용하지 않는다.
- registration QC는 calibration 관점으로 `blank corrected G`를 기준으로 계산한다.
