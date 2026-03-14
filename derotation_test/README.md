# Derotation verification (`solve_center.py`)

Landmark 기반 rotation center 풀이 + phase correlation per-angle shift 보정을 수행하고, 결과를 시각적으로 검증합니다.

## 방법론

1. **Phase 1 -- Center solve**: 여러 각도에서 수동으로 찍은 landmark 좌표를 `scipy.optimize.least_squares`로 fitting하여 rotation center를 산출합니다.
2. **Phase 1.5 -- Auto-refine**: NCC가 낮은 landmark에 대해 ±20 px grid search로 위치를 자동 보정한 뒤 center를 재계산합니다.
3. **Phase 2 -- Per-angle shift**: 34개 전 각도(0~165, 5도 간격)를 derotation한 뒤, 0도 기준 ROI와 phase correlation으로 잔차 (dx, dy) shift를 측정합니다. NCC가 개선될 때만 shift를 적용합니다.

## 실행

```powershell
.\venv\Scripts\python.exe derotation_test\solve_center.py
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--rgb-dir PATH` | `data/Calibration/.../normal` | RGB TIFF 디렉터리 |
| `--output-dir PATH` | `derotation_test` | 출력 디렉터리 |
| `--crop-size N` | `1000` | 비교 크롭 크기 (px) |
| `--landmarks "0:x,y;45:x,y;..."` | 스크립트 내장 값 | 수동 landmark (full-res 좌표) |

## 출력 파일

| 파일 | 설명 |
|------|------|
| `solved_correction.json` | 최종 결과: center, landmark, 34각도 shift 테이블 |
| `overlay_all_angles.png` | 34각도 전체 derotation 후 mean overlay (선명=good, 흐림=bad) |
| `ncc_per_angle.png` | 각도별 NCC 그래프 (shift 전/후, threshold 표시) |
| `compare_original_vs_derotated.png` | 0/45/90/135도 Original vs Derotated 2행 비교 |

## 현재 결과 요약

- Solved center: `(1099.2, 543.3)`
- NCC: 전 34각도 0.96 이상, 대부분 0.98 이상
- Suspect 각도: 없음 (threshold 0.80)

결과 JSON은 `analysis_outputs/rotation_calibration_new/solved_correction.json`에 복사되어 `configs/datasets/analysis_manifest.json`의 `rotation_correction` 키로 참조됩니다. 자세한 방법론 설명은 `docs/rotation_calibration.md` Section 14를 참조하세요.
