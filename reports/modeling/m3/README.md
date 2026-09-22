# M3 PyTorch 비교 모델 산출물

이 디렉터리에는 M2 Random Forest와 같은 데이터 분할·평가 지표로 실행한 M3 신경망
비교 실험의 집계 결과만 저장한다. checkpoint, manifest와 행 단위 예측은 Git에서 제외되는
`artifacts/modeling/m3/`에 저장한다.

## 재현 방법

```bash
uv sync --locked --group eda --group modeling
uv run --locked --group eda --group modeling python -m maritime_cbm.modeling.torch_benchmark select --device cpu
uv run --locked --group eda --group modeling python -m maritime_cbm.modeling.torch_benchmark evaluate --device cpu
```

`select`는 test를 읽지 않고 6개 후보를 세 seed로 반복 평가한다. 최종 선택에는 상태 그룹
validation의 대상별 평균 NRMSE만 사용한다. `evaluate`는 선택 manifest의 데이터·분할·환경과
세 seed 42 checkpoint를 검증한 뒤 test를 한 번 평가한다. 상태 그룹과 두 holdout은 selection
checkpoint를 재사용하며 행 랜덤 모델만 평가 단계에서 새로 학습한다.

## 선택 결과

선택 모델은 속도 중심화와 `(128, 64)` 은닉층을 사용하는 선형 잔차 MLP다.

| 순위 | 후보 | 세 seed 평균 NRMSE | 최악 대상 NRMSE |
|---:|---|---:|---:|
| 1 | 선형 잔차 MLP, speed-centered, 128·64 | 0.003762 | 0.004659 |
| 2 | MLP, speed-centered, 128·64 | 0.003969 | 0.004987 |
| 3 | 선형 잔차 MLP, speed-centered, 64·32 | 0.004152 | 0.004974 |
| 4 | MLP, speed-centered, 64·32 | 0.004260 | 0.005137 |

전체 후보·시나리오·seed 결과는 `candidate_validation_metrics.csv`, 정렬된 선택 결과는
`selection_summary.csv`에 있다.

## 최종 test 결과

| 시나리오 | 대상 | MAE | RMSE | R² | NRMSE |
|---|---|---:|---:|---:|---:|
| 행 랜덤 | `kMc` | 0.000107 | 0.000158 | 0.999882 | 0.003153 |
| 행 랜덤 | `kMt` | 0.000081 | 0.000117 | 0.999757 | 0.004661 |
| 상태 그룹 | `kMc` | 0.000100 | 0.000140 | 0.999903 | 0.002793 |
| 상태 그룹 | `kMt` | 0.000085 | 0.000126 | 0.999678 | 0.005022 |
| 압축기 holdout | `kMc` | 0.000811 | 0.001029 | 0.470539 | 0.020581 |
| 압축기 holdout | `kMt` | 0.000441 | 0.000551 | 0.994602 | 0.022042 |
| 터빈 holdout | `kMc` | 0.001429 | 0.001763 | 0.985648 | 0.035268 |
| 터빈 holdout | `kMt` | 0.001237 | 0.001385 | 0.040868 | 0.055401 |

M2 대비 상태 그룹 NRMSE는 `kMc` 82.8%, `kMt` 78.8% 감소했다. 강건성 평가 대상인
압축기 `kMc`와 터빈 `kMt` NRMSE도 각각 90.2%, 84.9% 감소했다. 그러나 두 holdout의
bias는 각각 `+0.000695`, `+0.000996`으로 여전히 실제보다 건강하게 추정하는 방향이며,
좁은 test 범위의 R²는 보조 지표로만 해석한다.

M2 holdout 결과를 확인한 뒤 M3 구조를 설계했으므로 이 비교는 완전히 미관측인 독립 test가
아니라 사전에 고정한 벤치마크의 탐색적 비교다.

## 파일

- `candidate_validation_metrics.csv`: 6개 후보 × 3개 시나리오 × 3개 seed validation 지표
- `selection_summary.csv`: 세 seed 상태 그룹 NRMSE 기준 후보 순위
- `selected_training_history.csv`: 선택 후보 seed 42의 세 시나리오 학습 곡선
- `final_metrics.csv`: 고정 모델의 validation·test 지표
- `baseline_comparison.csv`: M2 Random Forest와 M3 test 지표 비교
- `holdout_diagnostics.csv`: holdout 예측 범위·기울기·bias
- `test_error_by_speed.csv`, `test_error_by_target_state.csv`: test 오류 집계
- PNG 3개: 학습 곡선, M2 비교, holdout 실제값·예측값

수치의 해석 범위와 모델 한계는 [모델 카드](../../../docs/MODEL_CARD.md)를 따른다.

## 출처와 라이선스

집계 결과의 원본은 UCI `Condition Based Maintenance of Naval Propulsion Plants`이며 데이터는
CC BY 4.0을 따른다. 원본 파일, checkpoint와 행 단위 예측은 저장소에 포함하지 않는다.
