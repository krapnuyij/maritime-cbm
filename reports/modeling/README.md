# M2 기준 모델 산출물

이 디렉터리에는 UCI `Condition Based Maintenance of Naval Propulsion Plants` 원본으로
실행한 M2 모델 선택과 최종 평가의 집계 결과만 저장한다. 직렬화한 모델과 행 단위 예측은
Git에서 제외되는 `artifacts/modeling/`에 저장한다.

## 재현 방법

```bash
uv sync --locked --group eda
uv run --locked --group eda python -m maritime_cbm.modeling.benchmark select
uv run --locked --group eda python -m maritime_cbm.modeling.benchmark evaluate
```

`select`는 train과 validation만 사용해 `artifacts/modeling/selection.json`을 생성한다.
`evaluate`는 이 파일의 데이터·분할·라이브러리 버전이 현재 환경과 같은지 확인한 뒤 네
시나리오의 test를 한 번 평가한다. test 결과를 확인한 뒤 같은 M2 실험에서 모델을
재조정하지 않는다.

## 모델 선택 결과

선택 지표는 기본 상태 그룹 validation의 `kMc`, `kMt` NRMSE 평균이다. holdout validation은
외삽 진단으로만 보고 선택 점수에는 포함하지 않았다.

| 순위 | 후보 | 상태 그룹 평균 NRMSE | 선택 |
|---:|---|---:|---|
| 1 | Random Forest, leaf 1, features 1.0 | 0.021853 | 예 |
| 2 | Random Forest, leaf 1, features sqrt | 0.024000 | 아니오 |
| 3 | 속도 중심화 Ridge, alpha 0.01 | 0.025371 | 아니오 |
| 12 | 원시 입력 Ridge, alpha 0.01 | 0.108917 | 아니오 |

전체 17개 후보와 holdout validation 진단은 `candidate_validation_metrics.csv`, 정렬된
상태 그룹 선택 결과는 `model_selection_summary.csv`에 있다.

## 최종 test 결과

| 시나리오 | 대상 | MAE | RMSE | R² | NRMSE |
|---|---|---:|---:|---:|---:|
| 행 랜덤 | `kMc` | 0.000495 | 0.001029 | 0.994993 | 0.020583 |
| 행 랜덤 | `kMt` | 0.000366 | 0.000886 | 0.985928 | 0.035454 |
| 상태 그룹 | `kMc` | 0.000418 | 0.000810 | 0.996720 | 0.016198 |
| 상태 그룹 | `kMt` | 0.000276 | 0.000592 | 0.992838 | 0.023684 |
| 압축기 holdout | `kMc` | 0.009668 | 0.010489 | -54.009429 | 0.209780 |
| 압축기 holdout | `kMt` | 0.002402 | 0.003970 | 0.719813 | 0.158798 |
| 터빈 holdout | `kMc` | 0.004519 | 0.005969 | 0.835537 | 0.119388 |
| 터빈 holdout | `kMt` | 0.008830 | 0.009167 | -41.012679 | 0.366661 |

상태 그룹 test에서는 두 대상을 정확하게 추정했지만, 심한 열화 방향 holdout에서는 대상
계수를 학습 범위보다 건강한 방향으로 예측하는 포화가 나타났다. 압축기 holdout `kMc`의
bias는 `+0.009668`, 터빈 holdout `kMt`의 bias는 `+0.008830`이다. 계수가 높을수록 정상에
가까우므로 실제보다 건강하게 추정하는 방향이다. 좁은 holdout test 범위의 R²는 보조
지표이며, MAE와 RMSE를 주지표로 해석한다.

상태 그룹 test에서도 3 knots의 MAE는 `kMc` 0.001589, `kMt` 0.001092로 다른 속도보다
높았다. 속도별 상세 수치는 `test_error_by_speed.csv`에 있다.

## 파일

- `candidate_validation_metrics.csv`: 17개 후보의 상태 그룹·holdout validation 지표
- `model_selection_summary.csv`: 상태 그룹 NRMSE 기준 후보 순위
- `final_metrics.csv`: 고정된 최종 모델의 validation·test 지표
- `test_error_by_speed.csv`: test 속도별 오차
- `test_error_by_target_state.csv`: test 실제 열화 계수별 오차
- `state_group_predictions.png`: 상태 그룹 test 실제값과 예측값
- `holdout_extrapolation.png`: 압축기·터빈 holdout 포화
- `state_group_error_by_speed.png`: 상태 그룹 test 속도별 MAE

수치의 해석 범위와 모델 한계는 [모델 카드](../../docs/MODEL_CARD.md)를 따른다.

## 출처와 라이선스

집계 결과의 원본은 UCI `Condition Based Maintenance of Naval Propulsion Plants`이며 데이터는
CC BY 4.0을 따른다. 원본 파일과 행 단위 예측은 이 저장소에 포함하지 않는다. 상세한 출처,
인용과 배포 문서의 라이선스 충돌은 [데이터 카드](../../docs/DATASET.md)에 기록했다.
