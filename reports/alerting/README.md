# M4 회귀 예측값 기반 경보 정책 산출물

이 디렉터리에는 M2 Random Forest와 M3 선형 잔차 MLP의 고정된 회귀 예측을 이용한
경보 정책 평가의 집계 결과만 저장한다. 모델을 다시 학습하지 않으며, evaluation manifest와
저장 예측의 SHA-256을 확인한 뒤 분석한다. 행 단위 정책 예측과 M4 manifest는 Git에서
제외되는 `artifacts/alerting/`에 저장한다.

## 재현 방법

```bash
uv sync --locked --group eda --group modeling
uv run --locked --group eda --group modeling python -m maritime_cbm.alerting.benchmark
```

열화도는 `kMc`에 `(1-kMc)/0.050`, `kMt`에 `(1-kMt)/0.025`를 사용하며 clipping하지
않는다. 전체 열화도는 두 값의 최댓값이다. 주 임계값 0.8은 `kMc ≤ 0.960` 또는
`kMt ≤ 0.980`에 해당하는 사전 고정 PoC 정책이며 공식 고장 기준이 아니다.

## 주 임계값 결과

| 시나리오·채널 | M2 Recall | M2 FPR | M3 Recall | M3 FPR |
|---|---:|---:|---:|---:|
| 상태 그룹 · `any` | 0.924501 | 0.000918 | 0.955840 | 0.000000 |
| 압축기 holdout · `kMc` | 0.000000 | NA | 1.000000 | NA |
| 압축기 holdout · `any` | 0.118803 | NA | 1.000000 | NA |
| 터빈 holdout · `kMt` | 0.000000 | NA | 0.931590 | NA |
| 터빈 holdout · `any` | 0.115033 | NA | 0.953377 | NA |

M2 Random Forest는 학습 경계보다 낮은 계수를 예측하지 못해 두 holdout 대상 경보를 모두
놓쳤다. M3 선형 잔차 MLP는 누락을 크게 줄였지만 터빈 holdout `kMt`에서 157/2,295건을
놓쳤다. 두 holdout의 대상 reference는 전부 양성이므로 정상 표본이 필요한 Precision,
F1, FPR과 판별력 지표인 PR-AUC는 `NA`다.

상태 그룹 `any`에서는 임계값 0.5·0.6·0.7·0.8·0.9 전 구간에서 M3 FPR이 0이었고,
Recall은 각각 0.991285, 0.962963, 0.986667, 0.955840, 0.942761이었다. 임계값마다
reference 양성 정의도 함께 바뀌므로 이 수치를 동일 표본 집합의 단순한 단조 곡선으로
해석하지 않는다.

## validation 고정 오경보율

상태 그룹 validation에서 목표 FPR 이하인 cutoff를 고정하고 모든 test에 그대로 적용했다.

| 모델 | 목표 FPR | validation `any` cutoff | 상태 그룹 test Recall | 상태 그룹 test FPR |
|---|---:|---:|---:|---:|
| M2 Random Forest | 0.01 | 0.785200 | 0.981481 | 0.004591 |
| M2 Random Forest | 0.05 | 0.758533 | 0.992877 | 0.058770 |
| M3 선형 잔차 MLP | 0.01 | 0.780506 | 0.998575 | 0.007346 |
| M3 선형 잔차 MLP | 0.05 | 0.779096 | 0.998575 | 0.017447 |

목표 FPR은 validation에서 cutoff를 선택하는 제약이지 test FPR 보장이 아니다. 실제로 M2의
목표 5% cutoff는 상태 그룹 test에서 5.88% FPR을 보였다. 정상 표본이 없는 두 holdout의
realized FPR은 계속 `NA`다.

## 파일

- `threshold_sensitivity.csv`: 임계값 0.5~0.9의 validation·test 채널별 지표
- `primary_alert_metrics.csv`: 주 임계값 0.8의 test 지표
- `primary_model_comparison.csv`: M2·M3 주 정책 지표 비교
- `fixed_fpr_cutoffs.csv`: 상태 그룹 validation에서 고정한 1%·5% FPR cutoff
- `fixed_fpr_test_metrics.csv`: 고정 cutoff의 네 test 시나리오 결과
- `state_group_threshold_sensitivity.png`: 상태 그룹 test Recall·FPR 민감도
- `severe_holdout_recall.png`: 심한 열화 방향 holdout Recall 비교

수치의 해석 범위와 모델 한계는 [모델 카드](../../docs/MODEL_CARD.md)를 따른다.

## 출처와 라이선스

집계 결과의 원본은 UCI `Condition Based Maintenance of Naval Propulsion Plants`이며 데이터는
CC BY 4.0을 따른다. 실제 고장 라벨과 공식 경보 임계값은 없으며 원본 파일과 행 단위 예측은
저장소에 포함하지 않는다.
