# Maritime CBM 모델 카드

## 모델 개요

- 모델 ID: `m2-random-forest-baseline-v1`
- 상태: M2 기준 모델과 M3 비교 모델 평가 완료
- 구현체: scikit-learn 1.9.1 `RandomForestRegressor`
- 목적: 정상상태 시뮬레이션 센서값에서 `kMc`, `kMt` 열화 상태 계수를 동시에 추정
- 기준 Git commit: `129ae3c`

이 모델은 실제 선박 고장진단 모델이 아니다. 실제 고장 라벨, 타임스탬프와 공식 경보
임계값이 없는 공개 시뮬레이션 데이터로 만든 PoC 기준 모델이다.

M3 비교 모델 ID는 `m3-linear-residual-mlp-v1`이며 PyTorch 2.14.0으로 구현했다. M3 결과가
M2 배포 후보를 자동 교체하지 않으며, 최종 배포 모델 선택은 M4·M5 요구사항과 함께 별도로
결정한다.

## 데이터와 입력

- 데이터: UCI `Condition Based Maintenance of Naval Propulsion Plants`
- 행 수: 11,934
- 예측 대상: `kMc`, `kMt`
- 분할: `docs/DATASET.md`에 고정된 행 랜덤, 상태 그룹, 압축기·터빈 holdout
- 입력: 원본 16개 중 상수 `T1`, `P1`, 중복 `Tp`, `v`의 1:1 proxy인 `lp`를 제외한 12개

모델 입력 순서는 다음과 같다.

```text
v, GTT, GTn, GGn, Ts, T48, T2, P48, P2, Pexh, TIC, mf
```

전처리와 target scaler는 학습 행에서만 fit한다. Random Forest 입력에는 별도의 feature
scaling을 적용하지 않는다. `kMc`, `kMt`는 `StandardScaler`로 학습 시 표준화하고 예측 시
원래 단위로 역변환한다. 예측값 clipping은 적용하지 않는다.

## 모델 설정

```text
n_estimators=300
min_samples_leaf=1
max_features=1.0
random_state=42
n_jobs=1
```

모델은 `TransformedTargetRegressor`로 감싼 단일 다중 출력 Random Forest이다.

## 선택 절차

17개 후보를 상태 그룹·압축기 holdout·터빈 holdout train에서 각각 학습하고 validation을
평가했다. 최종 선택에는 기본 상태 그룹 validation의 `kMc`, `kMt` NRMSE 평균만 사용했다.
행 랜덤과 두 holdout validation은 비교·외삽 진단이며 선택 점수에는 포함하지 않았다.

- 선택 모델 평균 NRMSE: 0.021853
- 선택 모델 대상별 NRMSE: `kMc` 0.016707, `kMt` 0.026998
- 다음 후보: 속도 중심화 Ridge alpha 0.01, 평균 NRMSE 0.025371

선택 manifest를 고정한 뒤 이 모델만 네 시나리오 test에서 한 번 평가했다. holdout
validation은 선택 진단이고 holdout test는 고정 모델의 최종 강건성 평가다.
각 시나리오 모델은 해당 train 역할만으로 fit했으며 validation을 train에 합쳐 재학습하지
않았다.

## 최종 test 성능

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

연속 열화 구간 holdout 대상인 압축기 `kMc`와 터빈 `kMt`는 좁은 정답 범위 때문에 R²를
보조 지표로만 해석한다. 해당 평가의 주지표는 MAE와 RMSE다.

## 오류 분석

- 상태 그룹 test 성능은 두 대상 모두 R² 0.99 이상이다.
- 3 knots에서 상태 그룹 MAE가 다른 속도보다 높다.
- 압축기 holdout `kMc` bias는 `+0.009668`이다.
- 터빈 holdout `kMt` bias는 `+0.008830`이다.
- 두 holdout 대상 모두 학습에서 보지 못한 낮은 계수를 더 건강한 방향으로 추정했다.
- test 예측의 공식 계수 범위 이탈은 없었지만, 이는 tree 예측의 학습 범위 포화와 함께
  해석해야 하며 외삽 성공을 뜻하지 않는다.

## 제한사항과 금지된 해석

- 타임스탬프가 없어 시계열 예측이나 열화 진행 예측에 사용할 수 없다.
- 실제 고장 라벨이 없어 실제 고장 발생 여부나 잔여수명을 예측하지 않는다.
- 시뮬레이션과 실제 선박 사이의 domain gap을 검증하지 않았다.
- 승인된 네 분할은 모든 학습 세트에 9개 속도를 포함하므로 미관측 운항 속도를 평가하지
  않는다.
- 심한 미관측 열화 방향에서 성능이 크게 악화되므로 이 모델의 예측만으로 경보를 확정하면
  안 된다.
- M4 경보 정책에서는 holdout의 건강 방향 bias가 경보 누락으로 이어질 가능성을 별도로
  분석해야 한다.
- 직렬화한 모델이 약 253MB이므로 M5 서비스화 전에 압축, 메모리와 지연시간을 측정하고
  배포 artifact 정책을 결정해야 한다.

## M3 PyTorch 비교 모델

M3는 Random Forest의 격자 내부 성능과 심한 열화 방향 외삽 포화를 보완할 수 있는지
확인하기 위한 비교 실험이다. 원시 입력 MLP, 속도 중심화 MLP와 속도 중심화 선형 잔차
MLP의 6개 후보를 세 seed로 반복 평가했다. 상태 그룹 validation의 대상별 seed 평균
NRMSE만으로 다음 모델을 선택했다.

```text
architecture=linear_residual_mlp
preprocessing=speed_centered
hidden_sizes=(128, 64)
trainable_parameters=10,076
optimizer=AdamW(lr=1e-3, weight_decay=1e-4)
batch_size=256
```

선형 경로에 0 초기화한 비선형 잔차 경로를 더하며, ReLU 은닉층과 clipping 없는 선형
출력을 사용한다. 입력과 target scaler, 속도별 중심화 통계는 각 시나리오 train에만 fit했다.
공식 실험은 CPU·float32로 수행했고 최대 500 epoch, 최소 50 epoch, patience 40으로
early stopping한 best checkpoint를 복원했다.

선택 모델의 세 seed 상태 그룹 validation 평균 NRMSE는 `kMc` 0.002866, `kMt`
0.004659이다. 최종 test 결과는 다음과 같다.

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

M2 대비 상태 그룹 NRMSE는 `kMc` 82.8%, `kMt` 78.8% 감소했고, 강건성 평가 대상인
압축기 `kMc`와 터빈 `kMt` NRMSE도 각각 90.2%, 84.9% 감소했다. 두 holdout 대상의
bias는 `+0.000695`, `+0.000996`으로 줄었지만 여전히 실제보다 건강하게 추정하는 방향이다.
좁은 holdout 범위의 R²는 보조 지표로만 해석한다.

평균 bias만으로 개별 오차 방향을 일반화할 수는 없다. 압축기 `kMc`와 터빈 `kMt`의
예측 범위는 실제 test 범위의 각각 2.386배, 2.089배였고 회귀 기울기는 0.877, 0.813이었다.
따라서 예측은 실제 추세보다 완만하게 반응하면서 test 범위 밖으로도 퍼졌으며, 일부 값은
실제 최솟값보다 낮았다. M4에서는 평균적인 건강 방향 bias로 인한 경보 누락 가능성과 함께
임계값 주변의 산발적 오경보 가능성도 검토해야 한다. 공식 경보 임계값이 아직 없으므로
현재 결과만으로 실제 누락률이나 오경보율을 산출했다고 해석하지 않는다.

M3 상태 그룹·두 holdout은 selection 단계에서 train으로 학습하고 validation으로 early
stopping한 seed 42 checkpoint를 test에서 처음 평가했다. 행 랜덤만 최종 평가 단계에서
새로 학습했다. 이는 네 시나리오를 평가 단계에서 다시 학습한 M2와 절차가 다르지만 test
누수는 없다.

M2 holdout 결과를 확인한 뒤 M3 구조를 설계했으므로 holdout 개선은 완전히 미관측인 독립
test 성능이 아니라 사전에 고정한 벤치마크의 탐색적 비교다. 시뮬레이션과 실제 선박 간
domain gap, 미관측 속도와 실제 고장 상태는 여전히 검증하지 않았다.

M3 checkpoint는 `artifacts/modeling/m3/`에만 저장하고 Git에 포함하지 않는다. 기본 상태
그룹 checkpoint의 SHA-256은
`cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472`이다. 재현 명령과
추적 산출물은 [`reports/modeling/m3/`](../reports/modeling/m3/)에 있다. checkpoint 재로드
예측과 저장 CSV의 최대 절대 차이는 `1.11e-16`이었다.

M3 상태 그룹 checkpoint는 전처리 상태를 포함해 46,325 byte이며 M2 Random Forest
joblib 253,287,541 byte보다 약 5,468배 작다. 세 M3 checkpoint와 manifest·행 단위 예측을
모두 포함한 로컬 M3 artifact는 415,937 byte로, 구성 범위가 다른 M2 모델 파일과 직접적인
배포 크기 비교에는 사용하지 않는다. 작은 checkpoint는 M5 배포 후보 선정에 유리한
관찰이지만 API 지연시간과 실제 메모리 사용량은 별도로 측정해야 한다.

## 재현성과 산출물

- Python 3.13.13
- NumPy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- joblib 1.6.0
- random seed 42
- 운영체제: macOS 26.5.1 arm64

로컬 모델은 `artifacts/modeling/baseline_model.joblib`에 있으며 Git에 포함하지 않는다.

- 파일 크기: 253,287,541 byte
- SHA-256: `0c59bc9110ddd31965212bc9d46635071309ca50d4f68edbd76bffc0cd3a034a`
- 재로드 예측과 저장된 CSV의 최대 절대 차이: `1.11e-16`

joblib 파일은 pickle 기반이므로 신뢰할 수 없는 출처의 파일을 로드하지 않는다. 재현 명령과
추적되는 집계 산출물은 [`reports/modeling/`](../reports/modeling/)에 있다.
