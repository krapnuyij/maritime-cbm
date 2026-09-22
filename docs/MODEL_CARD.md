# Maritime CBM 모델 카드

## 모델 개요

- 모델 ID: `m2-random-forest-baseline-v1`
- 상태: M2 기준 모델
- 구현체: scikit-learn 1.9.1 `RandomForestRegressor`
- 목적: 정상상태 시뮬레이션 센서값에서 `kMc`, `kMt` 열화 상태 계수를 동시에 추정
- 기준 Git commit: `129ae3c`

이 모델은 실제 선박 고장진단 모델이 아니다. 실제 고장 라벨, 타임스탬프와 공식 경보
임계값이 없는 공개 시뮬레이션 데이터로 만든 PoC 기준 모델이다.

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
