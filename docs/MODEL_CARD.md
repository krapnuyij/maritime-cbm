# Maritime CBM 모델 카드

## 모델 개요

- M2 기준 모델 ID: `m2-random-forest-baseline-v1`
- M5 1차 배포 모델 ID: `m3-linear-residual-mlp-v1`
- 상태: M2·M3 비교, M4 경보 정책과 M5 FastAPI·Docker 로컬 검증 완료
- M2 구현체: scikit-learn 1.9.1 `RandomForestRegressor`
- M3 구현체: PyTorch 2.14.0 선형 잔차 MLP
- 목적: 정상상태 시뮬레이션 센서값에서 `kMc`, `kMt` 열화 상태 계수를 동시에 추정
- M2 기준 Git commit: `129ae3c`

이 모델은 실제 선박 고장진단 모델이 아니다. 실제 고장 라벨, 타임스탬프와 공식 경보
임계값이 없는 공개 시뮬레이션 데이터로 만든 PoC 기준 모델이다.

M3 결과와 M4 경보 정책 분석, artifact 크기와 macOS 예비 운영 측정을 함께 검토해 M3를
M5 1차 배포 모델로 확정했다. M2는 비교 기준으로 보존한다. 실제 M3 checkpoint를 사용한
Docker/Linux 기동·추론과 고정 프로토콜 운영 측정도 완료했다.

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
- 직렬화한 모델이 약 253MB이고 macOS 예비 측정에서도 M3보다 메모리와 추론 지연이 커
  M5 1차 배포 모델로 사용하지 않는다.

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
배포 크기 비교에는 사용하지 않는다. 작은 checkpoint와 M2 대비 낮은 예비 메모리·추론
지연은 M5 배포 모델 채택 근거이며, Docker/Linux API 측정 결과는 아래 M5 절에 별도로
기록한다.

## M4 회귀 예측값 기반 경보 정책

M4는 M2·M3 모델을 재학습하지 않고 각 evaluation manifest와 저장된 test 예측의 SHA-256을
검증한 뒤 수행한 downstream 정책 분석이다. reference alert도 실제 고장 라벨이 아니라
시뮬레이터 열화 상태 계수에서 파생한다.

- `kMc` 열화도: `(1-kMc)/0.050`
- `kMt` 열화도: `(1-kMt)/0.025`
- 전체 열화도: 두 열화도의 최댓값
- 상태: 0.5 미만 `normal`, 0.5 이상 0.8 미만 `watch`, 0.8 이상 `alert`

열화도는 회귀 예측의 공식 범위 이탈을 보존하기 위해 clipping하지 않는다. 주 임계값 0.8은
`kMc ≤ 0.960`, `kMt ≤ 0.980`에 해당하는 사전 고정 PoC 정책이며 공식 고장 임계값이
아니다.

### 주 임계값 test 결과

| 시나리오·채널 | M2 Recall | M2 FPR | M3 Recall | M3 FPR |
|---|---:|---:|---:|---:|
| 상태 그룹 · `kMc` | 0.944444 | 0.000000 | 0.962963 | 0.000000 |
| 상태 그룹 · `kMt` | 0.891975 | 0.000682 | 0.944444 | 0.000000 |
| 상태 그룹 · `any` | 0.924501 | 0.000918 | 0.955840 | 0.000000 |
| 압축기 holdout · `kMc` | 0.000000 | NA | 1.000000 | NA |
| 터빈 holdout · `kMt` | 0.000000 | NA | 0.931590 | NA |

M2 Random Forest는 압축기 `kMc`와 터빈 `kMt`의 심한 열화 방향 holdout에서 학습 경계
아래로 외삽하지 못해 대상 경보 Recall이 모두 0이었다. M3 선형 잔차 MLP는 각각 1.0000,
0.9316으로 개선했지만 터빈 대상 2,295건 중 157건을 놓쳤다. 이 개선은 M3를 M5 1차 배포
모델로 채택한 근거지만 실제 고장 탐지 성능이나 현장 안전성을 뜻하지 않는다.

압축기·터빈 holdout의 대상 reference는 전부 양성이다. 따라서 이 두 행의 Precision·F1·
FPR·PR-AUC는 모델 판별력을 나타낼 수 없어 `NA`로 기록했다. Recall·FN·miss rate만
해석하며, 이를 실제 고장 누락률이라고 부르지 않는다.

### validation 고정 오경보율

상태 그룹 validation에서 모델·채널별 목표 FPR 1%·5% 이하인 cutoff를 선택하고 test에
변경 없이 적용했다. `any` 채널 결과는 다음과 같다.

| 모델 | 목표 FPR | cutoff | 상태 그룹 test Recall | 상태 그룹 test FPR |
|---|---:|---:|---:|---:|
| M2 Random Forest | 0.01 | 0.785200 | 0.981481 | 0.004591 |
| M2 Random Forest | 0.05 | 0.758533 | 0.992877 | 0.058770 |
| M3 선형 잔차 MLP | 0.01 | 0.780506 | 0.998575 | 0.007346 |
| M3 선형 잔차 MLP | 0.05 | 0.779096 | 0.998575 | 0.017447 |

목표 FPR은 validation에서의 cutoff 선택 제약이므로 test realized FPR을 보장하지 않는다.
M2의 목표 5% cutoff가 상태 그룹 test에서 5.88% FPR을 보인 것이 이 차이를 보여 준다.
정상 표본이 없는 holdout에서는 고정 cutoff Recall만 해석하고 FPR은 `NA`로 유지한다.

### 경보 정책의 제한사항

- 임계값 0.8은 simulator 범위의 하위 20%를 표시하기 위한 정책 시나리오일 뿐 실제 정비·
  안전 기준이 아니다.
- M4는 이미 평가한 M2·M3 test 예측을 사용한 후속 분석이므로 새로운 독립 test가 아니다.
- 실제 고장 라벨이 없어 Precision·Recall 등의 명칭은 파생된 열화 상태와 정책 임계값에 대한
  일치도를 뜻한다.
- 타임스탬프가 없어 지속 시간, debounce와 hysteresis를 평가하지 않았다.
- 근거 있는 정상 모집단이 없어 Isolation Forest와 Autoencoder를 수행하지 않았다.
- 시뮬레이션과 실제 선박 사이의 domain gap, 미관측 운항 속도와 현장 오경보 비용은 검증하지
  않았다.

집계 결과와 그림은 [`reports/alerting/`](../reports/alerting/)에 있다. 비추적 행 단위 정책
예측의 SHA-256은
`d21156993e179c5f0968aeada29bf8d56e62eb8c2220e93246be25ec1c2bdad8`이다. gzip의 파일명과
생성 시각 metadata를 제거해 다른 경로에서 재실행해도 같은 해시가 생성됨을 확인했다.

## M5 배포 결정과 API 계약

M5 1차 배포 모델은 상태 그룹 seed 42 checkpoint를 사용하는
`m3-linear-residual-mlp-v1`이다. 추적되는 `config/deployment_model.json`에
모델·정책 버전, checkpoint SHA-256, target·입력 순서, 허용 속도와 센서 범위, 경보
임계값을 기록했다. 서비스는 계약과 실제 checkpoint·전처리 상태가 다르면 시작하지 않는다.
계약·checkpoint·runtime의 PyTorch 기본 버전도 3자 대조하며 Linux CPU wheel의 `+cpu`
suffix는 기본 버전 비교에서 제외한다.

상태 추정 API는 다음 12개 이름만 입력받으며 추가 필드, NaN과 무한대를 허용하지 않는다.

```text
v, GTT, GTn, GGn, Ts, T48, T2, P48, P2, Pexh, TIC, mf
```

`v`는 `{3, 6, 9, 12, 15, 18, 21, 24, 27}` 중 하나여야 한다. 나머지 센서는 기본 상태
그룹 train에서 관측한 [`DATASET.md`](DATASET.md)의 양 끝 포함 min/max를 적용한다. 단건은
12개 센서 한 건, 배치는 1~100건을 받는다. 범위 밖 입력은 clipping하지 않고 `422`로
거부한다.

- `GET /health`: 서비스와 모델 로드 상태
- `GET /model/info`: 모델·정책 버전, 입력 계약과 경보 기준
- `POST /v1/condition/predict`: 모델 버전과 `kMc`, `kMt` 상태 계수 추정
- `POST /v1/condition/batch`: 입력 순서를 보존한 1~100건 상태 계수 추정
- `POST /v1/alert/evaluate`: 입력한 `kMc`, `kMt`의 대상별·전체 열화도, 상태와 판단 기준

경보 평가 입력은 유한해야 하지만 공식 계수 범위로 제한하거나 clipping하지 않는다. 이는
회귀 예측의 범위 이탈을 M4 정책에서 그대로 해석하기 위한 결정이다. 상태 추정과 경보
평가는 별도 endpoint이므로 예측 응답을 경보나 고장 판정으로 오해하지 않아야 한다.

### 예비 운영 측정

아래 값은 재학습 없이 기존 artifact를 macOS 로컬에서 한 번 측정한 예비 결과다. 실행 명령,
부하 조건과 반복 측정이 고정된 최종 benchmark가 아니므로 방향성 확인에만 사용한다.

| 항목 | M2 Random Forest | M3 선형 잔차 MLP |
|---|---:|---:|
| import + model load | 1.256s | 1.582s |
| process peak RSS | 591.9MB | 294.2MB |
| 단건 예측 평균 지연시간 (`N=200`) | 5.572ms | 0.164ms |
| 100건 batch 예측 | 5.987ms | 0.227ms |

M3는 PyTorch import 비용 때문에 cold start가 약 0.3초 느렸지만 peak RSS는 약 절반이고
단건 추론은 약 34배 빨랐다. macOS와 Linux의 `ru_maxrss` 단위가 다르므로 이 수치를
Docker/Linux 결과와 직접 비교하지 않는다.

### Docker/Linux API benchmark

실제 상태 그룹 seed 42 checkpoint를 재학습 없이 read-only로 마운트하고 다음 고정 조건에서
측정했다.

- 환경: Debian GNU/Linux 13, Linux/aarch64, Python 3.13.15, PyTorch 2.14.0+cpu
- 서비스: FastAPI 0.141.1, Uvicorn 0.53.0, worker 1개, 순차 HTTP/1.1 요청
- 프로토콜: warm-up 50회, cold start 5회, 단건 1,000회, 100건 batch 200회
- Docker image: `sha256:c0405643bca6befec748bc3814eb1befbd57c9109923f6a864b73f80d8f82b7c`

| 작업 | 평균 | 중앙값 | p95 | 요청 오류 |
|---|---:|---:|---:|---:|
| cold start | 1,452.024ms | 1,301.285ms | 1,780.138ms | 0/5 |
| 단건 상태 추정 | 1.438ms | 1.277ms | 2.254ms | 0/1,000 |
| 100건 batch 상태 추정 | 3.901ms | 3.855ms | 5.127ms | 0/200 |

idle process RSS는 352.652MiB, 요청 후 peak process RSS는 355.934MiB였다. 같은 시점의
Docker cgroup 사용량은 242.9MiB이며, 계측 범위가 다른 값이므로 process RSS와 직접
차감하거나 같은 지표처럼 비교하지 않는다. image 크기는 338.263MiB다.

이 결과는 macOS ARM64 호스트 한 대의 Docker/Linux ARM64, concurrency 1 조건이다. 클라우드
처리량, 높은 동시성, Linux/X64 성능이나 운영 SLA를 보장하지 않는다. 실제 고장진단 성능을
검증한 결과도 아니다. 명령과 전체 결과는 [`reports/service/`](../reports/service/)에 있다.

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

M5 서비스는 checkpoint를 image에 포함하지 않고 read-only volume으로 마운트한다. 로드 전에
배포 계약의 SHA-256을 검증하고 `torch.load(..., weights_only=True)` 경로를 사용한다. 이
검증은 신뢰할 수 없는 artifact를 안전하게 만드는 보안 경계가 아니므로 checkpoint 출처는
계속 신뢰해야 한다. M5 측정 환경과 결과 JSON은 [`reports/service/`](../reports/service/)에
추적하며 실제 checkpoint 파일은 Git에 포함하지 않는다.
