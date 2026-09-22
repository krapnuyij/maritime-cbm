# 실험 기록

실제로 실행한 실험만 기록한다.

## 기록 규칙

- 실험 ID는 `EXP-YYYYMMDD-NNN` 형식을 사용한다.
- 성공한 실험뿐 아니라 실패하거나 중단한 실험도 원인과 함께 기록한다.
- 지표는 실제 실행 결과만 기록하며 추정값이나 예시 값을 결과처럼 남기지 않는다.
- 데이터와 artifact 경로는 저장소 루트 기준 상대경로로 기록한다.
- `docs/DATASET.md`에 기록된 데이터 해시를 Git commit과 함께 참조한다.
- 커밋하지 않은 변경이 있는 상태에서 실행했다면 dirty 상태와 관련 파일을 기록한다.
- 설정 파일을 사용한 경우 경로를 기록하고, 실행 시 덮어쓴 값은 별도로 명시한다.
- 모델 간 비교에는 동일한 데이터 분할과 평가 지표를 사용한다.

## 실험 목록

| 실험 ID | 날짜 | 목적 | 데이터 카드 | 분할 | 모델 | 상태 | 상세 기록 |
|---|---|---|---|---|---|---|---|
| EXP-20260922-001 | 2026-09-22 | validation 기반 기준 모델 선택 | `DATASET.md@129ae3c` | 상태 그룹·두 holdout validation | 17개 scikit-learn 후보 | 완료 | [상세](#exp-20260922-001--validation-기반-기준-모델-선택) |
| EXP-20260922-002 | 2026-09-22 | 고정 기준 모델 최종 평가 | `DATASET.md@129ae3c` | 네 시나리오 validation·test | Random Forest | 완료 | [상세](#exp-20260922-002--고정-기준-모델-최종-평가) |
| EXP-20260922-003 | 2026-09-22 | validation 기반 M3 모델 선택 | `DATASET.md@ef70ee1` | 상태 그룹·두 holdout validation | 6개 PyTorch 후보 × 3 seed | 완료 | [상세](#exp-20260922-003--validation-기반-m3-모델-선택) |
| EXP-20260922-004 | 2026-09-22 | 고정 M3 모델 최종 평가 | `DATASET.md@ef70ee1` | 네 시나리오 validation·test | 선형 잔차 MLP | 완료 | [상세](#exp-20260922-004--고정-m3-모델-최종-평가) |
| EXP-20260922-005 | 2026-09-22 | 회귀 예측값 기반 경보 정책 평가 | `DATASET.md@8b8d00b` | 상태 그룹 validation·네 test | M2 Random Forest·M3 선형 잔차 MLP | 완료 | [상세](#exp-20260922-005--회귀-예측값-기반-경보-정책-평가) |
| EXP-20260923-006 | 2026-09-23 | M5 Docker API 성능·운영 검증 | `DATASET.md@bc7e8da` | 상태 그룹 checkpoint·UCI 입력 | M3 선형 잔차 MLP FastAPI | 완료 | [상세](#exp-20260923-006--m5-docker-api-성능운영-검증) |

## EXP-20260922-001 — validation 기반 기준 모델 선택

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-22 11:47 KST
- Git commit: `129ae3c`
- 작업 트리 상태: 실행 시작 시 clean
- 관련 미커밋 파일: 해당 없음
- 실행 명령: `uv run --locked --group eda python -m maritime_cbm.modeling.benchmark select`
- 실험 목적: test를 사용하지 않고 기본 상태 그룹 validation NRMSE로 M2 기준 모델을 선택

### 데이터

- 데이터셋과 릴리스: UCI `Condition Based Maintenance of Naval Propulsion Plants`
- 원본 파일 상대경로: `data/raw/uci_cbm/`
- 데이터 카드 참조: `docs/DATASET.md`와 Git commit `129ae3c`
- 행·열 수: 11,934행·18열
- 입력 변수: 구조적 상수·중복 4개를 제외한 12개
- 예측 대상: `kMc`, `kMt`
- 스키마 및 격자 검증 결과: 공식 원본 해시와 9 × 51 × 26 전체 격자 검증 통과
- 전처리: 후보별 원시 입력 또는 train-only 속도 중심화, feature·target scaling은 Pipeline 내부에서 train에만 fit

### 데이터 분할

- 분할 역할: 기본 그룹 validation으로 선택, 두 holdout validation은 강건성 진단
- 분할 방식: `docs/DATASET.md`의 상태 그룹·압축기 holdout·터빈 holdout
- 학습·검증 표본 수: 상태 그룹 8,352/1,791, 압축기 9,594/1,170, 터빈 7,344/2,295
- 그룹 또는 holdout 정의: 데이터 카드의 고정 정의와 SHA-256 12개 사용
- random seed: 42

### 설정과 실행 환경

- 설정 파일: `src/maritime_cbm/config.py`, `uv.lock`
- 주요 override: 없음
- Python 버전: 3.13.13
- 의존성 상태 또는 `uv.lock` 기준: NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0
- 운영체제와 아키텍처: macOS 26.5.1 arm64
- 실행 장치: CPU

### 모델

- 모델 이름과 구현체: Dummy 1개, 원시 Ridge 5개, 속도 중심화 Ridge 5개, Random Forest 6개
- 하이퍼파라미터: Ridge alpha 0.01~100, Random Forest leaf 1·3·9와 features 1.0·sqrt, tree 300개
- 학습 중단 또는 모델 선택 기준: 상태 그룹 validation의 `kMc`, `kMt` NRMSE 평균 최소화

### 결과

| 후보 | 상태 그룹 평균 NRMSE | 최악 대상 NRMSE | 선택 |
|---|---:|---:|---|
| Random Forest, leaf 1, features 1.0 | 0.021853 | 0.026998 | 예 |
| Random Forest, leaf 1, features sqrt | 0.024000 | 0.030028 | 아니오 |
| 속도 중심화 Ridge, alpha 0.01 | 0.025371 | 0.027435 | 아니오 |
| 원시 Ridge, alpha 0.01 | 0.108917 | 0.129106 | 아니오 |
| Dummy mean | 0.282651 | 0.290092 | 선택 대상 아님 |

선택 모델의 상태 그룹 validation 결과:

| 대상 | MAE | RMSE | R² | NRMSE |
|---|---:|---:|---:|---:|
| `kMc` | 0.000412 | 0.000835 | 0.996680 | 0.016707 |
| `kMt` | 0.000286 | 0.000675 | 0.990373 | 0.026998 |

- 경보 정책 지표 및 임계값 근거: 해당 없음, M4 범위
- 주요 오류 사례: 압축기 holdout validation `kMc` NRMSE 0.096588, 터빈 holdout validation `kMt` NRMSE 0.161319로 외삽 성능 저하
- 실행 시간: 후보별 fit 합계 80.515964초, validation prediction 합계 0.832870초

### 산출물

- 모델: 해당 없음, selection 단계에서는 직렬화하지 않음
- 전처리기: 선택 manifest에 후보 Pipeline 설정 기록
- 지표: `reports/modeling/candidate_validation_metrics.csv`, `reports/modeling/model_selection_summary.csv`
- 로그와 그림: `artifacts/modeling/selection.json`, 그림은 해당 없음

### 결론

- 결과 요약: `random_forest_raw_leaf_1_features_1p0`을 고정 최종 후보로 선택
- 한계: holdout validation 결과는 외삽 진단이며 선택 점수에 반영하지 않음
- 다음 결정: selection manifest를 변경하지 않고 네 시나리오 test를 한 번 평가

## EXP-20260922-002 — 고정 기준 모델 최종 평가

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-22 11:48 KST
- Git commit: `129ae3c`
- 작업 트리 상태: dirty
- 관련 미커밋 파일: selection 단계가 생성한 `reports/modeling/candidate_validation_metrics.csv`, `reports/modeling/model_selection_summary.csv`
- 실행 명령: `uv run --locked --group eda python -m maritime_cbm.modeling.benchmark evaluate`
- 실험 목적: 고정된 최종 후보를 네 test 시나리오에서 한 번 평가

### 데이터

- 데이터셋과 릴리스: UCI `Condition Based Maintenance of Naval Propulsion Plants`
- 원본 파일 상대경로: `data/raw/uci_cbm/`
- 데이터 카드 참조: `docs/DATASET.md`와 Git commit `129ae3c`
- 행·열 수: 11,934행·18열
- 입력 변수: `v`를 포함한 확정 12개 입력
- 예측 대상: `kMc`, `kMt`
- 스키마 및 격자 검증 결과: selection manifest의 원본 해시·분할 해시 12개와 현재 환경 일치
- 전처리: 원시 12개 입력, train-only target `StandardScaler`, 예측 후 원 단위 역변환, clipping 없음

### 데이터 분할

- 분할 역할: 비교용 행 랜덤, 기본 상태 그룹, 압축기·터빈 강건성 holdout
- 분할 방식: `docs/DATASET.md`의 고정된 네 시나리오
- 학습·검증·테스트 표본 수: 데이터 카드와 selection manifest 참조
- 그룹 또는 holdout 정의: 상태 그룹 조합과 연속 저계수 구간
- random seed: 42

### 설정과 실행 환경

- 설정 파일: `src/maritime_cbm/config.py`, `artifacts/modeling/selection.json`, `uv.lock`
- 주요 override: 없음
- Python 버전: 3.13.13
- 의존성 상태 또는 `uv.lock` 기준: NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0
- 운영체제와 아키텍처: macOS 26.5.1 arm64
- 실행 장치: CPU

### 모델

- 모델 이름과 구현체: `m2-random-forest-baseline-v1`, scikit-learn `RandomForestRegressor`
- 하이퍼파라미터: tree 300개, `min_samples_leaf=1`, `max_features=1.0`, `random_state=42`, `n_jobs=1`
- 학습 중단 또는 모델 선택 기준: EXP-20260922-001 결과를 고정해 재선택하지 않음

### 결과

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

- 경보 정책 지표 및 임계값 근거: 해당 없음, M4 범위
- 주요 오류 사례: 압축기 `kMc`와 터빈 `kMt` holdout에서 각각 `+0.009668`, `+0.008830` 건강 방향 bias, 상태 그룹 3 knots 오차 증가
- 실행 시간: 네 모델 fit 합계 29.446586초, validation·test prediction 합계 0.478417초

### 산출물

- 모델: `artifacts/modeling/baseline_model.joblib`, SHA-256 `0c59bc9110ddd31965212bc9d46635071309ca50d4f68edbd76bffc0cd3a034a`
- 전처리기: 모델 joblib 내부 target scaler
- 지표: `reports/modeling/final_metrics.csv`, 속도·상태별 집계 CSV
- 로그와 그림: `artifacts/modeling/evaluation.json`, `reports/modeling/` PNG 3개

### 결론

- 결과 요약: 상태 그룹 test에서는 높은 정확도를 보였지만 심한 열화 방향 외삽에는 실패
- 한계: 실제 고장·시간 진행을 예측하지 않으며 holdout에서 더 건강한 상태로 과대 추정
- 다음 결정: M3 모델도 동일한 선택·평가 규칙으로 비교하고 M4 경보 정책에서 holdout bias를 명시적으로 검토

## EXP-20260922-003 — validation 기반 M3 모델 선택

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-22 15:34 KST
- Git commit: `ef70ee1`
- 작업 트리 상태: 실행 시작 시 clean
- 관련 미커밋 파일: 해당 없음
- 실행 명령: `uv run --locked --group eda --group modeling python -m maritime_cbm.modeling.torch_benchmark select --device cpu`
- 실험 목적: test를 사용하지 않고 세 seed 상태 그룹 validation NRMSE로 M3 비교 모델 선택

### 데이터·분할

- 데이터셋: UCI `Condition Based Maintenance of Naval Propulsion Plants`, 11,934행·18열
- 데이터 카드 참조: `docs/DATASET.md`와 Git commit `ef70ee1`
- 입력·대상: 확정 12개 입력, `kMc`, `kMt`
- 분할: 상태 그룹 validation으로 선택, 두 holdout validation은 early stopping과 외삽 진단
- random seed: 42·43·44
- 전처리: 후보별 원시 입력 또는 train-only 속도 중심화, train-only feature·target 표준화

### 설정과 실행 환경

- Python 3.13.13, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, PyTorch 2.14.0
- 운영체제와 장치: macOS 26.5.1 arm64, CPU
- 학습: float32, AdamW, learning rate 0.001, weight decay 0.0001, batch 256, 최대 500 epoch, patience 40

### 결과

| 후보 | 상태 그룹 평균 NRMSE | 최악 대상 NRMSE | 선택 |
|---|---:|---:|---|
| 선형 잔차 MLP, speed-centered, 128·64 | 0.003762 | 0.004659 | 예 |
| MLP, speed-centered, 128·64 | 0.003969 | 0.004987 | 아니오 |
| 선형 잔차 MLP, speed-centered, 64·32 | 0.004152 | 0.004974 | 아니오 |
| MLP, speed-centered, 64·32 | 0.004260 | 0.005137 | 아니오 |

- 선택 모델 대상별 세 seed 평균 NRMSE: `kMc` 0.002866, `kMt` 0.004659
- 실행 시간: 54회 학습 합계 327.883961초
- 산출물: `artifacts/modeling/m3/selection.json`, seed 42 checkpoint 3개, `reports/modeling/m3/` 선택 CSV 3개
- 결론: `linear_residual_mlp_speed_centered_hidden_128_64`를 고정 최종 후보로 선택

## EXP-20260922-004 — 고정 M3 모델 최종 평가

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-22 15:35 KST
- Git commit: `ef70ee1`
- 작업 트리 상태: dirty
- 관련 미커밋 파일: selection 단계가 생성한 `reports/modeling/m3/` CSV 3개
- 실행 명령: `uv run --locked --group eda --group modeling python -m maritime_cbm.modeling.torch_benchmark evaluate --device cpu`
- 실험 목적: 고정된 M3 checkpoint를 네 test 시나리오에서 한 번 평가하고 M2와 비교

### 절차와 환경

- 데이터·분할: `docs/DATASET.md`의 고정된 네 시나리오와 SHA-256 12개
- checkpoint: 상태 그룹·압축기·터빈은 selection seed 42 checkpoint 재사용, 행 랜덤만 신규 fit
- manifest 검증: 원본·분할 해시, 후보 grid, 학습 설정, 런타임 버전과 PyTorch 버전 일치
- 실행 환경: Python 3.13.13, PyTorch 2.14.0, macOS 26.5.1 arm64, CPU
- clipping: 적용하지 않음

### 결과

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

- M2 대비 NRMSE 감소: 상태 그룹 `kMc` 82.8%, `kMt` 78.8%, 압축기 holdout `kMc` 90.2%, 터빈 holdout `kMt` 84.9%
- 주요 한계: holdout bias가 압축기 `kMc` +0.000695, 터빈 `kMt` +0.000996으로 건강 방향에 남아 있음
- 실행 시간: 네 시나리오 학습 기록 합계 17.831090초, validation·test prediction 합계 0.005996초
- checkpoint: `artifacts/modeling/m3/checkpoints/state_group_seed_42.pt`, SHA-256 `cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472`
- checkpoint 재로드 예측과 저장 CSV 최대 절대 차이: `1.11e-16`
- 지표·그림: `reports/modeling/m3/`, 행 단위 예측과 evaluation manifest는 `artifacts/modeling/m3/`
- 결론: M3는 M2의 격자 내부 오차와 tree 외삽 포화를 크게 줄였으나 실제 고장진단 또는 미관측 조건 보장을 의미하지 않음
- 다음 결정: M4에서 M3 자동 교체 여부를 별도로 판단하고, 경보 정책에 남은 건강 방향 bias를 반영

M2 holdout 결과를 본 뒤 M3 구조를 설계했으므로 두 holdout 비교는 완전히 미관측인 독립
test가 아니라 사전에 고정한 벤치마크의 탐색적 비교다. M3는 selection checkpoint를 최종
평가에 재사용했고 M2는 네 시나리오를 평가 단계에서 다시 학습했다.

## EXP-20260922-005 — 회귀 예측값 기반 경보 정책 평가

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-22 18:40 KST
- Git commit: `8b8d00b`
- 작업 트리 상태: dirty
- 관련 미커밋 파일: 첫 실행에서 생성한 `reports/alerting/`과 결과 문서화 중인 README·문서 3개, 코드 변경 없음
- 실행 명령: `uv run --locked --group eda --group modeling python -m maritime_cbm.alerting.benchmark`
- 실험 목적: 고정된 M2·M3 회귀 예측으로 PoC 경보 정책의 민감도와 누락·오경보 trade-off 평가

### 데이터와 upstream artifact

- 데이터셋: UCI `Condition Based Maintenance of Naval Propulsion Plants`, 11,934행·18열
- 데이터 카드 참조: `docs/DATASET.md`와 Git commit `8b8d00b`
- 분할: 상태 그룹 validation과 행 랜덤·상태 그룹·압축기·터빈 holdout test
- M2 모델 SHA-256: `0c59bc9110ddd31965212bc9d46635071309ca50d4f68edbd76bffc0cd3a034a`
- M2 test 예측 SHA-256: `6c3ba563d65d539daba5697a4e7fc321134127c5666d899941b3df9b78b69417`
- M3 checkpoint SHA-256: `cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472`
- M3 test 예측 SHA-256: `d1632e09d1db378f0e6a44ec185ac916344234101c2da37dbd094796ccef10c7`
- 처리 방식: upstream manifest·분할·런타임·artifact 해시 검증 후 재학습 없이 기존 test 예측 재사용

### 설정과 실행 환경

- 정책: 열화도 clipping 없음, `watch` 0.5, 주 `alert` 0.8
- 민감도 임계값: 0.5·0.6·0.7·0.8·0.9
- 고정 오경보율 목표: 상태 그룹 validation FPR 1%·5%
- random seed: 42
- Python 3.13.13, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, PyTorch 2.14.0, joblib 1.6.0
- 운영체제와 아키텍처: macOS 26.5.1 arm64
- 모델 학습: 수행하지 않음

### 주 임계값 0.8 결과

| 시나리오·채널 | M2 Recall | M2 FPR | M3 Recall | M3 FPR |
|---|---:|---:|---:|---:|
| 상태 그룹 · `kMc` | 0.944444 | 0.000000 | 0.962963 | 0.000000 |
| 상태 그룹 · `kMt` | 0.891975 | 0.000682 | 0.944444 | 0.000000 |
| 상태 그룹 · `any` | 0.924501 | 0.000918 | 0.955840 | 0.000000 |
| 압축기 holdout · `kMc` | 0.000000 | NA | 1.000000 | NA |
| 터빈 holdout · `kMt` | 0.000000 | NA | 0.931590 | NA |

- 상태 그룹 `any` F1/PR-AUC: M2 0.960059/0.998252, M3 0.977422/0.999947
- 압축기·터빈 holdout 대상은 모두 양성이므로 Precision·F1·FPR·PR-AUC는 `NA`
- M2 holdout 대상 FN: 압축기 1,170/1,170, 터빈 2,295/2,295
- M3 holdout 대상 FN: 압축기 0/1,170, 터빈 157/2,295

### validation 고정 오경보율 결과

| 모델 | 목표 FPR | validation `any` cutoff | 상태 그룹 test Recall | 상태 그룹 test FPR |
|---|---:|---:|---:|---:|
| M2 Random Forest | 0.01 | 0.785200 | 0.981481 | 0.004591 |
| M2 Random Forest | 0.05 | 0.758533 | 0.992877 | 0.058770 |
| M3 선형 잔차 MLP | 0.01 | 0.780506 | 0.998575 | 0.007346 |
| M3 선형 잔차 MLP | 0.05 | 0.779096 | 0.998575 | 0.017447 |

목표 FPR은 validation 제약이며 test realized FPR을 보장하지 않는다. 정상 표본이 없는
holdout에서는 고정 cutoff Recall만 보고하고 FPR은 `NA`로 기록했다.

### 산출물과 결론

- 행 단위 정책 예측: `artifacts/alerting/policy_predictions.csv.gz`
- 행 단위 예측 SHA-256: `d21156993e179c5f0968aeada29bf8d56e62eb8c2220e93246be25ec1c2bdad8`
- 결정성 확인: 다른 출력 경로에서 재실행한 gzip 바이트·SHA-256과 집계 CSV 5개가 모두 일치
- manifest: `artifacts/alerting/evaluation.json`
- 집계 지표와 그림: `reports/alerting/`
- 결과 요약: M3는 상태 그룹 성능을 개선하고 M2의 심한 열화 방향 경보 완전 누락을 크게 줄임
- 한계: reference는 실제 고장이 아닌 simulator 상태 계수 기반이며 두 holdout 대상은 단일 클래스
- 비지도 이상탐지: 근거 있는 정상 모집단이 없어 Isolation Forest·Autoencoder를 수행하지 않음
- 다음 결정: M5에서 배포 모델과 API 계약을 확정하고 실제 지연시간·메모리를 측정

## EXP-20260923-006 — M5 Docker API 성능·운영 검증

### 실행 정보

- 상태: 완료
- 실행 일시: 2026-09-23 00:42 KST
- Git commit: `bc7e8da525518234cddf983e3e667043a9a6712f`
- 작업 트리 상태: dirty
- 관련 미커밋 파일: `scripts/benchmark_service.py`
- 실행 명령:

  ```bash
  uv run --locked --group service python scripts/benchmark_service.py \
    --image maritime-cbm:m5-local \
    --checkpoint artifacts/modeling/m3/checkpoints/state_group_seed_42.pt \
    --data data/raw/uci_cbm/data.txt \
    --output-directory reports/service \
    --host-port 18082 \
    --platform linux/arm64
  ```

- 실험 목적: 실제 M3 checkpoint를 제공하는 Docker API의 고정 조건 지연시간·메모리와
  checkpoint 불변성 확인
- 모델 학습: 수행하지 않음

### 데이터와 배포 artifact

- 데이터셋: UCI `Condition Based Maintenance of Naval Propulsion Plants`, 11,934행·18열
- 데이터 카드 참조: `docs/DATASET.md`와 Git commit `bc7e8da`
- 요청 payload: 공식 데이터의 첫 100행에서 선택한 확정 12개 입력
- 배포 모델: `m3-linear-residual-mlp-v1`, 상태 그룹 seed 42 checkpoint
- checkpoint SHA-256: `cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472`
- checkpoint 처리: read-only mount, benchmark 전후 SHA-256 일치

### 고정 실행 조건

- host: macOS 26.5.1 arm64, Docker server 29.5.3
- container: Debian GNU/Linux 13, Linux/aarch64, Python 3.13.15
- runtime: PyTorch 2.14.0+cpu, FastAPI 0.141.1, Uvicorn 0.53.0
- service: Uvicorn worker 1개, concurrency 1, 하나의 지속 HTTP/1.1 client로 순차 요청
- protocol: warm-up 50회, cold start 5회, 단건 1,000회, 100건 batch 200회
- Docker image: `sha256:c0405643bca6befec748bc3814eb1befbd57c9109923f6a864b73f80d8f82b7c`
- image 크기: 354,694,718 byte, 338.263MiB

### 결과

| 작업 | 평균 | 중앙값 | p95 | 요청 오류 |
|---|---:|---:|---:|---:|
| cold start | 1,452.024ms | 1,301.285ms | 1,780.138ms | 0/5 |
| 단건 상태 추정 | 1.438ms | 1.277ms | 2.254ms | 0/1,000 |
| 100건 batch 상태 추정 | 3.901ms | 3.855ms | 5.127ms | 0/200 |

- idle process RSS: 352.652MiB
- process peak RSS: 355.934MiB
- 요청 후 Docker cgroup 사용량: 242.9MiB
- 산출물: `reports/service/latency_summary.csv`, `reports/service/benchmark_results.json`

1차 실행은 모든 요청이 끝난 뒤 Docker stats의 공백 없는 `242.9MiB` 문자열을 두 token으로
가정한 parser 오류로 결과 저장 전에 실패했다. 이는 API나 모델 오류가 아니다. 단위를
정규식으로 해석하도록 수정한 뒤 같은 고정 프로토콜 전체를 처음부터 다시 실행해 위 결과를
얻었다.

### 결론과 한계

- 실제 checkpoint를 Linux container에서 로드해 요청 오류 없이 단건·배치 추론을 완료했다.
- 합성 smoke 결과가 아니라 실제 artifact와 UCI 입력으로 얻은 운영 측정이다.
- process RSS와 Docker cgroup 값은 계측 범위가 다르므로 직접 차감하거나 같은 지표로
  비교하지 않는다.
- 단일 ARM64 MacBook Docker, concurrency 1 결과이므로 Linux/X64, 높은 동시성, 클라우드
  처리량이나 SLA로 일반화하지 않는다.
- 이 실험은 서비스 운영 특성 검증이며 실제 고장진단 성능 검증이 아니다.

## 실험별 기록 양식

아래 양식을 복사해 실험 목록 다음에 추가한다. 해당하지 않는 항목은 삭제하지 않고
`해당 없음`과 그 이유를 기록한다.

````markdown
## EXP-YYYYMMDD-NNN — 실험 목적

### 실행 정보

- 상태: 완료 / 실패 / 중단
- 실행 일시: YYYY-MM-DD HH:MM KST
- Git commit:
- 작업 트리 상태: clean / dirty
- 관련 미커밋 파일:
- 실행 명령:
- 실험 목적:

### 데이터

- 데이터셋과 릴리스:
- 원본 파일 상대경로:
- 데이터 카드 참조: `docs/DATASET.md`와 해당 Git commit
- 행·열 수:
- 입력 변수:
- 예측 대상:
- 스키마 및 격자 검증 결과:
- 전처리:

### 데이터 분할

- 분할 역할: 비교용 랜덤 / 기본 그룹 / 강건성 holdout
- 분할 방식:
- 학습·검증·테스트 비율 또는 표본 수:
- 그룹 또는 holdout 정의:
- random seed:

### 설정과 실행 환경

- 설정 파일:
- 주요 override:
- Python 버전:
- 의존성 상태 또는 `uv.lock` 기준:
- 운영체제와 아키텍처:
- 실행 장치: CPU / MPS

### 모델

- 모델 이름과 구현체:
- 하이퍼파라미터:
- 학습 중단 또는 모델 선택 기준:

### 결과

| 대상 | MAE | RMSE | R² |
|---|---:|---:|---:|
| `kMc` |  |  |  |
| `kMt` |  |  |  |
| 전체 |  |  |  |

- 경보 정책 지표 및 임계값 근거:
- 주요 오류 사례:
- 실행 시간:

### 산출물

- 모델:
- 전처리기:
- 지표:
- 로그와 그림:

### 결론

- 결과 요약:
- 한계:
- 다음 결정:
````
