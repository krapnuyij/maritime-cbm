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
