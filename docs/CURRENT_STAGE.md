# 현재 진행 상황

## 현재 마일스톤

M5. 서비스화

## 완료

- 기본 브랜치가 `main`인 로컬 Git 저장소 초기화
- Python 3.13과 uv 기반 개발 환경 및 의존성 잠금 구성
- `src/maritime_cbm/` 패키지와 `tests/` 기본 구조 구성
- Ruff와 pytest 설정 및 smoke test 통과
- 중앙 config와 표준 `logging` 구성
- UCI 원본 파일 로더, 입력·정답 분리, SHA-256 계산과 필수 스키마·격자 검증 모듈 구현
- 상수·중복 열과 `lp`–`v` 관계를 실패와 분리해 관찰하는 검증 보고서 구현
- 실제 UCI 원본 11,934행·18열, 결측값과 공식 격자 구조 검증 완료
- `Features.txt`와 `README.txt`의 변수 순서를 공식 스키마와 대조 완료
- 구조적으로 중복되거나 정보가 없는 4개 입력을 제외하고 12개 모델 입력 확정
- 비교용 행 랜덤, 기본 상태 그룹, 압축기·터빈 연속 열화 구간 holdout 분할 구현
- 네 분할 시나리오의 12개 행 인덱스 SHA-256을 실제 원본으로 생성하고 통합 테스트에 고정
- 전체 데이터 집계 통계, pooled·속도 조건부 상관 및 속도 간 분산 비율 CSV와 핵심 EDA 그림 3개 생성
- 합성 상태 격자의 분할 해시 회귀 테스트와 Ubuntu 24.04 Linux CI 구성
- 첫 GitHub Actions Linux/X64 실행에서 lock·Ruff·포맷·pytest 검증 통과
- scikit-learn 1.9.1 기반 다중 출력 기준 모델 후보와 누수 방지 Pipeline 구현
- 17개 후보를 validation에서 비교하고 상태 그룹 NRMSE 기준 최종 Random Forest 선택
- 고정된 최종 모델을 네 시나리오 test에서 한 번 평가하고 외삽 포화 확인
- 집계 지표·오류 분석 그림, 실험 기록과 `MODEL_CARD.md` 작성
- PyTorch 2.14 기반 6개 신경망 후보와 결정적 CPU 학습·checkpoint Pipeline 구현
- 세 seed 상태 그룹 validation으로 속도 중심화 선형 잔차 MLP 선택
- 고정 checkpoint를 네 시나리오 test에서 한 번 평가하고 M2 기준 모델과 비교
- 상태 그룹과 심한 열화 방향 holdout에서 Random Forest 대비 NRMSE 감소 확인
- 회귀 예측값 기반 열화도와 `normal`·`watch`·`alert` 정책 모듈 구현
- 단일 클래스 안전 경보 지표와 validation 고정 FPR cutoff 구현
- M2·M3 artifact 검증 후 재학습 없이 M4 경보 정책 공식 평가 완료
- M3가 M2의 심한 열화 방향 경보 누락을 크게 줄이는 결과 확인
- README 초안과 코드용 MIT License 작성
- UCI 데이터 출처, CC BY 4.0 라이선스, 인용 및 다운로드 방법 문서화
- UCI `Condition Based Maintenance of Naval Propulsion Plants` 릴리스 선택
- 16개 운항·센서 입력과 2개 정답 `kMc`, `kMt`로 문제 정의 확정
- 타임스탬프가 없는 정상상태 시뮬레이션 데이터라는 해석 범위 확정
- 다중 출력 회귀, 기본 그룹 분할, holdout 강건성 평가 및 회귀 예측값 기반 경보 정책 방향 문서화

## 진행 중

- 확정된 M5 배포 모델·API 계약을 기준으로 FastAPI·Docker 서비스화 구현 계획 수립 준비

## 진행 관리 원칙

- 별도의 전체 구현 계획 문서를 만들지 않고 `PROJECT_SPEC.md`와 `CURRENT_STAGE.md`로 범위와 진행 상태를 관리한다.
- 미확정 사항은 해당 마일스톤에서 근거를 확인한 뒤 결정한다.
- 결정이 확정되면 이 문서의 `미확정 사항`에서 제거하고 `확정된 결정`에 근거와 함께 기록한다.
- `DATASET.md`는 M1부터 실제 데이터 검증 결과를 누적하는 데이터 카드로 관리한다.
- `MODEL_CARD.md`는 M2에서 최종 기준 모델을 선정한 후 생성하고 M3~M5에서 갱신한다.

## 다음 작업

1. M5 FastAPI·Docker 전체 구현 계획 수립과 승인
2. 배포 계약 파일, FastAPI 추론·경보 API와 입력 검증 구현
3. Docker 실행 환경과 smoke test 구성
4. 고정 Docker/Linux 조건의 API 지연시간·메모리 측정과 포트폴리오 문서 최종화

## 확정된 결정

- 프로젝트명: Maritime CBM
- Python 버전: 3.13
- 패키지 관리 방식: uv와 `uv.lock`
- M1에서 `src/maritime_cbm/config.py`를 생성해 random seed 기본값 42와 데이터 경로를 중앙 관리한다.
- 로깅은 Python 표준 `logging`을 사용하고 별도 로깅 라이브러리는 추가하지 않는다.
- 외부 설정 파일은 실제 필요가 확인될 때 도입하며 M1 시작 시점에는 추가하지 않는다.
- 프로젝트 코드 라이선스: MIT License
- UCI 데이터셋 라이선스: CC BY 4.0
- 사용 데이터: UCI `Condition Based Maintenance of Naval Propulsion Plants`
- 공식 설명 기준 데이터 구조: 11,934행, 16개 입력, 2개 정답
- 예측 대상: `kMc`, `kMt`
- 핵심 과제: `kMc`, `kMt` 다중 출력 회귀
- `kMc`, `kMt`는 관측된 실제 고장 라벨이 아니라 센서값 생성에 사용된 시뮬레이터 열화 상태 계수다.
- 모델은 정상상태 센서값으로부터 `kMc`, `kMt`를 역으로 추정한다.
- 공식 격자 구조의 운항 속도는 3~27 knots, 3 knots 간격이며 예상 고유값은 9개다.
- 공식 격자 구조의 `kMc`는 0.950~1.000, 0.001 간격이며 예상 고유값은 51개다.
- 공식 격자 구조의 `kMt`는 0.975~1.000, 0.001 간격이며 예상 고유값은 26개다.
- 공식 격자 구조의 예상 행 수는 9 × 51 × 26 = 11,934이다.
- 실제 파일 로드 후 행·열 수, 변수 순서와 격자 구조를 공식 README와 대조한다.
- 부동소수점 격자 값은 허용오차 또는 반올림 후 검증한다.
- 실제 원본 데이터는 11,934행·18열이며 결측값 없이 공식 격자 전체를 구성한다.
- `T1`, `P1`은 상수이고 `Ts`, `Tp`는 동일하며 `lp`, `v`는 1:1로 대응한다.
- 모델 입력에서는 상수 `T1`, `P1`, 중복 `Tp`, `v`의 1:1 proxy인 `lp`를 제거하고 `v`, `Ts`를 포함한 12개를 사용한다.
- 구조적 특성 제거 규칙은 `schema.py`에서 중앙 관리하고 학습·추론이 같은 선택 함수를 사용한다.
- 원본 데이터는 재배포하지 않으며 현재 UCI CC BY 4.0 표기와 배포 README의 과거 상업적 이용 금지 문구 간 충돌을 문서화한다.
- 데이터는 타임스탬프가 없는 정상상태 시뮬레이션 데이터로 해석한다.
- 시계열 예측, 미래 고장 예측 또는 실제 선박 데이터라고 표현하지 않는다.
- 행 단위 랜덤 분할은 비교용 기준으로만 사용한다.
- 행 단위 랜덤 분할은 seed 42로 8,354/1,790/1,790행을 배정한다.
- `(kMc, kMt)` 상태 조합 그룹 1,326개 중 경계 그룹 150개는 학습에 고정하고, 내부 그룹을 포함한 최종 928/199/199개 그룹을 학습·검증·테스트에 배정한다.
- 기본 그룹 분할은 엄격한 보간이 아니라 `격자 내부 상태 일반화 평가`로 표현한다.
- 압축기 holdout은 `kMc` 0.950~0.954/0.955~0.959/0.960~1.000을 각각 테스트/검증/학습에 배정한다.
- 터빈 holdout은 `kMt` 0.975~0.979/0.980~0.984/0.985~1.000을 각각 테스트/검증/학습에 배정한다.
- 두 holdout은 계수 값이 낮은, 즉 열화가 더 심한 방향의 외삽 강건성 평가다.
- 강건성 holdout의 대상 열화 계수는 MAE와 RMSE를 주지표로 사용하고 R²는 좁은 범위의 분산에 민감한 보조 지표로 해석한다.
- 분할은 NumPy `default_rng`와 중앙 seed를 사용하며 분할별로 난수 생성기를 독립 생성한다.
- 분할 인덱스는 little-endian int64 바이트의 SHA-256으로 고정한다.
- EDA 그림은 `eda` 의존성 그룹의 Matplotlib Agg 백엔드로 생성하며, 집계 CSV와 핵심 PNG만 커밋 대상으로 한다.
- Linux CI는 원본 데이터를 다운로드하지 않고 합성 격자로 문서화된 분할 해시 12개를 검증한다.
- M2 모델 후보는 평균 예측 Dummy, 원시 입력 Ridge, 속도 중심화 Ridge와 원시 입력 Random Forest로 구성한다.
- 속도 중심화 Ridge는 `v`를 입력에 유지하고 나머지 11개 센서만 학습 데이터의 속도별 평균으로 중심화한다.
- M2 최종 모델은 기본 상태 그룹 validation의 `kMc`, `kMt` NRMSE 평균이 가장 낮은 후보로 선택한다.
- NRMSE는 `kMc` RMSE를 0.050, `kMt` RMSE를 0.025로 나눠 계산하며 반올림 전 값으로 비교한다.
- 선택 점수가 같으면 두 대상 중 더 큰 NRMSE, 사전 정의한 모델 복잡도 순서로 결정하고 Dummy는 선택 대상에서 제외한다.
- 행 랜덤과 압축기·터빈 holdout validation은 모델 선택에 사용하지 않고 비교·외삽 강건성 진단으로 보고한다.
- holdout validation은 선택 진단이고 holdout test는 고정된 최종 모델의 최종 강건성 평가로 구분한다.
- 최종 선택 결과를 고정한 뒤 네 시나리오 test를 한 번 평가하며 test 결과를 근거로 같은 M2 실험에서 모델을 재조정하지 않는다.
- 모델과 행 단위 예측은 `artifacts/modeling/`에 저장하고 집계 지표와 핵심 그림만 `reports/modeling/`에 커밋한다.
- M2 최종 기준 모델은 `random_forest_raw_leaf_1_features_1p0`이며 tree 300개, `min_samples_leaf=1`, `max_features=1.0`, seed 42를 사용한다.
- 최종 모델의 상태 그룹 validation 평균 NRMSE는 0.021853이며 test R²는 `kMc` 0.996720, `kMt` 0.992838이다.
- 압축기 holdout test `kMc` NRMSE는 0.209780, 터빈 holdout test `kMt` NRMSE는 0.366661로 심한 열화 방향 외삽 성능이 크게 저하된다.
- holdout 대상의 건강 방향 bias는 압축기 `kMc` +0.009668, 터빈 `kMt` +0.008830이며 M4 경보 정책의 핵심 제한사항으로 다룬다.
- 로컬 기준 모델 joblib은 `artifacts/modeling/`에만 저장하고 Git에 커밋하지 않는다.
- M3의 목적은 소형 MLP와 선형 잔차 신경망으로 Random Forest의 격자 내부 성능과 심한 열화 방향 외삽 포화 한계를 같은 조건에서 비교하는 것이다.
- M3 후보는 원시 입력 MLP, 속도 중심화 MLP, 속도 중심화 선형 잔차 MLP의 세 구조와 `(64, 32)`, `(128, 64)` 은닉층 조합으로 구성한 6개다.
- 모든 M3 후보는 ReLU 은닉층과 선형 출력층을 사용하며 sigmoid와 예측값 clipping은 적용하지 않는다.
- 선형 잔차 MLP는 표준화 입력의 affine 경로와 MLP 잔차를 더하고 잔차 출력층을 0으로 초기화한다.
- 속도 중심화 후보는 `v`를 유지하고 나머지 11개 센서를 train의 속도별 평균으로 중심화한 뒤 12개 입력 전체를 표준화한다.
- M3 입력 scaler와 target scaler는 각 시나리오의 train에만 fit하며 target은 원 단위로 역변환한 뒤 평가한다.
- M3 학습은 `float32`, 표준화 target MSE, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch 256을 사용한다.
- 최대 500 epoch, 최소 50 epoch, patience 40과 validation 평균 NRMSE `min_delta=1e-5`로 early stopping하고 best checkpoint를 복원한다.
- early stopping과 best checkpoint 복원은 validation 지표가 인위적으로 악화되는 합성 테스트로 별도 검증한다.
- M3 후보 선택은 기본 seed 42에서 파생한 42·43·44 세 seed의 상태 그룹 validation 대상별 NRMSE 평균만 사용한다.
- M3 동점은 평균 NRMSE, 최악 대상 평균 NRMSE, 더 적은 trainable parameter, 일반 MLP·속도 중심화 MLP·선형 잔차 MLP 순서와 candidate ID로 결정한다.
- 행 랜덤 validation은 고정 후보의 early stopping에만 사용하고 두 holdout validation은 early stopping과 외삽 강건성 진단에 사용하되, 세 validation 모두 M3 후보 선택에는 사용하지 않는다.
- 선택된 신경망 구조의 최종 평가 seed는 42로 고정하고 네 시나리오 test를 한 번 평가한 뒤 결과를 근거로 재조정하지 않는다.
- M3 상태 그룹·압축기·터빈 최종 평가는 selection 단계의 seed 42 checkpoint를 재사용하고 행 랜덤 모델만 최종 평가 단계에서 새로 학습한다.
- M3 checkpoint 재사용은 train으로만 학습된 모델을 test에서 처음 평가하는 절차이며, 네 시나리오를 다시 학습한 M2와의 차이를 실험 기록과 모델 카드에 명시한다.
- M3 공식 실험은 CPU에서 수행하고 MPS는 선택 경로로만 제공하며 사용할 수 없거나 결정적 실행이 불가능하면 CPU로 fallback한다.
- M3 재현성 설정은 Python·NumPy·PyTorch seed, 결정적 알고리즘, DataLoader worker 0과 후보별 독립 초기화를 포함한다.
- PyTorch는 `>=2.14,<2.15` 범위를 별도 `modeling` 의존성 그룹으로 관리하고 Linux CI에서는 명시적 CPU wheel index를 사용하며 macOS에서는 MPS를 포함한 기본 wheel을 사용한다.
- M3 checkpoint·manifest·행 단위 예측은 `artifacts/modeling/m3/`에 저장하고 집계 CSV와 핵심 그림만 `reports/modeling/m3/`에 커밋한다.
- M3의 외삽 보완 여부는 두 holdout 대상의 NRMSE와 절대 bias가 모두 Random Forest보다 낮은지로 해석하되 모델 선택 기준으로 사용하지 않는다.
- M2 holdout test를 확인한 뒤 M3 구조를 설계했으므로 M3 holdout 결과를 완전히 미관측인 독립 test가 아닌 사전 고정한 벤치마크의 탐색적 비교로 표현한다.
- M3 최종 모델은 `linear_residual_mlp_speed_centered_hidden_128_64`이며 trainable parameter는 10,076개다.
- M3 선택 모델의 세 seed 상태 그룹 validation 평균 NRMSE는 `kMc` 0.002866, `kMt` 0.004659이다.
- M3 상태 그룹 test NRMSE는 `kMc` 0.002793, `kMt` 0.005022이며 M2보다 각각 82.8%, 78.8% 낮다.
- M3 압축기 holdout `kMc` NRMSE는 0.020581, 터빈 holdout `kMt` NRMSE는 0.055401이며 M2보다 각각 90.2%, 84.9% 낮다.
- M3 holdout 대상 bias는 압축기 `kMc` +0.000695, 터빈 `kMt` +0.000996으로 감소했지만 건강 방향 과대 추정은 남아 있다.
- M4는 회귀 예측값 기반 경보 정책을 핵심으로 한다.
- M4 열화도는 `kMc`에 `(1-kMc)/0.050`, `kMt`에 `(1-kMt)/0.025`를 사용하고 두 값의 최댓값을 전체 열화도로 사용한다.
- 회귀 예측의 공식 범위 이탈을 감추지 않기 위해 M4 열화도는 clipping하지 않는다.
- M4 상태는 열화도 0.5 미만 `normal`, 0.5 이상 0.8 미만 `watch`, 0.8 이상 `alert`로 정의한다.
- 주 경보 임계값 0.8은 `kMc ≤ 0.960`, `kMt ≤ 0.980`에 대응하는 PoC 정책 시나리오이며 공식 고장 임계값이 아니다.
- 임계값 0.5·0.6·0.7·0.8·0.9의 민감도를 비교하되 test 결과로 임계값을 조정하지 않는다.
- M4는 M2·M3에서 저장한 test 예측을 SHA-256 검증 후 재사용하고 모델을 재학습하거나 test 예측을 다시 생성하지 않는다.
- 경보 평가 채널은 `kMc`, `kMt`, 두 상태 중 하나라도 경보인 `any`로 구성한다.
- 기본 경보 지표는 TP·FP·TN·FN, Precision, Recall, F1, FPR, miss rate, Average Precision 기반 PR-AUC와 reference prevalence다.
- reference가 단일 클래스인 경우 PR-AUC를 계산하지 않고 클래스 구성에 따라 정의 가능한 지표만 보고하며 나머지는 `NA`와 사유를 기록한다.
- 전체 양성에서는 Recall·FN·miss rate만, 전체 음성에서는 FP·TN·FPR만 경보 성능 지표로 해석한다.
- 고정 오경보율 cutoff는 상태 그룹 validation에서 목표 FPR 1%·5% 이하 조건으로 정하고 test·holdout에 그대로 적용한다.
- holdout에 정상 표본이 없으면 고정 cutoff Recall은 보고하되 realized FPR은 `NA`로 기록한다.
- M4 reference alert는 simulator 상태 계수에서 파생한 정책 상태이며 실제 고장 라벨이나 실제 고장 탐지 결과가 아니다.
- M4는 기존 M2·M3 test를 활용한 downstream 분석이므로 새로운 독립 test라고 주장하지 않는다.
- 타임스탬프가 없어 지속 시간·debounce·hysteresis 경보 정책은 평가하지 않는다.
- 실제 정상 운항 라벨과 정상 모집단의 근거가 없어 M4에서 Isolation Forest와 Autoencoder는 수행하지 않는다.
- 실제 고장 라벨과 공식 경보 임계값이 없다는 한계를 명시한다.
- M4 주 임계값 0.8의 상태 그룹 `any` Recall/FPR은 M2 0.924501/0.000918, M3 0.955840/0이다.
- 압축기 holdout `kMc` Recall은 M2 0, M3 1.0이고 터빈 holdout `kMt` Recall은 M2 0, M3 0.931590이다.
- 압축기·터빈 holdout의 대상 reference는 모두 양성이므로 Precision·F1·FPR·PR-AUC는 `NA`로 기록한다.
- 상태 그룹 validation 목표 FPR 1% cutoff를 적용한 상태 그룹 test `any` Recall/FPR은 M2 0.981481/0.004591, M3 0.998575/0.007346이다.
- 목표 FPR은 validation 제약이며 test realized FPR을 보장하지 않는다. 목표 5%에서 M2 상태 그룹 test `any` FPR은 0.058770이었다.
- M5 1차 배포 모델은 `m3-linear-residual-mlp-v1`로 확정한다. M2 Random Forest는 비교 기준으로 유지한다.
- M3 채택 근거는 M2보다 낮은 상태 그룹·holdout 오차, holdout 경보 누락 개선, 46,325 byte checkpoint와 macOS 예비 측정의 낮은 peak RSS·추론 지연이다.
- macOS 예비 측정은 모델 결정의 보조 근거이며 최종 운영 수치는 고정 Docker/Linux 조건에서 다시 측정한다.
- M5 API는 상태 추정과 경보 평가를 분리한 `GET /health`, `GET /model/info`, `POST /v1/condition/predict`, `POST /v1/condition/batch`, `POST /v1/alert/evaluate`로 구성한다.
- 상태 추정 API는 원본 16개가 아니라 `v`, `GTT`, `GTn`, `GGn`, `Ts`, `T48`, `T2`, `P48`, `P2`, `Pexh`, `TIC`, `mf` 12개 이름을 입력받는다.
- `v`는 3~27 knots의 3 knots 간격 9개 값만 허용하고, 나머지 11개 센서는 기본 상태 그룹 train 8,352행의 min/max를 양 끝 포함 범위로 사용한다.
- 전체 11,934행과 네 시나리오 validation·test 8개 역할에서 위 11개 센서 범위를 벗어난 행은 0개다.
- 상태 추정 입력은 추가 필드와 NaN·무한대를 거부하고 범위 밖 값을 clipping하지 않고 `422`로 반환한다.
- 배치 요청은 1~100건으로 강제하고 입력 순서와 같은 순서로 예측을 반환한다.
- 경보 평가는 유한한 `kMc`, `kMt`를 입력받고 공식 계수 범위 이탈을 clipping하거나 거부하지 않은 채 M4 열화도·상태 정책을 적용한다.
- validation 오류는 `VALIDATION_ERROR` 코드, 공통 메시지와 정제된 세부 정보로 구성한 `error` envelope로 반환하며 요청 본문과 내부 경로를 노출하지 않는다.
- `config/deployment_model.json`은 모델·정책 버전, checkpoint SHA-256, 입력 순서·범위, target 순서와 경보 임계값을 기록하며 모델·계약 불일치 시 시작 단계에서 실패한다.
- 원본 데이터는 Git에 커밋하지 않는다.
- 기준 모델 결과를 확보한 뒤 PyTorch 비교 모델을 구현했다.
- RAG와 프론트엔드는 MVP에서 제외한다.

## 미확정 사항

- 현재 없음

## 마지막 검증

- 2026-09-22: 기본 상태 그룹 train 기준 11개 연속 센서 범위를 전체 데이터와 네 시나리오 validation·test에 적용해 범위 이탈 0행 확인
- 2026-09-22: macOS 예비 측정에서 M2/M3 peak RSS 591.9/294.2MB, 단건 평균 지연시간 5.572/0.164ms 확인
- 2026-09-22: commit `8b8d00b` 기준에서 M2·M3 artifact SHA-256 검증 후 재학습 없이 M4 공식 평가 완료
- 2026-09-22: M4 주 임계값 0.8에서 상태 그룹과 압축기·터빈 holdout 경보 지표 및 단일 클래스 `NA` 처리 확인
- 2026-09-22: 결정적 gzip으로 재생성한 M4 행 단위 정책 예측 SHA-256 `d21156993e179c5f0968aeada29bf8d56e62eb8c2220e93246be25ec1c2bdad8`
- 2026-09-22: M4 구현 후 `pytest -q` 110개 테스트 통과, Ruff·포맷·`uv lock --check`·`git diff --check` 통과
- 2026-09-22: commit `ef70ee1`의 clean 상태에서 M3 6개 후보 × 3개 시나리오 × 3개 seed validation 선택 실험 완료
- 2026-09-22: 고정된 M3 seed 42 checkpoint와 행 랜덤 신규 fit으로 네 시나리오 test 1회 평가 완료
- 2026-09-22: M3 상태 그룹 test `kMc`/`kMt` R² 0.999903/0.999678, 두 holdout 대상 NRMSE와 건강 방향 bias 감소 확인
- 2026-09-22: M3 상태 그룹 checkpoint 재로드 예측과 저장 CSV 최대 절대 차이 `1.11e-16`
- 2026-09-22: M3 구현 후 `pytest -q` 90개 테스트 통과, Ruff·포맷·`uv lock --check`·`git diff --check` 통과
- 2026-09-22: commit `129ae3c`의 clean 상태에서 validation 전용 17개 후보 선택 실험 완료
- 2026-09-22: 고정 Random Forest로 행 랜덤·상태 그룹·압축기·터빈 holdout test 1회 평가 완료
- 2026-09-22: 상태 그룹 test `kMc`/`kMt` R² 0.996720/0.992838, holdout 외삽 포화와 건강 방향 bias 확인
- 2026-09-22: joblib 재로드 예측과 저장 CSV 최대 절대 차이 `1.11e-16`
- 2026-09-22: M2 구현 후 `pytest -q` 63개 테스트 통과, Ruff·포맷·`uv lock --check` 통과
- 2026-09-21: GitHub Actions CI 실행 #1(`5948fd3`) 성공, Ubuntu 24.04.5 LTS·Linux/X64(`x86_64`)·Runner Image `20260907.300.1`
- 2026-09-21: CI에서 uv 0.12.17·Python 3.13.15로 lock·Ruff·포맷 검사 통과, 39개 테스트 성공·원본 데이터 통합 테스트 3개 skip
- 2026-09-21: Linux/aarch64 컨테이너(`5948fd3`, Debian 13, Python 3.13.15, uv 0.12.17)에서 원본 데이터 포함 42개 테스트와 Ruff·포맷 검사 통과
- 2026-09-21: Linux/aarch64 사전 검증(`main@ada158e`, Python 3.13.15, uv 0.12.17)에서 원본 데이터 포함 41개 테스트와 Ruff·포맷 검사 통과
- 2026-09-21: macOS에서 `uv lock --check`, Ruff·포맷 검사 및 원본 데이터 포함 42개 테스트 통과
- 2026-09-21: `ruff check .` 통과
- 2026-09-21: `ruff format --check .` 통과, 24개 파일 형식 확인
- 2026-09-21: `pytest -q` 통과, 실제 원본·분할 해시·EDA 관찰 통합 테스트 포함 41개 테스트 성공
- 2026-09-21: `uv lock --check` 통과, EDA 그룹 포함 20개 패키지 해석
- 2026-09-21: 네 분할 시나리오의 행 수·경계·재현성 및 12개 SHA-256 검증
- 2026-09-21: 집계·상관 CSV 4개와 핵심 EDA PNG 3개 생성 및 육안 확인
- 2026-09-21: `ruff check .` 통과
- 2026-09-21: `ruff format --check .` 통과, 10개 파일 형식 확인(Ruff 제외 경로 수정 전)
- 2026-09-21: `pytest -q` 통과, 실제 원본 통합 테스트 포함 24개 테스트 성공
- 2026-09-21: `uv lock --check` 통과, 13개 패키지 해석
- 2026-09-21: 실제 UCI 원본 파일 3개의 SHA-256 및 11,934행·18열 구조 검증
- 2026-09-21: 결측값 0개, 운항 속도 9개, `kMc` 51개, `kMt` 26개 및 전체 격자 조합 검증
- 2026-09-20: Python 3.13.13과 uv 0.12.17로 `uv.lock` 생성 및 환경 동기화
- 2026-09-20: `ruff check .` 통과
- 2026-09-20: `ruff format --check .` 통과, 7개 파일 형식 확인
- 2026-09-20: `pytest -q` 통과, 1개 테스트 성공
- 2026-09-18: 기본 브랜치 `main`인 로컬 Git 저장소 초기화 확인
- 2026-09-18: 데이터 정의, 분할 원칙 및 경보 정책의 문서 간 정합성 검토
