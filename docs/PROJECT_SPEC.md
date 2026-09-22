# Maritime CBM 프로젝트 명세

## 1. 프로젝트명

Maritime CBM: 선박 가스터빈 열화 상태 추정 및 경보 API

## 2. 문제 정의

시뮬레이션 기반 다변량 선박 가스터빈 센서 데이터에서 압축기·터빈 열화 상태를
추정하고, 제한된 조건에서 경보 정책의 가능성을 검토한다.

이 프로젝트는 실제 고장을 확정적으로 진단하는 의료·안전 시스템이 아니라
공개 시뮬레이션 데이터를 사용한 상태 기반 유지보수 PoC이다. 실제 고장 라벨과
공식 경보 임계값이 없으므로 결과를 실제 고장 판정이나 미래 고장 예측으로 해석하지 않는다.

## 3. 데이터

사용 데이터:

- UCI `Condition Based Maintenance of Naval Propulsion Plants` 릴리스
- 타임스탬프가 없는 정상상태 선박 가스터빈 시뮬레이션 데이터
- 공식 설명 기준 11,934행, 16개 운항·센서 입력, 2개 정답

입력 변수:

1. Lever position (`lp`)
2. Ship speed (`v`)
3. Gas Turbine shaft torque (`GTT`)
4. Gas Turbine rate of revolutions (`GTn`)
5. Gas Generator rate of revolutions (`GGn`)
6. Starboard Propeller Torque (`Ts`)
7. Port Propeller Torque (`Tp`)
8. High Pressure Turbine exit temperature (`T48`)
9. Gas Turbine Compressor inlet air temperature (`T1`)
10. Gas Turbine Compressor outlet air temperature (`T2`)
11. High Pressure Turbine exit pressure (`P48`)
12. Gas Turbine Compressor inlet air pressure (`P1`)
13. Gas Turbine Compressor outlet air pressure (`P2`)
14. Gas Turbine exhaust gas pressure (`Pexh`)
15. Turbine Injection Control (`TIC`)
16. Fuel flow (`mf`)

예측 대상:

- `kMc`: 가스터빈 압축기 열화 계수
- `kMt`: 가스터빈 터빈 열화 계수

`kMc`, `kMt`는 실제 고장 발생을 관측해 부여한 라벨이 아니라 시뮬레이터에서
센서값 생성에 사용한 열화 상태 계수다. 모델의 과제는 정상상태 센서값으로부터 이 열화
계수를 역으로 추정하는 것이다. 따라서 결과를 실제 고장 발생이나 시간에 따른 열화 진행을
예측하는 것으로 해석하지 않는다.

공식 설명 기준 격자 구조:

- 운항 속도: 3~27 knots, 3 knots 간격, 예상 고유값 9개
- `kMc`: 0.950~1.000, 0.001 간격, 예상 고유값 51개
- `kMt`: 0.975~1.000, 0.001 간격, 예상 고유값 26개
- 예상 행 수: 9 × 51 × 26 = 11,934

실제 파일을 로드한 뒤 11,934행과 총 18개 변수 여부, 입력·정답 순서 및 격자 구조를
공식 README와 대조한다. 부동소수점 값은 그대로 equality 비교하지 않고 허용오차 또는
반올림 후 검증한다. 공식 설명과 불일치하면 데이터 처리를 중단하고 원인을 먼저 기록한다.

## 4. MVP 범위

### M0. 프로젝트 기반

- 기본 저장소 및 `src` 패키지 구조
- Python 3.13과 uv 기반 개발 환경 및 의존성 잠금
- Ruff와 pytest 설정
- 설정값 중앙 관리와 표준 `logging` 사용 원칙 확정
- README 초안과 코드용 MIT License
- 데이터 출처, CC BY 4.0 라이선스, 인용 및 다운로드 안내

### M1. 데이터 파이프라인

- 데이터 로드
- 행·열 수, 변수 순서, 스키마 및 결측값 검증
- `docs/DATASET.md`에 원본 파일 해시와 실제 검증 결과를 추가하고 데이터 카드로 관리
- 입력·정답 분리
- 기본 EDA
- 상수 `T1`, `P1`, 중복 `Tp`, 운항 속도 `v`의 1:1 proxy인 `lp`를 제외한 12개 모델 입력 확정
- 행 단위 랜덤 분할을 비교용 기준으로 구성
- 경계 상태 그룹을 학습에 고정한 `(kMc, kMt)` 상태 조합 단위 그룹 분할을 기본 평가로 구성
- 압축기와 터빈의 연속된 저계수 구간 holdout을 심한 열화 방향 외삽 강건성 평가로 구성
- 분할 행 인덱스의 SHA-256을 데이터 카드와 통합 테스트에 고정
- 집계 CSV와 핵심 그림만 `reports/eda/`에 저장
- 데이터 검증 테스트가 준비되면 Linux CI에서 Ruff와 pytest 실행

### M2. 기준 모델

- scikit-learn `DummyRegressor`, Ridge와 Random Forest 기준 모델
- `kMc`, `kMt` 다중 출력 회귀
- 속도별 학습 평균으로 11개 센서를 중심화하고 `v`를 유지하는 Ridge 후보 비교
- 기본 상태 그룹 validation의 대상별 NRMSE 평균으로 모델과 하이퍼파라미터 선택
- NRMSE 분모는 공식 계수 범위인 `kMc` 0.050, `kMt` 0.025로 고정
- 행 랜덤과 압축기·터빈 holdout validation은 선택에 사용하지 않고 비교·강건성 진단으로 보고
- 선택 결과를 고정한 뒤 네 시나리오 test를 한 번 평가하고 결과 확인 후 재조정하지 않음
- 기본 평가는 대상별 MAE, RMSE, R²를 함께 보고
- 연속 열화 구간 holdout의 대상 계수는 MAE와 RMSE를 주지표, R²를 보조 지표로 해석
- 예측값을 공식 계수 범위로 clipping하지 않고 범위 이탈과 외삽 포화를 오류 분석에 포함
- 평가 리포트
- 오류 사례 분석
- 최종 기준 모델 선정 후 `docs/MODEL_CARD.md` 생성

### M3. PyTorch 모델

- 소형 MLP와 선형 잔차 경로를 가진 신경망을 M2와 동일한 입력·분할·지표로 비교
- Random Forest의 격자 내부 성능과 심한 열화 방향 외삽 포화 한계를 보완할 가능성 검토
- 원시 입력 MLP, 속도 중심화 MLP, 속도 중심화 선형 잔차 MLP의 세 구조 비교
- 각 구조에서 `(64, 32)`, `(128, 64)` 은닉층을 사용한 총 6개 후보 평가
- 모든 후보는 ReLU 은닉층과 선형 출력층을 사용하고 예측값을 공식 계수 범위로 clipping하지 않음
- 속도 중심화 후보는 `v`를 유지하고 나머지 11개 센서만 train의 속도별 평균으로 중심화한 뒤 12개 입력을 표준화
- 입력 scaler와 target scaler는 각 분할의 train에만 fit
- AdamW, 표준화 target MSE, batch 256, 최대 500 epoch와 validation NRMSE 기반 early stopping 사용
- 기본 seed 42에서 파생한 42·43·44 세 seed의 상태 그룹 validation 평균 NRMSE로 신경망 후보 선택
- 동점은 최악 대상 평균 NRMSE, trainable parameter 수, 사전 정의한 구조 순서와 candidate ID로 결정
- 행 랜덤 validation은 고정 후보의 early stopping에만 사용하고 압축기·터빈 holdout validation은
  early stopping과 외삽 진단에 사용하되, 세 validation 모두 후보 선택에는 사용하지 않음
- 선택 구조와 seed 42를 고정한 뒤 네 시나리오 test를 한 번 평가하고 결과 확인 후 재조정하지 않음
- 상태 그룹·압축기·터빈은 selection 단계의 seed 42 checkpoint를 재사용하고 행 랜덤만 최종 평가 단계에서 학습
- CPU와 `float32`를 공식 실험 기준으로 사용하고 MPS는 선택 실행 경로와 CPU fallback만 제공
- 학습 곡선, 과적합, 성능·bias·외삽 범위, 학습·추론 시간, parameter 수와 artifact 크기 비교
- checkpoint와 행 단위 예측은 `artifacts/modeling/m3/`, 집계 지표와 핵심 그림은 `reports/modeling/m3/`에 저장
- M2 holdout 결과를 본 뒤 구조를 설계했으므로 M3 holdout을 완전히 미관측인 독립 test가 아닌 고정 벤치마크의 탐색적 비교로 해석
- 비교 결과와 모델별 한계를 `docs/MODEL_CARD.md`에 갱신

### M4. 경보 정책

- `kMc`, `kMt` 회귀 예측값을 각 simulator 범위로 정규화한 열화도와 전체 최대 열화도 계산
- 예측 범위 이탈을 오류 분석에 유지하기 위해 열화도를 clipping하지 않음
- 열화도 0.5 미만은 `normal`, 0.5 이상 0.8 미만은 `watch`, 0.8 이상은 `alert`로 정의
- 주 경보 임계값 0.8은 `kMc ≤ 0.960`, `kMt ≤ 0.980`에 대응하며 공식 고장 기준이 아닌
  simulator 열화 범위 하위 20%의 PoC 정책 시나리오로 해석
- 열화도 임계값 0.5·0.6·0.7·0.8·0.9를 사전 고정해 민감도를 분석하고 test 결과로 조정하지 않음
- M2 Random Forest와 M3 선형 잔차 MLP의 기존 test 예측과 evaluation manifest SHA-256을
  재사용하며 모델을 다시 학습하거나 test 예측을 재조정하지 않음
- `kMc`, `kMt`, 두 상태 중 하나라도 경보인 `any` 채널별 TP·FP·TN·FN, Precision,
  Recall, F1, FPR, miss rate, Average Precision 기반 PR-AUC와 reference prevalence 보고
- reference가 단일 클래스이면 클래스 판별력을 나타낼 수 없는 지표를 `NA`와 명시적 사유로 기록
  - 전체 양성: PR-AUC·Precision·F1·FPR은 `NA`, Recall·FN·miss rate만 보고
  - 전체 음성: PR-AUC·Precision·Recall·F1은 `NA`, FP·TN·FPR만 보고
  - scikit-learn 지표 호출 전에 고유 클래스 수를 검사해 인공적인 PR-AUC 1.0·0.0을 차단
- 상태 그룹 validation에서 모델·채널별 목표 FPR 1%·5% 이하인 cutoff 중 Recall이 가장 높은
  값을 선택하고, 동점은 낮은 FPR·높은 Precision·높은 cutoff 순으로 결정
- validation에서 고정한 cutoff를 test와 holdout에 그대로 적용하고, 정상 표본이 없는
  holdout에서는 결과 Recall을 보고하되 realized FPR은 `NA`로 기록
- M4 reference alert는 실제 고장 라벨이 아니라 simulator 열화 상태 계수에서 파생한 정책 상태이며,
  결과를 실제 고장 탐지 성능이나 새로운 독립 test로 해석하지 않음
- 타임스탬프가 없으므로 지속 시간·debounce·hysteresis 정책은 평가하지 않음
- 실제 정상 운항 라벨과 정상 모집단 근거가 없어 Isolation Forest와 Autoencoder 실험은 수행하지 않음
- 경보 정책, 임계값 근거, 민감도와 해석 제한을 `docs/MODEL_CARD.md`에 갱신

### M5. 서비스화

- M3 `m3-linear-residual-mlp-v1`을 1차 배포 모델로 사용하는 FastAPI 추론 API
- 배포 모델·전처리·입력 범위·경보 정책 계약을 `config/deployment_model.json`에서 버전 관리
- 계약에는 모델·정책 버전, checkpoint SHA-256, 12개 입력 순서, 허용 운항 속도,
  연속 센서 범위, target 순서와 경보 임계값을 기록하고 불일치하면 서비스 시작을 중단
- 상태 추정과 경보 평가 API를 분리하고 단건·배치 상태 추정 제공
- Pydantic 공통 요청 설정으로 추가 필드와 NaN·무한대를 거부
- 운항 속도 `v`는 3~27 knots의 3 knots 간격 9개 값만 허용하고 나머지 11개 센서는
  상태 그룹 train에서 관측한 min/max 범위를 양 끝 포함으로 검증
- 범위 밖 입력을 clipping하지 않고 `422`로 거부하며 배치는 1~100건으로 제한
- FastAPI 기본 validation 오류를 민감한 요청 본문과 내부 경로를 제외한 공통 오류 형식으로 변환
- 단위·통합 테스트
- Docker 실행
- Docker smoke test
- 확정된 구성 기준 아키텍처와 재현 절차 시각화
- 고정된 Docker/Linux 조건에서 API 지연시간, cold start와 메모리 사용량 기록
- README 실행 예시, 실험 결과 및 포트폴리오 설명 최종 보완
- 배포 모델 버전, API 입력·출력과 제한사항을 `docs/MODEL_CARD.md`에 최종 반영

## 5. 비범위

지원서 제출 전에는 다음 기능을 우선 구현하지 않는다.

- RAG 또는 LLM Agent 결합
- 복잡한 프론트엔드
- 클라우드 배포
- 실시간 Kafka 파이프라인
- 멀티 에이전트
- 시계열 예측, 미래 고장 예측 및 잔여수명 예측
- 근거 없는 LSTM 시계열 모델

## 6. API 계약

| endpoint | 요청 | 응답 최소 요건 |
|---|---|---|
| `GET /health` | 없음 | 서비스 상태와 모델 로드 상태 |
| `GET /model/info` | 없음 | 모델·정책 버전, 입력 순서·범위와 경보 기준 |
| `POST /v1/condition/predict` | 이름이 명시된 12개 센서값 1건 | 모델 버전과 `kMc`, `kMt` 예측값 |
| `POST /v1/condition/batch` | 같은 입력 1~100건 | 입력 순서를 보존한 모델 버전과 예측 목록 |
| `POST /v1/alert/evaluate` | 유한한 `kMc`, `kMt` 계수 | 정책 버전, 대상별·전체 열화도와 상태, 판단 기준 |

상태 추정과 경보 평가를 결합하지 않는다. `/v1/condition/*`는 열화 상태 계수만 추정하고,
`/v1/alert/evaluate`는 전달받은 계수에 M4 정책을 적용한다. 경보 입력 계수는 회귀 모델의
공식 범위 이탈을 보존하기 위해 clipping하거나 공식 계수 범위로 제한하지 않는다.

상태 추정 입력 순서는 다음 12개 이름으로 고정한다.

```text
v, GTT, GTn, GGn, Ts, T48, T2, P48, P2, Pexh, TIC, mf
```

요청 모델은 `ConfigDict(extra="forbid", allow_inf_nan=False)`를 공통 적용한다. validation
실패는 `422`와 다음 envelope로 반환하며 `details`에는 정제한 위치·유형·메시지만 포함한다.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "입력값이 API 계약을 만족하지 않는다.",
    "details": []
  }
}
```

## 7. 완료 기준

- 입력과 정답 사이의 데이터 누수가 없다.
- 실제 데이터 구조, 변수 순서 및 격자 구조가 공식 README와 대조 검증된다.
- `(kMc, kMt)` 상태 조합 단위 그룹 분할 결과가 주요 평가로 보고된다.
- 압축기·터빈 저계수 구간 holdout 강건성 평가가 보고된다.
- 기준 모델과 PyTorch 모델이 같은 조건으로 비교된다.
- 결과가 실제 실행 로그로 재현된다.
- 경보 정책의 임계값 근거와 실제 고장 라벨 부재 한계가 명시된다.
- API 입력 검증과 오류 테스트가 존재한다.
- 새 환경에서 README만 보고 실행할 수 있다.
- 데이터 한계와 모델 한계가 문서에 명시된다.
