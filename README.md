# Maritime CBM

시뮬레이션 기반 다변량 선박 가스터빈 센서 데이터에서 압축기·터빈 열화 상태를
추정하고, 제한된 조건에서 경보 정책의 가능성을 검토하는 PoC이다.

## 프로젝트 범위

- UCI `Condition Based Maintenance of Naval Propulsion Plants` 데이터 사용
- 정상상태 센서값으로부터 `kMc`, `kMt` 열화 상태 계수를 추정하는 다중 출력 회귀
- 회귀 예측값을 이용한 제한적 경보 정책 검토
- FastAPI 추론 API와 Docker 실행 환경 제공

이 데이터에는 타임스탬프와 실제 고장 라벨이 없다. 따라서 이 프로젝트를 시계열 예측,
미래 고장 예측, 잔여수명 예측 또는 실제 선박 고장진단으로 해석하지 않는다.

## 현재 상태

M0 프로젝트 기반 구성을 완료하고 M1 데이터 파이프라인을 준비하고 있다. 실험 결과와
성능 지표는 아직 없다.

세부 범위와 진행 상황은 다음 문서에서 관리한다.

- [프로젝트 명세](docs/PROJECT_SPEC.md)
- [현재 진행 상황](docs/CURRENT_STAGE.md)
- [데이터셋 안내](docs/DATASET.md)
- [실험 기록](docs/EXPERIMENT_LOG.md)

## 개발 환경

- Python 3.13
- uv
- Ruff
- pytest

환경을 구성하고 기본 검증을 실행한다.

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## 디렉터리 구조

```text
.
├── data/                  # 로컬 데이터, Git 제외
│   ├── raw/
│   └── processed/
├── docs/                  # 명세, 데이터 및 실험 문서
├── src/maritime_cbm/      # 애플리케이션 패키지
├── tests/                 # 자동화 테스트
├── pyproject.toml         # 프로젝트 및 도구 설정
└── uv.lock                # 재현 가능한 의존성 잠금 파일
```

원본 데이터는 저장소에 포함하지 않는다. 다운로드와 인용 방법은
[데이터셋 안내](docs/DATASET.md)를 따른다.

## 라이선스

프로젝트 코드는 [MIT License](LICENSE)를 따른다. UCI 데이터셋은 별도의
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따르며,
프로젝트의 MIT License가 데이터셋에 적용되지는 않는다.
