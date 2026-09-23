# Artifact 배포 정책

## 목적

이 문서는 Git에 포함하지 않는 모델·데이터 산출물 중 `v0.1.0`에서 공개 배포할 범위와
무결성, 라이선스 및 신뢰 경계를 정의한다. 소스 코드 release와 모델 checkpoint의 버전을
연결하되 원본 데이터나 중간 실험 산출물을 함께 배포하지 않는다.

## `v0.1.0` 배포 대상

GitHub Release에는 다음 두 파일만 첨부한다.

| 파일 | 역할 |
|---|---|
| `maritime-cbm-m3-linear-residual-mlp-v1.pt` | M5 1차 배포용 M3 상태 그룹 seed 42 checkpoint |
| `SHA256SUMS` | Release asset의 SHA-256 검증값 |

Checkpoint 계약은 다음과 같다.

| 항목 | 값 |
|---|---|
| 소프트웨어 release | `v0.1.0` |
| 모델 버전 | `m3-linear-residual-mlp-v1` |
| 후보 | `linear_residual_mlp_speed_centered_hidden_128_64` |
| 시나리오·seed | `state_group`, 42 |
| PyTorch 기본 버전 | `2.14.0` |
| 크기 | 46,325 byte |
| SHA-256 | `cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472` |
| 런타임 계약 | `config/deployment_model.json` |

Release asset 이름은 배포 모델 버전을 드러내고, 서비스가 사용하는 로컬 파일명은 기존
계약과 Compose 구성을 유지한다. 이름 변경은 파일 내용과 SHA-256을 바꾸지 않는다.

```text
Release: maritime-cbm-m3-linear-residual-mlp-v1.pt
Local:   artifacts/modeling/m3/checkpoints/state_group_seed_42.pt
```

## 배포하지 않는 항목

다음 파일은 `v0.1.0` Release asset, Git 저장소와 Docker image에 포함하지 않는다.

- UCI 원본 `data.txt`, `Features.txt`, `README.txt`
- M2 `baseline_model.joblib`
- 압축기·터빈 holdout 등 나머지 M3 checkpoint
- M2·M3·M4 행 단위 예측과 evaluation·selection manifest
- 로컬 Docker image와 benchmark 실행 중간 파일

원본 데이터는 [`DATASET.md`](DATASET.md)의 공식 UCI 경로에서 사용자가 직접 내려받는다.
모델을 다시 학습하거나 전체 실험을 재현해야 할 때만 로컬 `data/raw/`와 `artifacts/`를
사용하며 두 경로는 Git에서 제외한다.

## 다운로드와 로컬 배치

`v0.1.0` 발행 후 checkpoint를 서비스 기본 경로에 직접 저장한다.

```bash
mkdir -p artifacts/modeling/m3/checkpoints

curl --fail --location \
  --output artifacts/modeling/m3/checkpoints/state_group_seed_42.pt \
  https://github.com/krapnuyij/maritime-cbm/releases/download/v0.1.0/maritime-cbm-m3-linear-residual-mlp-v1.pt
```

다운로드 직후 다음 명령으로 배포 계약의 SHA-256과 대조한다.

```bash
printf '%s  %s\n' \
  'cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472' \
  'artifacts/modeling/m3/checkpoints/state_group_seed_42.pt' \
  | shasum -a 256 -c -
```

검증이 성공한 뒤에만 로컬 API 또는 Docker Compose를 실행한다.

## Release 생성과 독립 검증

Release staging 디렉터리에서 checkpoint를 최종 asset 이름으로 복사한 뒤 `SHA256SUMS`를
생성한다. `shasum` 출력은 검증 명령이 요구하는 `<해시>  <파일명>` 형식을 사용한다.

```bash
shasum -a 256 maritime-cbm-m3-linear-residual-mlp-v1.pt > SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

Release 발행 후에는 새 임시 디렉터리에 두 asset을 다시 내려받고 동일한 검증을 반복한다.
다운로드 파일의 해시는 `SHA256SUMS`와 `config/deployment_model.json`의
`checkpoint_sha256` 양쪽과 일치해야 한다. 이번 PoC release에는 별도의 artifact attestation
생성 workflow를 추가하지 않는다.

## 로드와 신뢰 경계

서비스는 checkpoint를 로드하기 전에 SHA-256을 검증하고 candidate, seed, 시나리오,
전처리 상태, 입력·target 순서와 PyTorch 기본 버전을 배포 계약과 대조한다. Checkpoint는
`torch.load(..., weights_only=True)`로 읽고 Docker에서는 read-only volume으로 마운트한다.

`weights_only=True`와 해시 검증은 임의 출처의 파일을 신뢰할 수 있게 만드는 보안 경계가
아니다. GitHub 공식 release URL에서 내려받은 파일만 사용하고, 해시가 다르면 로드하거나
기존 모델 버전으로 배포하지 않는다.

## 데이터 출처와 라이선스 경계

- 프로젝트 코드는 저장소의 MIT License를 따른다.
- 학습 데이터는 UCI `Condition Based Maintenance of Naval Propulsion Plants`이며 현재 공식
  페이지의 CC BY 4.0 표기를 따른다.
- Checkpoint는 위 데이터로 학습한 가중치와 train-only 전처리 통계 및 학습 metadata를
  포함하지만 UCI 원본 행이나 원본 파일을 포함하지 않는다.
- 데이터셋명, 저자, DOI `10.24432/C5K31K`, UCI 출처와 CC BY 4.0을 release note에 표시한다.
- 프로젝트의 MIT License가 UCI 원본 데이터의 이용 조건을 대체하지 않는다.
- UCI 공식 CC BY 4.0 표기와 배포 README의 과거 상업적 이용 금지 문구가 충돌하므로,
  상업적 사용 전에는 권리자 또는 UCI에 별도로 확인한다.

이 checkpoint는 공개 시뮬레이션 데이터로 만든 PoC 산출물이다. 실제 선박 고장진단,
미래 고장 예측, 잔여수명 예측이나 운영 안전 보장을 제공하지 않는다.
