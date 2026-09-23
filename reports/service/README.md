# M5 Docker API benchmark

실제 M3 상태 그룹 seed 42 checkpoint를 재학습 없이 FastAPI Docker image에 read-only로
마운트하고, 하나의 고정 순차 요청 프로토콜로 운영 특성을 측정한 결과다.

## 측정 조건

- 측정 시각: 2026-09-23 00:42 KST
- host: macOS 26.5.1 arm64, Docker server 29.5.3
- target: Linux/ARM64, Debian GNU/Linux 13
- runtime: Python 3.13.15, PyTorch 2.14.0+cpu, FastAPI 0.141.1, Uvicorn 0.53.0
- service: Uvicorn worker 1개, concurrency 1, 지속 HTTP/1.1 client의 순차 요청
- protocol: warm-up 50회, cold start 5회, 단건 1,000회, 100건 batch 200회
- checkpoint SHA-256: `cbb56741b2a9209afea71bfdc7b8f0a575b2ece4e0795343170e2c3c086cf472`
- image ID: `sha256:c0405643bca6befec748bc3814eb1befbd57c9109923f6a864b73f80d8f82b7c`

## 결과

| 작업 | 평균 | 중앙값 | p95 | 요청 오류 |
|---|---:|---:|---:|---:|
| cold start | 1,452.024ms | 1,301.285ms | 1,780.138ms | 0/5 |
| 단건 상태 추정 | 1.438ms | 1.277ms | 2.254ms | 0/1,000 |
| 100건 batch 상태 추정 | 3.901ms | 3.855ms | 5.127ms | 0/200 |

- idle process RSS: 352.652MiB
- process peak RSS: 355.934MiB
- 요청 후 Docker cgroup 사용량: 242.9MiB
- `docker image inspect .Size`: 354,694,718 byte, 338.263MiB
- 후속 `docker system df -v`: virtual 1.7GB, shared 201.5MB, unique 1.503GB
- benchmark 전후 checkpoint SHA-256: 동일

process RSS는 container 프로세스의 `/proc/1/status`, cgroup 사용량은 Docker stats에서
측정했다. 계측 범위가 다르므로 두 값을 직접 차감하거나 같은 메모리 지표처럼 비교하지
않는다.

`docker system df -v`의 virtual size는 shared와 unique의 합이다. shared·unique 구분은 같은
로컬 image store에 존재하는 다른 image에 따라 달라질 수 있다. `docker image inspect`
`Size` 필드가 압축 전송 크기를 뜻한다고 단정하지 않고 실제 명령·필드와 측정값을 그대로
기록한다. 계측 정의는 Docker의 [`system df` 문서](https://docs.docker.com/reference/cli/docker/system/df/)와
[`containerd` image store 문서](https://docs.docker.com/engine/storage/containerd/)를 따른다.

## 재현

원본 UCI 파일과 실제 checkpoint는 Git에 포함하지 않는다. 각각 `data/raw/uci_cbm/data.txt`,
`artifacts/modeling/m3/checkpoints/state_group_seed_42.pt`에 준비한 뒤 실행한다.

```bash
docker build --tag maritime-cbm:m5-local .

uv run --locked --group service python scripts/benchmark_service.py \
  --image maritime-cbm:m5-local \
  --checkpoint artifacts/modeling/m3/checkpoints/state_group_seed_42.pt \
  --data data/raw/uci_cbm/data.txt \
  --output-directory reports/service \
  --host-port 18082 \
  --platform linux/arm64
```

산출물은 다음과 같다.

- `latency_summary.csv`: 작업별 요청 수, 평균·중앙값·p95 지연시간과 오류 수
- `benchmark_results.json`: 프로토콜, 환경, image, 메모리, source와 전체 측정값

## 해석 제한

- 단일 ARM64 MacBook의 Docker/Linux ARM64 결과다.
- concurrency 1의 순차 요청이므로 최대 처리량이나 높은 동시성 성능을 뜻하지 않는다.
- Linux/X64, 클라우드 instance와 production network의 성능을 검증하지 않았다.
- 측정에 사용한 실제 checkpoint와 UCI 입력은 합성 artifact가 아니지만, 이 benchmark는
  모델 정확도나 실제 고장진단 성능을 평가하는 실험이 아니다.
