"""Measure the deployed API under one fixed sequential Docker/Linux protocol."""

import argparse
import csv
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from maritime_cbm.data.features import select_model_features
from maritime_cbm.data.loader import compute_sha256, load_raw_dataset

WARMUP_REQUESTS = 50
SINGLE_REQUESTS = 1_000
BATCH_REQUESTS = 200
BATCH_SIZE = 100
COLD_START_RUNS = 5
STARTUP_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Aggregate timing for one fixed request shape."""

    operation: str
    request_count: int
    items_per_request: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    error_count: int


def _run(*arguments: str, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _container_exists(name: str) -> bool:
    result = _run(
        "docker",
        "ps",
        "-a",
        "--filter",
        f"name=^/{name}$",
        "--format",
        "{{.Names}}",
    )
    return bool(result.stdout.strip())


def _remove_container(name: str) -> None:
    if _container_exists(name):
        _run("docker", "rm", "--force", name)


def _start_container(
    *,
    name: str,
    image: str,
    checkpoint_path: Path,
    host_port: int,
    target_platform: str,
) -> None:
    if _container_exists(name):
        raise RuntimeError(f"Benchmark container already exists: {name}")
    _run(
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--platform",
        target_platform,
        "--publish",
        f"127.0.0.1:{host_port}:8000",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=67108864",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--mount",
        (
            f"type=bind,src={checkpoint_path.resolve()},"
            "dst=/run/model/state_group_seed_42.pt,readonly"
        ),
        image,
    )


def _wait_until_ready(base_url: str) -> None:
    deadline = time.perf_counter() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "ok" and payload.get("model_loaded") is True:
                return
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(0.01)
    raise RuntimeError("Container did not become ready within the fixed timeout") from last_error


def _measure_cold_starts(
    *,
    image: str,
    checkpoint_path: Path,
    host_port: int,
    target_platform: str,
) -> tuple[list[float], str]:
    samples = []
    retained_name = "maritime-cbm-benchmark-4"
    for index in range(COLD_START_RUNS):
        name = f"maritime-cbm-benchmark-{index}"
        try:
            started = time.perf_counter_ns()
            _start_container(
                name=name,
                image=image,
                checkpoint_path=checkpoint_path,
                host_port=host_port,
                target_platform=target_platform,
            )
            _wait_until_ready(f"http://127.0.0.1:{host_port}")
        except Exception:
            _remove_container(name)
            raise
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if index < COLD_START_RUNS - 1:
            _remove_container(name)
    return samples, retained_name


def _latency_summary(
    operation: str,
    samples_ms: list[float],
    *,
    items_per_request: int,
    error_count: int,
) -> LatencySummary:
    values = np.asarray(samples_ms, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"No valid latency samples were collected for {operation}")
    return LatencySummary(
        operation=operation,
        request_count=len(values),
        items_per_request=items_per_request,
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        error_count=error_count,
    )


def _measure_requests(
    client: httpx.Client,
    path: str,
    payload: dict[str, object],
    request_count: int,
) -> tuple[list[float], int]:
    samples = []
    errors = 0
    for _ in range(request_count):
        started = time.perf_counter_ns()
        try:
            response = client.post(path, json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            errors += 1
            continue
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, errors


def _proc_status_kib(container_name: str) -> dict[str, int]:
    status = _run("docker", "exec", container_name, "cat", "/proc/1/status").stdout
    values: dict[str, int] = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError(f"Unexpected /proc memory unit: {unit}")
            values[key.rstrip(":")] = int(value)
    if set(values) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("Container process memory fields are unavailable")
    return values


def _memory_to_mib(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", value.strip())
    if match is None:
        raise RuntimeError(f"Unexpected Docker memory value: {value}")
    number = float(match.group(1))
    unit = match.group(2)
    factors = {
        "B": 1 / (1024 * 1024),
        "KiB": 1 / 1024,
        "MiB": 1.0,
        "GiB": 1024.0,
    }
    if unit not in factors:
        raise RuntimeError(f"Unexpected Docker memory unit: {unit}")
    return number * factors[unit]


def _docker_memory_usage_mib(container_name: str) -> float:
    usage = _run(
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{.MemUsage}}",
        container_name,
    ).stdout.strip()
    current = usage.split("/", maxsplit=1)[0]
    return _memory_to_mib(current)


def _container_environment(container_name: str) -> dict[str, object]:
    script = (
        "import importlib.metadata as m,json,platform,sys,torch;"
        "print(json.dumps({"
        "'python':platform.python_version(),"
        "'torch':str(torch.__version__),"
        "'fastapi':m.version('fastapi'),"
        "'uvicorn':m.version('uvicorn'),"
        "'machine':platform.machine(),"
        "'platform':platform.platform(),"
        "'executable':sys.executable}))"
    )
    environment = json.loads(_run("docker", "exec", container_name, "python", "-c", script).stdout)
    os_release = _run("docker", "exec", container_name, "cat", "/etc/os-release").stdout
    pretty_name = next(
        (
            line.split("=", maxsplit=1)[1].strip('"')
            for line in os_release.splitlines()
            if line.startswith("PRETTY_NAME=")
        ),
        "unknown",
    )
    environment["os_pretty_name"] = pretty_name
    return environment


def _write_outputs(
    output_directory: Path,
    summaries: list[LatencySummary],
    result: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "latency_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    (output_directory / "benchmark_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_benchmark(arguments: argparse.Namespace) -> None:
    """Run the fixed protocol and persist only measured values."""
    checkpoint_path = arguments.checkpoint.resolve()
    data_path = arguments.data.resolve()
    if not checkpoint_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Official checkpoint and data file are required")
    frame = select_model_features(load_raw_dataset(data_path))
    single_payload = {key: float(value) for key, value in frame.iloc[0].items()}
    batch_payload = {
        "items": [
            {key: float(value) for key, value in row.items()}
            for _, row in frame.iloc[:BATCH_SIZE].iterrows()
        ]
    }

    base_url = f"http://127.0.0.1:{arguments.host_port}"
    retained_name = ""
    checkpoint_before = compute_sha256(checkpoint_path)
    try:
        cold_samples, retained_name = _measure_cold_starts(
            image=arguments.image,
            checkpoint_path=checkpoint_path,
            host_port=arguments.host_port,
            target_platform=arguments.platform,
        )
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            for _ in range(WARMUP_REQUESTS):
                response = client.post("/v1/condition/predict", json=single_payload)
                response.raise_for_status()
            idle_memory = _proc_status_kib(retained_name)
            single_samples, single_errors = _measure_requests(
                client,
                "/v1/condition/predict",
                single_payload,
                SINGLE_REQUESTS,
            )
            batch_samples, batch_errors = _measure_requests(
                client,
                "/v1/condition/batch",
                batch_payload,
                BATCH_REQUESTS,
            )

        peak_memory = _proc_status_kib(retained_name)
        container_memory_mib = _docker_memory_usage_mib(retained_name)
        environment = _container_environment(retained_name)
        if single_errors or batch_errors:
            raise RuntimeError(
                f"Benchmark requests failed: single={single_errors}, batch={batch_errors}"
            )

        summaries = [
            _latency_summary(
                "cold_start",
                cold_samples,
                items_per_request=0,
                error_count=0,
            ),
            _latency_summary(
                "single_predict",
                single_samples,
                items_per_request=1,
                error_count=single_errors,
            ),
            _latency_summary(
                "batch_100_predict",
                batch_samples,
                items_per_request=BATCH_SIZE,
                error_count=batch_errors,
            ),
        ]
        image_id = _run(
            "docker", "image", "inspect", "--format", "{{.Id}}", arguments.image
        ).stdout.strip()
        image_size = int(
            _run(
                "docker", "image", "inspect", "--format", "{{.Size}}", arguments.image
            ).stdout.strip()
        )
        git_commit = _run("git", "rev-parse", "HEAD").stdout.strip()
        git_status = _run("git", "status", "--porcelain").stdout.splitlines()
        checkpoint_after = compute_sha256(checkpoint_path)
        if checkpoint_before != checkpoint_after:
            raise RuntimeError("Checkpoint SHA-256 changed during the benchmark")

        result: dict[str, Any] = {
            "schema_version": 1,
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "protocol": {
                "warmup_requests": WARMUP_REQUESTS,
                "single_requests": SINGLE_REQUESTS,
                "batch_requests": BATCH_REQUESTS,
                "batch_size": BATCH_SIZE,
                "cold_start_runs": COLD_START_RUNS,
                "concurrency": 1,
                "uvicorn_workers": 1,
                "http_connection": "one persistent HTTP/1.1 client, sequential requests",
            },
            "latency": [asdict(summary) for summary in summaries],
            "memory": {
                "idle_process_rss_mib": idle_memory["VmRSS"] / 1024.0,
                "process_peak_rss_mib": peak_memory["VmHWM"] / 1024.0,
                "container_usage_after_requests_mib": container_memory_mib,
            },
            "image": {
                "reference": arguments.image,
                "id": image_id,
                "size_bytes": image_size,
                "size_mib": image_size / (1024 * 1024),
            },
            "environment": {
                "target_platform": arguments.platform,
                "container": environment,
                "docker_server": _run(
                    "docker", "version", "--format", "{{.Server.Version}}"
                ).stdout.strip(),
                "host_platform": platform.platform(),
            },
            "source": {
                "git_commit": git_commit,
                "git_dirty": bool(git_status),
                "git_status": git_status,
                "checkpoint_path": str(arguments.checkpoint),
                "checkpoint_sha256": checkpoint_after,
                "data_path": str(arguments.data),
            },
        }
        _write_outputs(arguments.output_directory, summaries, result)
    finally:
        if retained_name:
            _remove_container(retained_name)


def parse_arguments() -> argparse.Namespace:
    """Parse fixed-protocol deployment inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--host-port", type=int, default=18082)
    parser.add_argument("--platform", default="linux/arm64")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_arguments())
