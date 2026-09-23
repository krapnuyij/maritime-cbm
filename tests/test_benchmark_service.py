"""Regression tests for the fixed-protocol service benchmark outputs."""

from pathlib import Path
from runpy import run_path


def test_write_outputs_uses_deterministic_lf_csv(tmp_path: Path) -> None:
    """Write path-independent CSV bytes without platform CRLF defaults."""
    namespace = run_path(str(Path(__file__).parents[1] / "scripts" / "benchmark_service.py"))
    latency_summary = namespace["LatencySummary"]
    write_outputs = namespace["_write_outputs"]
    summaries = [
        latency_summary(
            operation="single_predict",
            request_count=1,
            items_per_request=1,
            mean_ms=1.0,
            median_ms=1.0,
            p95_ms=1.0,
            error_count=0,
        )
    ]
    result = {"schema_version": 1}

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_outputs(first, summaries, result)
    write_outputs(second, summaries, result)

    first_bytes = (first / "latency_summary.csv").read_bytes()
    second_bytes = (second / "latency_summary.csv").read_bytes()
    assert first_bytes == second_bytes
    assert b"\r\n" not in first_bytes
    assert first_bytes.endswith(b"\n")
