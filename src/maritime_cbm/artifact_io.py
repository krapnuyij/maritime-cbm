"""Deterministic serialization helpers for local experiment artifacts."""

from io import BytesIO
from pathlib import Path

import pandas as pd


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write gzip bytes without filename or timestamp metadata."""
    buffer = BytesIO()
    frame.to_csv(
        buffer,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    path.write_bytes(buffer.getvalue())
