from pathlib import Path

import pandas as pd

from maritime_cbm.artifact_io import write_deterministic_gzip_csv


def test_gzip_csv_is_deterministic_across_paths(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "row_index": [0, 1],
            "kMc_predicted": [0.95, 0.96],
            "alert": [True, False],
        }
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "different-name.csv.gz"

    write_deterministic_gzip_csv(frame, first)
    write_deterministic_gzip_csv(frame, second)

    assert first.read_bytes() == second.read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(first), frame)
    pd.testing.assert_frame_equal(pd.read_csv(second), frame)
