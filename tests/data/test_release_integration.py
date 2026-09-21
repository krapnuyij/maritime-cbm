"""Integration test for a locally downloaded official UCI release."""

import pytest

from maritime_cbm.config import DEFAULT_RAW_DATA_DIR
from maritime_cbm.data.validation import validate_release

EXPECTED_RELEASE_HASHES = {
    "data.txt": "de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6",
    "Features.txt": "3318e98f507c3bba7ba674d341bcda679c7634027e691c1fe3a43fa3e18b17ae",
    "README.txt": "1ea823d918fed1225329563244e8b0d804912dea4af7a3dad7b0c6caa7356b6b",
}


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_RAW_DATA_DIR.exists(),
    reason="official UCI release is not available under data/raw/uci_cbm",
)
def test_official_uci_release() -> None:
    report = validate_release(DEFAULT_RAW_DATA_DIR)

    assert report.row_count == 11_934
    assert report.column_count == 18
    assert report.missing_value_count == 0
    assert report.unique_speed_count == 9
    assert report.unique_kmc_count == 51
    assert report.unique_kmt_count == 26
    assert report.observations.constant_columns == ("T1", "P1")
    assert report.observations.duplicate_column_pairs == (("Ts", "Tp"),)
    assert report.observations.lp_v_one_to_one is True
    assert {item.relative_path: item.sha256 for item in report.files} == EXPECTED_RELEASE_HASHES
