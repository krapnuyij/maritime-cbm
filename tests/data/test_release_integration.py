"""Integration test for a locally downloaded official UCI release."""

import pytest

from maritime_cbm.config import DEFAULT_RAW_DATA_DIR
from maritime_cbm.data.eda import (
    SENSOR_FEATURE_COLUMNS,
    build_between_speed_variance_ratios,
    build_speed_conditioned_correlations,
    build_train_correlation_matrix,
)
from maritime_cbm.data.loader import load_raw_dataset
from maritime_cbm.data.splitting import build_dataset_splits, compute_split_hashes
from maritime_cbm.data.validation import validate_release

EXPECTED_RELEASE_HASHES = {
    "data.txt": "de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6",
    "Features.txt": "3318e98f507c3bba7ba674d341bcda679c7634027e691c1fe3a43fa3e18b17ae",
    "README.txt": "1ea823d918fed1225329563244e8b0d804912dea4af7a3dad7b0c6caa7356b6b",
}
EXPECTED_SPLIT_HASHES = {
    "random_row": {
        "train": "830e5c6ad4ef9a309336e40de92c90495dc11f7fa49a08d33065270cf077fd50",
        "validation": "63ba78c446800528028b5552030d8179c5b60767be1d87dc497811b8d7a49b7a",
        "test": "fc17cf1842355c9daa35eac1e7c0b3cbf5e7154f2c0091dbce1683ae8e46648b",
    },
    "state_group": {
        "train": "158edd80d6f67c91405d34d20a7657fb29557536893e996758bf6e884e0a0917",
        "validation": "84ac9e1059e569e4085825061a4c549d3dda1042d5a608fe89dcc1e7ca01d997",
        "test": "88bfc5b3cf14be31f862936ecfc9f7fe3663a8bd953a95270feb0934392ccbfe",
    },
    "compressor_holdout": {
        "train": "a0af865d4f671fe7ed56e77b3da3547646451a9027ad714f5e4a8c578c5c326a",
        "validation": "b40891ddb6b4e822958517ffb7f90265ddffc752c52d8162e34584143644848d",
        "test": "e180abb2ca8bcefd29840d4a221966a6a4220ccf8bd66fc1184a1df96ba3a6c2",
    },
    "turbine_holdout": {
        "train": "a074708beb2ec894011265993fd3077f36ef686321d9badfce6fac516a0fb79e",
        "validation": "dc49fecf576259f83e10fbeca6cdf0fcd48f758a21965cd4d38602860be46c55",
        "test": "cd1e52f7efdd68ca4ec4f3f9d818315bc1fa816aa1d9a44854b2fd809d87a180",
    },
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


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_RAW_DATA_DIR.exists(),
    reason="official UCI release is not available under data/raw/uci_cbm",
)
def test_official_uci_release_split_hashes() -> None:
    frame = load_raw_dataset(DEFAULT_RAW_DATA_DIR / "data.txt")

    actual_hashes = compute_split_hashes(build_dataset_splits(frame, seed=42))

    assert actual_hashes == EXPECTED_SPLIT_HASHES


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_RAW_DATA_DIR.exists(),
    reason="official UCI release is not available under data/raw/uci_cbm",
)
def test_official_uci_release_eda_observations() -> None:
    frame = load_raw_dataset(DEFAULT_RAW_DATA_DIR / "data.txt")
    primary_split = build_dataset_splits(frame, seed=42)["state_group"]
    pooled = build_train_correlation_matrix(frame, primary_split).set_index("variable")
    conditioned = build_speed_conditioned_correlations(frame, primary_split)
    conditioned["absolute_correlation"] = conditioned["correlation"].abs()
    conditioned_medians = conditioned.groupby(["feature", "target"])[
        "absolute_correlation"
    ].median()
    variance_ratios = build_between_speed_variance_ratios(frame, primary_split).set_index("feature")

    pooled_maximum = pooled.loc[list(SENSOR_FEATURE_COLUMNS), ["kMc", "kMt"]].abs().max().max()
    assert pooled_maximum == pytest.approx(0.04756933243)
    assert conditioned["absolute_correlation"].max() == pytest.approx(0.9924284915)
    assert conditioned_medians.loc["T2", "kMc"] == pytest.approx(0.934091, abs=1e-6)
    assert conditioned_medians.loc["P2", "kMt"] == pytest.approx(0.948188, abs=1e-6)
    assert conditioned_medians.loc["GTT", "kMc"] == pytest.approx(0.837236, abs=1e-6)
    expected_variance_ratios = {
        "GTT": 0.998640,
        "P2": 0.998627,
        "mf": 0.995815,
        "T48": 0.977108,
    }
    for feature, expected in expected_variance_ratios.items():
        assert variance_ratios.loc[feature, "between_speed_variance_ratio"] == pytest.approx(
            expected,
            abs=1e-6,
        )
