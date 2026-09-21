from hashlib import sha256
from itertools import product

import numpy as np
import pandas as pd
import pytest

from maritime_cbm.data.schema import EXPECTED_KMC_VALUES, EXPECTED_KMT_VALUES
from maritime_cbm.data.splitting import (
    BOUNDARY_GROUP_COUNT,
    DatasetSplit,
    SplitDefinitionError,
    build_dataset_splits,
    compute_split_hashes,
    create_compressor_holdout_split,
    create_random_row_split,
    create_state_group_split,
    create_turbine_holdout_split,
    diagnose_group_neighbors,
    hash_row_indices,
)

# Rebaseline these together with the official-release hashes after any NumPy or
# split-definition change. They intentionally duplicate the documented release hashes.
EXPECTED_SYNTHETIC_SPLIT_HASHES = {
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


@pytest.fixture(scope="module")
def release_targets() -> pd.DataFrame:
    rows = [
        (speed, kmc, kmt)
        for kmc, kmt, speed in product(
            EXPECTED_KMC_VALUES,
            EXPECTED_KMT_VALUES,
            range(3, 28, 3),
        )
    ]
    return pd.DataFrame(rows, columns=["v", "kMc", "kMt"])


def _assert_complete_partition(split: DatasetSplit) -> None:
    combined = np.concatenate(tuple(split.indices_by_role().values()))
    assert len(np.unique(combined)) == 11_934
    assert np.array_equal(np.sort(combined), np.arange(11_934))
    assert all(
        np.array_equal(indices, np.sort(indices)) for indices in split.indices_by_role().values()
    )


def test_random_row_split_uses_exact_counts_and_is_reproducible() -> None:
    first = create_random_row_split(seed=42)
    second = create_random_row_split(seed=42)

    assert (len(first.train), len(first.validation), len(first.test)) == (8_354, 1_790, 1_790)
    assert all(
        np.array_equal(first.indices_by_role()[role], second.indices_by_role()[role])
        for role in first.indices_by_role()
    )
    _assert_complete_partition(first)


def test_random_row_split_rejects_non_release_size() -> None:
    with pytest.raises(SplitDefinitionError, match="11934"):
        create_random_row_split(row_count=10, seed=42)


def test_state_group_split_fixes_boundaries_to_train(release_targets: pd.DataFrame) -> None:
    split = create_state_group_split(release_targets, seed=42)
    coefficients = np.rint(release_targets.loc[:, ["kMc", "kMt"]].to_numpy() * 1_000).astype(
        np.int64
    )
    boundary_mask = np.isin(coefficients[:, 0], [950, 1_000]) | np.isin(
        coefficients[:, 1], [975, 1_000]
    )

    assert (len(split.train), len(split.validation), len(split.test)) == (8_352, 1_791, 1_791)
    assert np.all(np.isin(np.flatnonzero(boundary_mask), split.train))
    boundary_groups = np.unique(coefficients[boundary_mask], axis=0)
    assert len(boundary_groups) == BOUNDARY_GROUP_COUNT
    assert len(np.unique(coefficients[split.train], axis=0)) == 928
    assert len(np.unique(coefficients[split.validation], axis=0)) == 199
    assert len(np.unique(coefficients[split.test], axis=0)) == 199
    _assert_complete_partition(split)


def test_state_group_split_is_seed_reproducible_and_seed_sensitive(
    release_targets: pd.DataFrame,
) -> None:
    first = create_state_group_split(release_targets, seed=42)
    repeated = create_state_group_split(release_targets, seed=42)
    different_seed = create_state_group_split(release_targets, seed=43)

    assert np.array_equal(first.test, repeated.test)
    assert np.array_equal(first.validation, repeated.validation)
    assert not np.array_equal(first.test, different_seed.test)
    assert not np.array_equal(first.validation, different_seed.validation)


def test_robustness_holdouts_use_approved_ranges(release_targets: pd.DataFrame) -> None:
    compressor = create_compressor_holdout_split(release_targets)
    turbine = create_turbine_holdout_split(release_targets)

    assert (len(compressor.train), len(compressor.validation), len(compressor.test)) == (
        9_594,
        1_170,
        1_170,
    )
    assert release_targets.iloc[compressor.train]["kMc"].min() == pytest.approx(0.960)
    assert np.sort(release_targets.iloc[compressor.validation]["kMc"].unique()) == pytest.approx(
        [0.955, 0.956, 0.957, 0.958, 0.959]
    )
    assert release_targets.iloc[compressor.test]["kMc"].max() == pytest.approx(0.954)

    assert (len(turbine.train), len(turbine.validation), len(turbine.test)) == (
        7_344,
        2_295,
        2_295,
    )
    assert release_targets.iloc[turbine.train]["kMt"].min() == pytest.approx(0.985)
    assert np.sort(release_targets.iloc[turbine.validation]["kMt"].unique()) == pytest.approx(
        [0.980, 0.981, 0.982, 0.983, 0.984]
    )
    assert release_targets.iloc[turbine.test]["kMt"].max() == pytest.approx(0.979)
    _assert_complete_partition(compressor)
    _assert_complete_partition(turbine)


def test_split_hash_serializes_sorted_little_endian_int64() -> None:
    indices = np.asarray([3, 1, 2], dtype=np.int32)
    expected_bytes = np.asarray([1, 2, 3], dtype=np.dtype("<i8")).tobytes()

    assert hash_row_indices(indices) == sha256(expected_bytes).hexdigest()


def test_build_splits_and_neighbor_diagnostics(release_targets: pd.DataFrame) -> None:
    splits = build_dataset_splits(release_targets, seed=42)
    diagnostics = diagnose_group_neighbors(release_targets, splits["state_group"])

    assert tuple(splits) == (
        "random_row",
        "state_group",
        "compressor_holdout",
        "turbine_holdout",
    )
    assert diagnostics.test_group_count == 199
    assert diagnostics.test_groups_with_eight_neighbor == 199
    assert 0 < diagnostics.test_groups_with_four_neighbor <= 199


def test_canonical_synthetic_grid_reproduces_documented_split_hashes(
    release_targets: pd.DataFrame,
) -> None:
    """Reproduce release hashes from its kMc-major, then kMt, nine-row state blocks.

    The official-release integration hashes indirectly confirm that its state-block
    placement matches this synthetic grid. Row order for v within a state group does
    not affect these hashes. A different state-block placement would invalidate the
    correspondence between the synthetic and official-release hashes.
    """
    actual_hashes = compute_split_hashes(build_dataset_splits(release_targets, seed=42))

    assert actual_hashes == EXPECTED_SYNTHETIC_SPLIT_HASHES
