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
    create_compressor_holdout_split,
    create_random_row_split,
    create_state_group_split,
    create_turbine_holdout_split,
    diagnose_group_neighbors,
    hash_row_indices,
)


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
