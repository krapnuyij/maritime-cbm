"""Reproducible dataset splits for comparison and robustness evaluation."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from maritime_cbm.config import get_settings
from maritime_cbm.data.schema import EXPECTED_ROW_COUNT

RANDOM_TEST_ROWS = 1_790
RANDOM_VALIDATION_ROWS = 1_790

GROUP_TRAIN_COUNT = 928
GROUP_VALIDATION_COUNT = 199
GROUP_TEST_COUNT = 199
INTERIOR_TRAIN_GROUP_COUNT = 778
BOUNDARY_GROUP_COUNT = 150
ROWS_PER_STATE_GROUP = 9

KMC_MIN_INDEX = 950
KMC_MAX_INDEX = 1_000
KMT_MIN_INDEX = 975
KMT_MAX_INDEX = 1_000

SPLIT_ROLES: tuple[str, ...] = ("train", "validation", "test")
type SplitMapping = dict[str, "DatasetSplit"]


class SplitDefinitionError(ValueError):
    """Raised when the release cannot satisfy an approved split definition."""


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Sorted zero-based row positions for one evaluation scenario."""

    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def indices_by_role(self) -> dict[str, np.ndarray]:
        """Return split arrays in the canonical role order."""
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True, slots=True)
class GroupNeighborDiagnostics:
    """Train-neighbor coverage for test groups in the primary group split."""

    test_group_count: int
    test_groups_with_eight_neighbor: int
    test_groups_with_four_neighbor: int

    @property
    def eight_neighbor_percentage(self) -> float:
        """Return test coverage by any adjacent or diagonal train group."""
        return 100.0 * self.test_groups_with_eight_neighbor / self.test_group_count

    @property
    def four_neighbor_percentage(self) -> float:
        """Return test coverage by any axis-adjacent train group."""
        return 100.0 * self.test_groups_with_four_neighbor / self.test_group_count


def _require_release_shape(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_ROW_COUNT:
        raise SplitDefinitionError(
            f"Expected {EXPECTED_ROW_COUNT} rows for approved splits, found {len(frame)}"
        )
    missing_targets = tuple(column for column in ("kMc", "kMt") if column not in frame.columns)
    if missing_targets:
        raise SplitDefinitionError(f"Required target columns are missing: {missing_targets}")


def _coefficient_indices(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, ["kMc", "kMt"]].to_numpy(dtype=np.float64, copy=False)
    scaled = values * 1_000
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-6):
        raise SplitDefinitionError("kMc and kMt must lie on the approved 0.001 grid")
    return rounded.astype(np.int64)


def _sorted_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask).astype(np.int64)


def _validate_partition(split: DatasetSplit, row_count: int = EXPECTED_ROW_COUNT) -> None:
    arrays = tuple(split.indices_by_role().values())
    combined = np.concatenate(arrays)
    if len(combined) != row_count:
        raise SplitDefinitionError(f"{split.name} does not contain exactly {row_count} rows")
    if not all(np.array_equal(indices, np.sort(indices)) for indices in arrays):
        raise SplitDefinitionError(f"{split.name} indices must be sorted")
    if not np.array_equal(np.sort(combined), np.arange(row_count, dtype=np.int64)):
        raise SplitDefinitionError(f"{split.name} roles must be disjoint and cover every row")


def create_random_row_split(
    row_count: int = EXPECTED_ROW_COUNT,
    seed: int | None = None,
) -> DatasetSplit:
    """Create the comparison-only exact-count random row split."""
    if row_count != EXPECTED_ROW_COUNT:
        raise SplitDefinitionError(
            f"Expected {EXPECTED_ROW_COUNT} rows for random split, found {row_count}"
        )
    resolved_seed = get_settings().random_seed if seed is None else seed
    permutation = np.random.default_rng(resolved_seed).permutation(row_count)
    test_end = RANDOM_TEST_ROWS
    validation_end = test_end + RANDOM_VALIDATION_ROWS
    split = DatasetSplit(
        name="random_row",
        train=np.sort(permutation[validation_end:]).astype(np.int64),
        validation=np.sort(permutation[test_end:validation_end]).astype(np.int64),
        test=np.sort(permutation[:test_end]).astype(np.int64),
    )
    _validate_partition(split, row_count)
    return split


def create_state_group_split(frame: pd.DataFrame, seed: int | None = None) -> DatasetSplit:
    """Split lexicographically identified state groups while fixing boundaries to train."""
    _require_release_shape(frame)
    coefficient_indices = _coefficient_indices(frame)
    group_tuples = sorted({tuple(row) for row in coefficient_indices.tolist()})
    group_ids = {group: group_id for group_id, group in enumerate(group_tuples)}

    boundary_ids = np.asarray(
        [
            group_ids[group]
            for group in group_tuples
            if group[0] in {KMC_MIN_INDEX, KMC_MAX_INDEX}
            or group[1] in {KMT_MIN_INDEX, KMT_MAX_INDEX}
        ],
        dtype=np.int64,
    )
    boundary_id_set = set(boundary_ids.tolist())
    interior_ids = np.asarray(
        [group_ids[group] for group in group_tuples if group_ids[group] not in boundary_id_set],
        dtype=np.int64,
    )
    expected_interior_count = INTERIOR_TRAIN_GROUP_COUNT + GROUP_VALIDATION_COUNT + GROUP_TEST_COUNT
    if len(boundary_ids) != BOUNDARY_GROUP_COUNT or len(interior_ids) != expected_interior_count:
        raise SplitDefinitionError(
            "Expected 150 boundary groups and 1,176 interior groups in the release"
        )

    resolved_seed = get_settings().random_seed if seed is None else seed
    shuffled_interior = interior_ids[
        np.random.default_rng(resolved_seed).permutation(len(interior_ids))
    ]
    test_ids = set(shuffled_interior[:GROUP_TEST_COUNT].tolist())
    validation_ids = set(
        shuffled_interior[GROUP_TEST_COUNT : GROUP_TEST_COUNT + GROUP_VALIDATION_COUNT].tolist()
    )
    train_ids = boundary_id_set | set(
        shuffled_interior[GROUP_TEST_COUNT + GROUP_VALIDATION_COUNT :].tolist()
    )

    row_group_ids = np.fromiter(
        (group_ids[tuple(row)] for row in coefficient_indices.tolist()),
        dtype=np.int64,
        count=len(frame),
    )
    split = DatasetSplit(
        name="state_group",
        train=_sorted_indices(np.isin(row_group_ids, tuple(train_ids))),
        validation=_sorted_indices(np.isin(row_group_ids, tuple(validation_ids))),
        test=_sorted_indices(np.isin(row_group_ids, tuple(test_ids))),
    )
    _validate_partition(split)
    expected_rows = (
        GROUP_TRAIN_COUNT * ROWS_PER_STATE_GROUP,
        GROUP_VALIDATION_COUNT * ROWS_PER_STATE_GROUP,
        GROUP_TEST_COUNT * ROWS_PER_STATE_GROUP,
    )
    actual_rows = (len(split.train), len(split.validation), len(split.test))
    if actual_rows != expected_rows:
        raise SplitDefinitionError(
            f"State group split row counts differ: expected {expected_rows}, found {actual_rows}"
        )
    return split


def create_compressor_holdout_split(frame: pd.DataFrame) -> DatasetSplit:
    """Hold out consecutive lower kMc ranges for severe-degradation extrapolation."""
    _require_release_shape(frame)
    kmc = _coefficient_indices(frame)[:, 0]
    split = DatasetSplit(
        name="compressor_holdout",
        train=_sorted_indices(kmc >= 960),
        validation=_sorted_indices((kmc >= 955) & (kmc <= 959)),
        test=_sorted_indices((kmc >= 950) & (kmc <= 954)),
    )
    _validate_partition(split)
    return split


def create_turbine_holdout_split(frame: pd.DataFrame) -> DatasetSplit:
    """Hold out consecutive lower kMt ranges for severe-degradation extrapolation."""
    _require_release_shape(frame)
    kmt = _coefficient_indices(frame)[:, 1]
    split = DatasetSplit(
        name="turbine_holdout",
        train=_sorted_indices(kmt >= 985),
        validation=_sorted_indices((kmt >= 980) & (kmt <= 984)),
        test=_sorted_indices((kmt >= 975) & (kmt <= 979)),
    )
    _validate_partition(split)
    return split


def build_dataset_splits(frame: pd.DataFrame, seed: int | None = None) -> SplitMapping:
    """Build all approved M1 split scenarios without sharing RNG state."""
    _require_release_shape(frame)
    return {
        "random_row": create_random_row_split(len(frame), seed),
        "state_group": create_state_group_split(frame, seed),
        "compressor_holdout": create_compressor_holdout_split(frame),
        "turbine_holdout": create_turbine_holdout_split(frame),
    }


def hash_row_indices(indices: np.ndarray) -> str:
    """Hash sorted row positions serialized as contiguous little-endian int64 bytes."""
    normalized = np.sort(np.asarray(indices, dtype=np.dtype("<i8")))
    return sha256(normalized.tobytes(order="C")).hexdigest()


def compute_split_hashes(splits: SplitMapping) -> dict[str, dict[str, str]]:
    """Return one row-index hash for every scenario and role."""
    return {
        scenario: {
            role: hash_row_indices(indices) for role, indices in split.indices_by_role().items()
        }
        for scenario, split in splits.items()
    }


def diagnose_group_neighbors(
    frame: pd.DataFrame,
    split: DatasetSplit,
) -> GroupNeighborDiagnostics:
    """Measure train-neighbor coverage for primary-split test state groups."""
    coefficient_indices = _coefficient_indices(frame)
    train_groups = {tuple(row) for row in coefficient_indices[split.train].tolist()}
    test_groups = {tuple(row) for row in coefficient_indices[split.test].tolist()}

    eight_neighbor_count = 0
    four_neighbor_count = 0
    for kmc, kmt in test_groups:
        eight_neighbors = {
            (kmc + kmc_offset, kmt + kmt_offset)
            for kmc_offset in (-1, 0, 1)
            for kmt_offset in (-1, 0, 1)
            if (kmc_offset, kmt_offset) != (0, 0)
        }
        four_neighbors = {
            (kmc - 1, kmt),
            (kmc + 1, kmt),
            (kmc, kmt - 1),
            (kmc, kmt + 1),
        }
        eight_neighbor_count += bool(eight_neighbors & train_groups)
        four_neighbor_count += bool(four_neighbors & train_groups)

    return GroupNeighborDiagnostics(
        test_group_count=len(test_groups),
        test_groups_with_eight_neighbor=eight_neighbor_count,
        test_groups_with_four_neighbor=four_neighbor_count,
    )
