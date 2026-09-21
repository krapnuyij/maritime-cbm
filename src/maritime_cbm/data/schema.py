"""Official UCI naval propulsion dataset schema."""

FEATURE_COLUMNS: tuple[str, ...] = (
    "lp",
    "v",
    "GTT",
    "GTn",
    "GGn",
    "Ts",
    "Tp",
    "T48",
    "T1",
    "T2",
    "P48",
    "P1",
    "P2",
    "Pexh",
    "TIC",
    "mf",
)
TARGET_COLUMNS: tuple[str, ...] = ("kMc", "kMt")
ALL_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + TARGET_COLUMNS

EXPECTED_ROW_COUNT = 11_934
EXPECTED_COLUMN_COUNT = len(ALL_COLUMNS)
EXPECTED_SPEED_VALUES: tuple[float, ...] = tuple(float(value) for value in range(3, 28, 3))
EXPECTED_KMC_VALUES: tuple[float, ...] = tuple(value / 1_000 for value in range(950, 1_001))
EXPECTED_KMT_VALUES: tuple[float, ...] = tuple(value / 1_000 for value in range(975, 1_001))
EXPECTED_GRID_SIZE = (
    len(EXPECTED_SPEED_VALUES) * len(EXPECTED_KMC_VALUES) * len(EXPECTED_KMT_VALUES)
)
GRID_ABSOLUTE_TOLERANCE = 1e-6
GRID_ROUND_DECIMALS = 6

REQUIRED_RELEASE_FILES: tuple[str, ...] = ("data.txt", "Features.txt", "README.txt")
