"""Approved M2 baseline model candidates and deterministic construction."""

from dataclasses import dataclass
from typing import Literal

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maritime_cbm.config import get_settings
from maritime_cbm.modeling.preprocessing import SpeedConditionedCenterer

RIDGE_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
RANDOM_FOREST_MIN_SAMPLES_LEAF: tuple[int, ...] = (1, 3, 9)
RANDOM_FOREST_MAX_FEATURES: tuple[float | str, ...] = (1.0, "sqrt")
RANDOM_FOREST_ESTIMATORS = 300

type ModelFamily = Literal["dummy", "ridge", "random_forest"]
type PreprocessingKind = Literal["raw", "speed_centered"]
type ParameterValue = int | float | str


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Serializable definition of one pre-approved baseline candidate."""

    candidate_id: str
    family: ModelFamily
    preprocessing: PreprocessingKind
    parameters: tuple[tuple[str, ParameterValue], ...]
    complexity_rank: int
    selectable: bool = True

    def parameter_dict(self) -> dict[str, ParameterValue]:
        """Return hyperparameters as a new dictionary."""
        return dict(self.parameters)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "preprocessing": self.preprocessing,
            "parameters": self.parameter_dict(),
            "complexity_rank": self.complexity_rank,
            "selectable": self.selectable,
        }


def _value_token(value: ParameterValue) -> str:
    return str(value).replace(".", "p")


def build_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the frozen M2 candidate grid in deterministic order."""
    candidates = [
        CandidateSpec(
            candidate_id="dummy_mean",
            family="dummy",
            preprocessing="raw",
            parameters=(("strategy", "mean"),),
            complexity_rank=0,
            selectable=False,
        )
    ]
    for preprocessing, complexity_rank in (("raw", 1), ("speed_centered", 2)):
        for alpha in RIDGE_ALPHAS:
            candidates.append(
                CandidateSpec(
                    candidate_id=f"ridge_{preprocessing}_alpha_{_value_token(alpha)}",
                    family="ridge",
                    preprocessing=preprocessing,
                    parameters=(("alpha", alpha),),
                    complexity_rank=complexity_rank,
                )
            )
    for min_samples_leaf in RANDOM_FOREST_MIN_SAMPLES_LEAF:
        for max_features in RANDOM_FOREST_MAX_FEATURES:
            candidates.append(
                CandidateSpec(
                    candidate_id=(
                        "random_forest_raw"
                        f"_leaf_{min_samples_leaf}"
                        f"_features_{_value_token(max_features)}"
                    ),
                    family="random_forest",
                    preprocessing="raw",
                    parameters=(
                        ("max_features", max_features),
                        ("min_samples_leaf", min_samples_leaf),
                        ("n_estimators", RANDOM_FOREST_ESTIMATORS),
                    ),
                    complexity_rank=3,
                )
            )
    return tuple(candidates)


def get_candidate_spec(candidate_id: str) -> CandidateSpec:
    """Return one candidate by stable identifier."""
    matches = tuple(spec for spec in build_candidate_specs() if spec.candidate_id == candidate_id)
    if not matches:
        raise KeyError(f"Unknown model candidate: {candidate_id}")
    return matches[0]


def build_estimator(
    spec: CandidateSpec,
    *,
    seed: int | None = None,
) -> TransformedTargetRegressor:
    """Build an unfitted target-scaled multi-output estimator for one candidate."""
    resolved_seed = get_settings().random_seed if seed is None else seed
    parameters = spec.parameter_dict()

    if spec.family == "dummy":
        regressor = DummyRegressor(strategy=str(parameters["strategy"]))
    elif spec.family == "ridge":
        steps: list[tuple[str, object]] = []
        if spec.preprocessing == "speed_centered":
            steps.append(("speed_centering", SpeedConditionedCenterer()))
        steps.extend(
            (
                ("feature_scaling", StandardScaler()),
                ("regressor", Ridge(alpha=float(parameters["alpha"]), solver="svd")),
            )
        )
        regressor = Pipeline(steps)
    elif spec.family == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=parameters["max_features"],
            random_state=resolved_seed,
            n_jobs=1,
        )
    else:  # pragma: no cover - CandidateSpec is constructed from a closed literal set.
        raise ValueError(f"Unsupported model family: {spec.family}")

    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )
