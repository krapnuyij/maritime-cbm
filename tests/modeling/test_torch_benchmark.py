from pathlib import Path

import pytest

from maritime_cbm.modeling.torch_benchmark import (
    SELECTION_PROTOCOL,
    SELECTION_SCHEMA_VERSION,
    TorchSelectionManifestError,
    _load_json,
    _runtime_environment,
    _runtime_versions,
    _validate_selection_manifest,
)
from maritime_cbm.modeling.torch_evaluation import build_selection_seeds
from maritime_cbm.modeling.torch_models import build_torch_candidate_specs
from maritime_cbm.modeling.torch_training import TrainingConfig


def _manifest() -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    candidates = build_torch_candidate_specs()
    release_hashes = {"data.txt": "abc"}
    split_hashes = {"state_group": {"train": "def"}}
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "base_seed": 42,
        "selection_seeds": list(build_selection_seeds(42)),
        "release_hashes": release_hashes,
        "split_hashes": split_hashes,
        "runtime_versions": _runtime_versions(),
        "runtime_environment": _runtime_environment(),
        "device_preference": "cpu",
        "training_config": TrainingConfig().to_dict(),
        "candidate_grid": [candidate.to_dict() for candidate in candidates],
        "candidate_count": len(candidates),
        "selected_candidate": candidates[0].to_dict(),
    }
    return payload, release_hashes, split_hashes


def test_m3_manifest_accepts_matching_runtime_and_candidate() -> None:
    payload, release_hashes, split_hashes = _manifest()

    candidate = _validate_selection_manifest(
        payload,
        base_seed=42,
        release_hashes=release_hashes,
        split_hashes=split_hashes,
        config=TrainingConfig(),
        device="cpu",
    )

    assert candidate.to_dict() == payload["selected_candidate"]


@pytest.mark.parametrize(
    "changed_key",
    ["release_hashes", "runtime_versions", "runtime_environment", "training_config"],
)
def test_m3_manifest_rejects_environment_mismatch(changed_key: str) -> None:
    payload, release_hashes, split_hashes = _manifest()
    payload[changed_key] = "changed"

    with pytest.raises(TorchSelectionManifestError, match=changed_key):
        _validate_selection_manifest(
            payload,
            base_seed=42,
            release_hashes=release_hashes,
            split_hashes=split_hashes,
            config=TrainingConfig(),
            device="cpu",
        )


def test_m3_manifest_rejects_torch_version_change() -> None:
    payload, release_hashes, split_hashes = _manifest()
    versions = dict(payload["runtime_versions"])
    versions["torch"] = "changed"
    payload["runtime_versions"] = versions

    with pytest.raises(TorchSelectionManifestError, match="runtime_versions"):
        _validate_selection_manifest(
            payload,
            base_seed=42,
            release_hashes=release_hashes,
            split_hashes=split_hashes,
            config=TrainingConfig(),
            device="cpu",
        )


def test_m3_manifest_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TorchSelectionManifestError, match="Cannot load"):
        _load_json(tmp_path / "missing.json")
