from pathlib import Path

import pytest

from maritime_cbm.modeling.benchmark import (
    SELECTION_PROTOCOL,
    SELECTION_SCHEMA_VERSION,
    SelectionManifestError,
    _load_json,
    _runtime_versions,
    _validate_selection_manifest,
)
from maritime_cbm.modeling.models import build_candidate_specs


def _manifest() -> tuple[dict[str, object], dict[str, str], dict[str, dict[str, str]]]:
    candidate = next(spec for spec in build_candidate_specs() if spec.selectable)
    release_hashes = {"data.txt": "abc"}
    split_hashes = {"state_group": {"train": "def"}}
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "seed": 42,
        "release_hashes": release_hashes,
        "split_hashes": split_hashes,
        "runtime_versions": _runtime_versions(),
        "selected_candidate": candidate.to_dict(),
    }
    return payload, release_hashes, split_hashes


def test_selection_manifest_accepts_matching_frozen_candidate() -> None:
    payload, release_hashes, split_hashes = _manifest()

    candidate = _validate_selection_manifest(
        payload,
        seed=42,
        release_hashes=release_hashes,
        split_hashes=split_hashes,
    )

    assert candidate.to_dict() == payload["selected_candidate"]


@pytest.mark.parametrize("changed_key", ["seed", "release_hashes", "runtime_versions"])
def test_selection_manifest_rejects_environment_mismatch(changed_key: str) -> None:
    payload, release_hashes, split_hashes = _manifest()
    payload[changed_key] = "changed"

    with pytest.raises(SelectionManifestError, match=changed_key):
        _validate_selection_manifest(
            payload,
            seed=42,
            release_hashes=release_hashes,
            split_hashes=split_hashes,
        )


def test_selection_manifest_rejects_modified_candidate() -> None:
    payload, release_hashes, split_hashes = _manifest()
    stored_candidate = dict(payload["selected_candidate"])
    stored_candidate["parameters"] = {"alpha": 999}
    payload["selected_candidate"] = stored_candidate

    with pytest.raises(SelectionManifestError, match="candidate grid"):
        _validate_selection_manifest(
            payload,
            seed=42,
            release_hashes=release_hashes,
            split_hashes=split_hashes,
        )


def test_selection_manifest_error_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SelectionManifestError, match="Cannot load"):
        _load_json(missing)
