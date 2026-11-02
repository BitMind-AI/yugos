"""A fallback is signed, declared and attested without changing historical bundles."""

import json

import pytest
from bundles import MODEL, make_bundle

from yugos.inference.actions import ModelChoice
from yugos.miner import package
from yugos.submission.bundle import PreparationError, inspect_bundle, read_bundle_manifest
from yugos.submission.prepared import prepare_to_directory, read_prepared_directory


def test_old_bundle_metadata_and_manifest_omit_optional_fallback(tmp_path):
    archive, _, _ = make_bundle(tmp_path)
    assert "fallback_model" not in inspect_bundle(archive).model_dump(mode="json")
    assert "fallback_model" not in read_bundle_manifest(archive).model_dump(mode="json")


def test_fallback_survives_packaging_preparation_and_attestation(tmp_path):
    archive, _, _ = make_bundle(tmp_path / "source")
    destination = tmp_path / "prepared"
    original = inspect_bundle(archive)
    prepared = prepare_to_directory(
        archive, destination, submission_sha256=original.submission_sha256
    )
    fallback = ModelChoice(**MODEL)
    packed = package(
        prepared.root,
        tmp_path / "fallback.zip",
        name="with-fallback",
        required_models=original.required_models,
        fallback_model=fallback,
    )
    assert packed.fallback_model == fallback
    staged = tmp_path / "fallback-prepared"
    prepare_to_directory(
        tmp_path / "fallback.zip", staged, submission_sha256=packed.submission_sha256
    )
    assert (
        read_prepared_directory(
            staged, submission_sha256=packed.submission_sha256
        ).metadata.fallback_model
        == fallback
    )
    data = json.loads((staged / "metadata.json").read_text())
    data["fallback_model"] = original.required_models[1].model_dump(mode="json")
    (staged / "metadata.json").write_text(json.dumps(data))
    with pytest.raises(PreparationError, match="metadata differs from manifest"):
        read_prepared_directory(staged, submission_sha256=packed.submission_sha256)


def test_undeclared_fallback_rejected_before_loading_submission(tmp_path):
    archive, _, _ = make_bundle(
        tmp_path,
        manifest_changes={"fallback_model": {"provider_id": "other", "model_id": "missing"}},
    )
    with pytest.raises(ValueError, match="must also appear in required_models"):
        inspect_bundle(archive)
