"""A fallback is signed, declared and attested without changing historical bundles."""

import hashlib
import json
import zipfile

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
    original = inspect_bundle(archive)
    source = tmp_path / "router"
    source.mkdir()
    with zipfile.ZipFile(archive) as contents:
        for filename in ("model.py", "model.safetensors"):
            (source / filename).write_bytes(contents.read(filename))
    fallback = ModelChoice(**MODEL)
    config_raw = json.dumps(
        {
            "required_models": [m.model_dump(mode="json") for m in original.required_models],
            "fallback_model": MODEL,
        }
    ).encode()
    (source / "models.json").write_bytes(config_raw)
    packed = package(
        source,
        tmp_path / "fallback.zip",
        name="with-fallback",
    )
    assert packed.fallback_model == fallback
    with zipfile.ZipFile(tmp_path / "fallback.zip") as contents:
        assert contents.read("models.json") == config_raw
        manifest = json.loads(contents.read("manifest.json"))
        assert manifest["fallback_model"] == fallback.model_dump(mode="json")
        assert manifest["files"]["models.json"] == hashlib.sha256(config_raw).hexdigest()
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


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        "null",
        "",
        "{}",
        json.dumps({"required_models": [MODEL], "fallback_modle": MODEL}),
        json.dumps(
            {
                "required_models": [MODEL],
                "fallback_model": {"provider_id": "other", "model_id": "undeclared"},
            }
        ),
        json.dumps(
            {"required_models": [MODEL], "fallback_model": {**MODEL, "effort": "undeclared-effort"}}
        ),
        json.dumps({"required_models": [MODEL, MODEL]}),
        json.dumps([MODEL, MODEL]),
    ],
)
def test_invalid_config_cannot_silently_disable_fallback_or_replace_output(tmp_path, raw):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models.json").write_text(raw)
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous submission")
    with pytest.raises(ValueError):
        package(tmp_path, output, name="router", required_models=(ModelChoice(**MODEL),))
    assert output.read_bytes() == b"previous submission"


def test_models_config_is_bounded_and_cannot_be_overwritten(tmp_path):
    from yugos.submission.bundle import MAX_MANIFEST_BYTES

    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    config = tmp_path / "models.json"
    config.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="config size limit"):
        package(
            tmp_path,
            tmp_path / "submission.zip",
            name="router",
            required_models=(ModelChoice(**MODEL),),
        )
    config.write_text(json.dumps([MODEL]))
    with pytest.raises(ValueError, match="must not overwrite"):
        package(tmp_path, config, name="router", required_models=(ModelChoice(**MODEL),))
    assert json.loads(config.read_text()) == [MODEL]


@pytest.mark.parametrize(
    "settings",
    [
        [MODEL],
        {"required_models": [MODEL]},
        {"required_models": [MODEL], "fallback_model": None},
    ],
)
def test_legacy_array_and_object_without_fallback_are_equivalent(tmp_path, settings):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models.json").write_text(json.dumps(settings))
    metadata = package(tmp_path, tmp_path / "submission.zip", name="router")
    manifest = read_bundle_manifest(tmp_path / "submission.zip")
    assert metadata.required_models == (ModelChoice(**MODEL),)
    assert "fallback_model" not in metadata.model_dump(mode="json")
    assert "fallback_model" not in manifest.model_dump(mode="json")


def test_legacy_python_packaging_api_remains_supported(tmp_path):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    metadata = package(
        tmp_path,
        tmp_path / "submission.zip",
        name="router",
        required_models=(ModelChoice(**MODEL),),
    )
    assert metadata.required_models == (ModelChoice(**MODEL),)
    assert metadata.fallback_model is None


def test_conflicting_python_declarations_do_not_override_models_config(tmp_path):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models.json").write_text(
        json.dumps({"required_models": [MODEL], "fallback_model": MODEL})
    )
    with pytest.raises(ValueError, match="differs from models.json"):
        package(
            tmp_path,
            tmp_path / "submission.zip",
            name="router",
            required_models=(ModelChoice(provider_id="other", model_id="model"),),
        )


def test_unrelated_submission_json_remains_an_ordinary_asset(tmp_path):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models.json").write_text(json.dumps([MODEL]))
    (tmp_path / "submission.json").write_text('{"custom":"router asset"}')
    metadata = package(tmp_path, tmp_path / "submission.zip", name="router")
    assert metadata.fallback_model is None
    with zipfile.ZipFile(tmp_path / "submission.zip") as contents:
        assert contents.read("submission.json") == b'{"custom":"router asset"}'
