import json

import torch
from safetensors.torch import load_file

from meddeid_core.taxonomy import BERT_ENTITY_LABELS
from meddeid_training.export import export_bundle


def test_export_uses_canonical_14_label_head_and_self_contained_files(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 17,
            "model_state_dict": {
                "encoder.weight": torch.ones(2, 2),
                "bio_classifier.weight": torch.ones(3, 2),
                "label_classifier.weight": torch.ones(14, 2),
            },
        },
        checkpoint,
    )
    run_metadata = tmp_path / "train_metrics.json"
    run_metadata.write_text(json.dumps({
        "config": {
            "model_name": "stighellemans/meddeid-dutch-synth",
            "initialization_mode": "existing_model",
            "base_encoder": "DTAI-KULeuven/robbert-2023-dutch-base",
            "max_length": 256,
            "overlap": 32,
            "language_profile": "nl-BE",
            "language_profile_version": "1",
        },
        "split_docs": {"train": 8, "val": 1, "test": 1},
    }), encoding="utf-8")

    class FakeConfig:
        hidden_size = 2
        architectures = []

        def to_json_file(self, path):
            path.write_text(
                json.dumps({"hidden_size": self.hidden_size, "architectures": self.architectures}),
                encoding="utf-8",
            )

    class FakeTokenizer:
        def save_pretrained(self, root):
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "meddeid_training.export._transformer_assets",
        lambda base_encoder, base_revision: (FakeConfig(), FakeTokenizer()),
    )
    root = export_bundle(checkpoint, tmp_path / "bundle", run_metadata=run_metadata)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "meddeid-dutch-synth"
    assert manifest["labels"]["entity"] == list(BERT_ENTITY_LABELS)
    assert len(manifest["labels"]["entity"]) == 14
    assert manifest["postprocess"] == {
        "profile_id": "nl-BE",
        "profile_version": "1",
    }
    assert manifest["weights"] == {
        "filename": "model.safetensors",
        "format": "safetensors",
    }
    assert manifest["training"]["checkpoint_epoch"] == 17
    assert manifest["inference"]["max_length"] == 256
    assert manifest["inference"]["overlap"] == 32
    assert manifest["base_encoder"] == "DTAI-KULeuven/robbert-2023-dutch-base"
    assert manifest["training"]["resolved_config"]["initialization_mode"] == "existing_model"
    assert torch.equal(load_file(root / "model.safetensors")["encoder.weight"], torch.ones(2, 2))
    assert (root / "config.json").is_file()
    assert (root / "tokenizer.json").is_file()
    assert not (root / "model.pt").exists()


def test_export_rejects_missing_run_contract(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {"label_classifier.weight": torch.ones(14, 2)}}, checkpoint)
    try:
        export_bundle(checkpoint, tmp_path / "bundle")
    except ValueError as error:
        assert "run metadata" in str(error)
    else:
        raise AssertionError("missing run metadata must fail")
