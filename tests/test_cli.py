from pathlib import Path
import json

import pytest
import torch
from safetensors.torch import save_file

from meddeid_core.taxonomy import BERT_ENTITY_LABELS
from meddeid_training import DEFAULT_INITIAL_MODEL
from meddeid_training.cli import selected_epochs, training_command
from meddeid_training.train_script import (
    plot_history,
    read_jsonl,
    resolve_model_initialization,
    select_device,
    typed_ids_to_char_spans,
)


CONFIG = {
    "model_name": "example/existing-meddeid-model",
    "language_profile": "nl-BE",
    "language_profile_version": "1",
    "epochs": 9,
    "head_warmup_epochs": 2,
    "encoder_lr": 3e-5,
    "head_lr": 2e-4,
    "weight_decay": 0.02,
    "warmup_ratio": 0.2,
    "save_best_metric": "entity_f1",
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 0.002,
    "early_stopping_min_epochs": 5,
    "dataloader_num_workers": 2,
    "dataloader_prefetch_factor": 3,
    "fp16": True,
    "gradient_checkpointing": True,
    "device": "cpu",
    "attn_implementation": "eager",
}


def test_selected_epoch_is_already_one_based() -> None:
    assert selected_epochs({"best_epoch": 17}) == 17


def test_selection_forwards_release_config_and_withholds_benchmark(tmp_path) -> None:
    command = training_command(
        CONFIG,
        Path(tmp_path / "data"),
        Path(tmp_path / "run"),
        mode="select-epochs",
    )
    assert command[command.index("--head-warmup-epochs") + 1] == "2"
    assert command[command.index("--encoder-lr") + 1] == "3e-05"
    assert command[command.index("--early-stopping-patience") + 1] == "4"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--language-profile") + 1] == "nl-BE"
    assert command[command.index("--model-name") + 1] == CONFIG["model_name"]
    assert command[command.index("--attn-implementation") + 1] == "eager"
    assert "--skip-test-evaluation" in command
    assert "--fp16" in command
    assert "--gradient-checkpointing" in command
    assert "--from-base-encoder" not in command
    assert "--final-epoch-is-best" not in command


def test_fit_is_one_stage_training_with_validation_and_test(tmp_path) -> None:
    command = training_command(
        CONFIG,
        Path(tmp_path / "data"),
        Path(tmp_path / "run"),
        mode="fit",
        epochs=5,
    )

    assert command[command.index("--epochs") + 1] == "5"
    assert command[command.index("--early-stopping-patience") + 1] == "4"
    assert "--skip-test-evaluation" not in command
    assert "--final-epoch-is-best" not in command


def test_refit_uses_selected_epoch_and_disables_validation_selection(tmp_path) -> None:
    command = training_command(
        CONFIG,
        Path(tmp_path / "data"),
        Path(tmp_path / "run"),
        mode="refit",
        epochs=17,
    )
    assert command[command.index("--epochs") + 1] == "17"
    assert command[command.index("--model-name") + 1] == CONFIG["model_name"]
    assert command[command.index("--early-stopping-patience") + 1] == "0"
    assert "--final-epoch-is-best" in command
    assert "--skip-test-evaluation" not in command


def test_training_reader_rejects_historical_annotations(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({
        "document_id": "d1",
        "text": "Jan",
        "annotations": [{
            "begin": 0,
            "end": 3,
            "text": "Jan",
            "label": "Name:Patient",
            "Category": "Name",
            "Subtype": "Patient",
        }],
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical key 'spans'"):
        read_jsonl(source)


def test_cpu_device_can_always_be_forced() -> None:
    assert str(select_device("cpu")) == "cpu"


def test_existing_meddeid_model_is_the_default_initialization(tmp_path) -> None:
    config = dict(CONFIG)
    config.pop("model_name")

    command = training_command(
        config,
        Path(tmp_path / "data"),
        Path(tmp_path / "run"),
        mode="select-epochs",
    )

    assert command[command.index("--model-name") + 1] == DEFAULT_INITIAL_MODEL
    assert "--from-base-encoder" not in command


def test_base_encoder_initialization_requires_explicit_config(tmp_path) -> None:
    config = {
        **CONFIG,
        "model_name": "example/base-encoder",
        "model_revision": "abc1234",
        "from_base_encoder": True,
    }

    command = training_command(
        config,
        Path(tmp_path / "data"),
        Path(tmp_path / "run"),
        mode="select-epochs",
    )

    assert command[command.index("--model-name") + 1] == "example/base-encoder"
    assert command[command.index("--model-revision") + 1] == "abc1234"
    assert "--from-base-encoder" in command


def test_character_decoder_collapses_duplicate_token_boundaries() -> None:
    spans = typed_ids_to_char_spans(
        tag_ids=[1, 2],
        id2label={1: "B-Date", 2: "B-Name:Patient"},
        offsets=[(4, 5), (4, 5)],
        text="abcdX",
    )

    assert spans == [{"begin": 4, "end": 5, "label": "Date", "text": "X"}]


def test_plot_history_writes_readable_raster_and_vector_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    history = [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "train_eval_loss": 0.7,
            "val_loss": 0.75,
            "train_bio_token_f1_macro": 0.70,
            "val_bio_token_f1_macro": 0.65,
            "train_label_token_f1_macro": 0.60,
            "val_label_token_f1_macro": 0.55,
            "train_entity_f1": 0.50,
            "val_entity_f1": 0.45,
        },
        {
            "epoch": 2,
            "train_loss": 0.5,
            "train_eval_loss": 0.45,
            "val_loss": 0.55,
            "train_bio_token_f1_macro": 0.82,
            "val_bio_token_f1_macro": 0.78,
            "train_label_token_f1_macro": 0.74,
            "val_label_token_f1_macro": 0.69,
            "train_entity_f1": 0.68,
            "val_entity_f1": 0.62,
        },
    ]

    plot_history(history, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "loss_curve.png",
        "loss_curve.pdf",
        "f1_curve.png",
        "f1_curve.pdf",
    }
    assert all(path.stat().st_size > 1_000 for path in tmp_path.iterdir())


def test_explicit_plain_model_directory_is_treated_as_base_encoder(tmp_path) -> None:
    resolved = resolve_model_initialization(
        str(tmp_path),
        from_base_encoder=False,
    )

    assert resolved.mode == "base_encoder"
    assert resolved.encoder_source == str(tmp_path)
    assert resolved.state_dict is None


def test_existing_bundle_initialization_restores_heads_and_provenance(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    state = {
        "encoder.weight": torch.ones(2, 2),
        "bio_classifier.weight": torch.ones(3, 2),
        "label_classifier.weight": torch.ones(len(BERT_ENTITY_LABELS), 2),
    }
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_version": "meddeid.bundle.v1",
                "artifact_version": "0.1.0",
                "model_version": "1",
                "name": "test-model",
                "task": "dual_head_token_classification",
                "base_encoder": "example/base-encoder",
                "base_encoder_revision": "abc1234",
                "hidden_size": 2,
                "weights": {
                    "filename": "model.safetensors",
                    "format": "safetensors",
                },
                "encoder_config": "config.json",
                "tokenizer_path": ".",
                "labels": {
                    "bio": ["O", "B", "I"],
                    "entity": list(BERT_ENTITY_LABELS),
                },
                "inference": {
                    "max_length": 128,
                    "overlap": 32,
                    "min_entity_score": 0.0,
                },
                "postprocess": {
                    "profile_id": "nl-BE",
                    "profile_version": "1",
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_initialization(
        str(tmp_path),
        from_base_encoder=False,
    )

    assert resolved.mode == "existing_model"
    assert resolved.initialize_encoder_from_pretrained is False
    assert resolved.local_files_only is True
    assert resolved.base_encoder == "example/base-encoder"
    assert resolved.base_revision == "abc1234"
    assert torch.equal(resolved.state_dict["bio_classifier.weight"], state["bio_classifier.weight"])
