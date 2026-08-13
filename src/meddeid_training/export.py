from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddeid_core.taxonomy import BERT_ENTITY_LABELS, bio_labels


def _load_run_metadata(checkpoint: Path, run_metadata: str | Path | None) -> tuple[Path, dict[str, Any]]:
    path = Path(run_metadata).expanduser().resolve() if run_metadata else checkpoint.parent.parent / "train_metrics.json"
    if not path.is_file():
        raise ValueError(
            "export requires the resolved training run metadata; pass --run-metadata "
            f"(looked for {path})"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("config"), dict):
        raise ValueError(f"run metadata has no resolved config: {path}")
    return path, payload


def _validate_heads(state: dict[str, Any]) -> None:
    entity = state.get("label_classifier.weight")
    bio = state.get("bio_classifier.weight")
    if entity is None or int(entity.shape[0]) != len(BERT_ENTITY_LABELS):
        raise ValueError(
            f"checkpoint entity head must have {len(BERT_ENTITY_LABELS)} canonical labels"
        )
    if bio is None or int(bio.shape[0]) != len(bio_labels()):
        raise ValueError(f"checkpoint BIO head must have {len(bio_labels())} labels")


def _checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("model export requires meddeid-training[train]") from exc
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a state dict or contain model_state_dict")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint contains no model tensors")
    non_tensors = [name for name, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise ValueError(f"checkpoint state contains non-tensors: {non_tensors[:5]}")
    return payload, state


def _transformer_assets(base_encoder: str, base_revision: str | None):
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("model export requires meddeid-training[train]") from exc
    kwargs = {"revision": base_revision} if base_revision else {}
    config = AutoConfig.from_pretrained(base_encoder, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        base_encoder,
        use_fast=True,
        add_prefix_space=True,
        **kwargs,
    )
    return config, tokenizer


def export_bundle(
    checkpoint: str | Path,
    output: str | Path,
    *,
    run_metadata: str | Path | None = None,
    base_encoder: str | None = None,
    base_revision: str | None = None,
    name: str = "meddeid-dutch-synth",
    max_length: int | None = None,
    overlap: int | None = None,
    unsafe_override: bool = False,
) -> Path:
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    metadata_path, run = _load_run_metadata(source, run_metadata)
    resolved = run["config"]
    run_encoder = str(resolved.get("base_encoder") or resolved["model_name"])
    run_max_length = int(resolved["max_length"])
    run_overlap = int(resolved["overlap"])
    profile_id = str(resolved["language_profile"])
    profile_version = str(resolved["language_profile_version"])
    conflicts = {
        "base_encoder": (base_encoder, run_encoder),
        "max_length": (max_length, run_max_length),
        "overlap": (overlap, run_overlap),
    }
    conflicting = [name for name, (manual, actual) in conflicts.items() if manual is not None and manual != actual]
    if conflicting and not unsafe_override:
        raise ValueError(
            "manual export values conflict with the training run: "
            + ", ".join(conflicting)
            + "; remove them or pass --unsafe-override"
        )
    base_encoder = str(base_encoder if base_encoder is not None else run_encoder)
    max_length = int(max_length if max_length is not None else run_max_length)
    overlap = int(overlap if overlap is not None else run_overlap)
    base_revision = base_revision or resolved.get("base_revision")
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"model export directory must be empty: {root}")

    payload, state = _checkpoint_state(source)
    _validate_heads(state)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("model export requires safetensors") from exc
    save_file(state, root / "model.safetensors", metadata={"format": "pt"})

    config, tokenizer = _transformer_assets(base_encoder, base_revision)
    config.architectures = ["MedDeIDDualHeadTokenClassifier"]
    config.to_json_file(root / "config.json")
    tokenizer.save_pretrained(root)

    checkpoint_epoch = payload.get("epoch") if payload is not state else None
    manifest = {
        "bundle_version": "meddeid.bundle.v1",
        "artifact_version": "0.1.0",
        "model_version": "1",
        "name": name,
        "task": "dual_head_token_classification",
        "base_encoder": base_encoder,
        "base_encoder_revision": base_revision,
        "hidden_size": int(config.hidden_size),
        "weights": {"filename": "model.safetensors", "format": "safetensors"},
        "encoder_config": "config.json",
        "tokenizer_path": ".",
        "labels": {"bio": list(bio_labels()), "entity": list(BERT_ENTITY_LABELS)},
        "inference": {"max_length": max_length, "overlap": overlap, "min_entity_score": 0.0},
        "postprocess": {"profile_id": profile_id, "profile_version": profile_version},
    }
    if checkpoint_epoch is not None:
        manifest["training"] = {"checkpoint_epoch": int(checkpoint_epoch)}
    manifest.setdefault("training", {}).update(
        {
            "run_metadata": metadata_path.name,
            "resolved_config": resolved,
            "split_documents": run.get("split_docs"),
        }
    )
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root
