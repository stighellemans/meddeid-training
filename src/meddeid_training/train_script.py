#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meddeid_core.normalize import normalize_record
from meddeid_core.taxonomy import BERT_ENTITY_LABELS
from meddeid_core.validate import validate_record
from meddeid_eval.span_edits import evaluate_span_edits

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Adafactor,
    get_linear_schedule_with_warmup,
)

from . import DEFAULT_INITIAL_MODEL

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data paths, model settings, and training hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-style model for token classification (DEID)"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_INITIAL_MODEL,
        help=(
            "Existing MedDeID model bundle to fine-tune (local directory or Hugging "
            "Face repository). With --from-base-encoder, this instead names the base "
            "encoder whose classifier heads will be initialized from scratch."
        ),
    )
    parser.add_argument(
        "--from-base-encoder",
        action="store_true",
        help="Explicitly initialize from a base encoder with new random classifier heads",
    )
    parser.add_argument(
        "--model-revision",
        help="Immutable Hub revision for the existing model or explicit base encoder",
    )
    parser.add_argument("--language-profile", required=True)
    parser.add_argument("--language-profile-version", required=True)
    parser.add_argument("--output-dir", default=os.environ.get("JOB_OUT", "output"))
    parser.add_argument("--ckpt-dir", default=os.environ.get("JOB_CKPT", "checkpoints"))

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=4,
        help="Number of prefetched batches per DataLoader worker",
    )
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory even on CUDA",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers between epochs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Print training/evaluation progress every N batches (0 disables periodic logs)",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars for training/evaluation loops",
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-warmup-epochs", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adafactor"],
        default="adamw",
        help="Optimizer to use. Adafactor is lower-memory for local full-model runs.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa"),
        default="auto",
        help="Attention backend; auto uses eager on MPS for compatibility",
    )
    parser.add_argument(
        "--save-best-metric",
        choices=["entity_f1", "token_f1_macro", "val_loss"],
        default="entity_f1",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help=(
            "Stop after N consecutive epochs without validation improvement "
            "on the selection metric (0 disables early stopping)"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum improvement required to reset early-stopping patience",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=3,
        help="Earliest epoch where patience counting starts",
    )
    parser.add_argument(
        "--final-epoch-is-best",
        action="store_true",
        help=(
            "Treat the FINAL epoch as the best model instead of selecting best.pt by the "
            "validation metric. Use for a full-train refit where val is folded into train "
            "and val-based selection would be in-sample. best.pt then equals last.pt, and "
            "the reported best_* / test metrics equal the final-epoch metrics -- so best.pt "
            "and eval.json's `test` field are always the model and score to report. Forces "
            "early stopping off."
        ),
    )
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help=(
            "Do not read or score test.jsonl. Use during validation-based epoch "
            "selection so the external test set remains completely withheld."
        ),
    )
    parser.add_argument("--label-key", default="label")
    parser.add_argument(
        "--entity-labels",
        default="",
        help="Comma-separated canonical entity label set (fixed classifier head, "
        "consistent size+indices across all models). Empty -> CANONICAL_ENTITY_TYPES.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use fp16 mixed precision on CUDA"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM use",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device with actionable errors."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; retry with --device cpu")
        return torch.device("cuda")
    if requested == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested but is unavailable; retry with --device cpu")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def empty_device_cache(device: torch.device) -> None:
    """Release cached accelerator allocations between large training phases."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return

    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and canonicalize JSONL, rejecting invalid offsets or labels."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = normalize_record(json.loads(line))
            problems = validate_record(row, strict_taxonomy=True)
            if problems:
                rendered = "; ".join(problems[:5])
                raise ValueError(f"{path}:{line_number}: invalid canonical record: {rendered}")
            rows.append(row)
    return rows


def _doc_identity(row: dict[str, Any]) -> str:
    """Best-effort stable identity for a document, used for split-overlap checks."""
    for key in ("doc_id", "document_id", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return str(row.get("text", ""))


def check_split_disjointness(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    early_stopping_enabled: bool,
) -> None:
    """Guard against silently evaluating or selecting on in-sample data.

    Early stopping and best-checkpoint selection are driven by the val set, and the
    reported test metrics must come from documents the model never trained on. This
    verifies that assumption BEFORE training so it can never fail silently:

    - Test overlapping train or val is always a leak -> hard error.
    - Val overlapping train while early stopping is enabled makes the early-stopping
      and best-checkpoint signal in-sample (and therefore invalid) -> hard error.
      With early stopping disabled this is the intended full-train refit, where val is
      only a sanity metric, so it is reported as an informational note instead.

    Set DEID_ALLOW_SPLIT_OVERLAP=1 to downgrade these hard errors to warnings.
    """
    train_ids = {_doc_identity(r) for r in train_rows}
    val_ids = {_doc_identity(r) for r in val_rows}
    test_ids = {_doc_identity(r) for r in test_rows}

    test_in_train = test_ids & train_ids
    test_in_val = test_ids & val_ids
    val_in_train = val_ids & train_ids

    allow_override = str(
        os.environ.get("DEID_ALLOW_SPLIT_OVERLAP", "")
    ).strip().lower() in {"1", "true", "yes", "on"}

    def fail(message: str) -> None:
        if allow_override:
            print(f"[split-guard] WARNING (override): {message}", flush=True)
        else:
            raise RuntimeError(
                message
                + " Set DEID_ALLOW_SPLIT_OVERLAP=1 to bypass this guard if intentional."
            )

    if test_in_train:
        fail(
            f"TEST LEAKAGE: {len(test_in_train)} test doc(s) also appear in train; "
            "test metrics would be invalid."
        )
    if test_in_val:
        fail(
            f"TEST LEAKAGE: {len(test_in_val)} test doc(s) also appear in val; "
            "test metrics would be invalid."
        )

    if val_in_train:
        if early_stopping_enabled:
            fail(
                f"IN-SAMPLE EARLY STOPPING: {len(val_in_train)} val doc(s) also appear in "
                "train, so early stopping and best-checkpoint selection would run on data "
                "the model trained on. Use the held-out-val data layout for epoch "
                "selection, or disable early stopping (--early-stopping-patience 0) and "
                "set a fixed --epochs for a full-train refit."
            )
        else:
            print(
                f"[split-guard] note: {len(val_in_train)} val doc(s) are also in train "
                "(full-train refit layout); val metrics are a sanity signal only, not a "
                "held-out estimate.",
                flush=True,
            )
    elif not test_in_train and not test_in_val:
        print(
            "[split-guard] ok: train/val/test document sets are disjoint",
            flush=True,
        )


def normalize_spans_for_edit_evaluation(
    spans: list[dict[str, Any]], text: str, label_key: str
) -> list[dict[str, Any]]:
    """Normalize dataset spans to begin/end/label/text records for span-edit scoring."""
    normalized: list[dict[str, Any]] = []
    text_len = len(text)
    for span in spans:
        try:
            begin = int(span.get("begin", 0))
            end = int(span.get("end", 0))
        except (TypeError, ValueError):
            continue

        begin = max(0, min(begin, text_len))
        end = max(0, min(end, text_len))
        if end <= begin:
            continue

        label = (
            span.get(label_key)
            or span.get("label")
            or span.get("Category")
            or span.get("type")
        )
        if label is None:
            continue

        normalized.append(
            {
                "begin": begin,
                "end": end,
                "label": str(label),
                "text": text[begin:end],
            }
        )

    normalized.sort(key=lambda item: (int(item["begin"]), int(item["end"]), item["label"]))
    return normalized


def load_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
):
    def _from_pretrained(use_fast: bool):
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            add_prefix_space=True,
            revision=revision,
            local_files_only=local_files_only,
        )

    try:
        tokenizer = _from_pretrained(use_fast=True)
    except ImportError as exc:
        if "protobuf" not in str(exc).lower():
            raise
        print(
            "[model] protobuf not available; falling back to slow tokenizer",
            flush=True,
        )
        tokenizer = _from_pretrained(use_fast=False)

    stable_offsets_enabled = (
        str(os.environ.get("DEID_TOKENIZER_STABLE_OFFSETS", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    if (
        stable_offsets_enabled
        and getattr(tokenizer, "is_fast", False)
        and backend is not None
        and "ByteLevel" in str(pre_tokenizer)
    ):
        try:
            from tokenizers.pre_tokenizers import ByteLevel

            backend.pre_tokenizer = ByteLevel(
                add_prefix_space=True,
                use_regex=True,
            )
            print(
                "[tokenizer] applied stable-offset byte-level pretokenizer override",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[tokenizer] stable-offset override unavailable: {exc}", flush=True)

    return tokenizer


def windows_full_tail(n_tokens: int, usable: int, overlap: int):
    """
    Yield overlapping token windows that always include the full tail.

    Windows are ``[begin, end)`` over the original token sequence. The final window is
    forced to end at ``n_tokens`` so the document tail is never dropped.
    """
    if n_tokens <= 0:
        return
    if n_tokens <= usable:
        yield 0, n_tokens
        return

    step = usable - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than usable sequence length")

    last_begin = n_tokens - usable
    begin = 0
    while begin < last_begin:
        end = begin + usable
        yield begin, end
        begin += step
    yield last_begin, n_tokens


def spans_to_bio(
    offsets: list[tuple[int, int]], spans: list[dict[str, Any]], label_key: str
) -> list[str]:
    """
    Convert character-level entity spans to BIO token tags.

    Uses token ``offsets`` from a fast tokenizer and resolves overlapping spans with
    first-assignment precedence after sorting by ``(begin, end)``.
    """
    labels = ["O"] * len(offsets)
    span_sorted = sorted(
        spans, key=lambda x: (int(x.get("begin", 0)), int(x.get("end", 0)))
    )

    assigned_span_id = [None] * len(offsets)
    assigned_label = [None] * len(offsets)

    for span_id, span in enumerate(span_sorted):
        begin = int(span.get("begin", 0))
        end = int(span.get("end", 0))
        if end <= begin:
            continue

        ent = (
            span.get(label_key)
            or span.get("label")
            or span.get("Category")
            or span.get("type")
        )
        if ent is None:
            continue
        ent = str(ent)

        for i, (s, e) in enumerate(offsets):
            if e <= begin:
                continue
            if s >= end:
                continue
            if e <= s:
                continue
            if assigned_span_id[i] is None:
                assigned_span_id[i] = span_id
                assigned_label[i] = ent

    prev_span = None
    for i, (s, e) in enumerate(offsets):
        sid = assigned_span_id[i]
        ent = assigned_label[i]
        if sid is None or ent is None:
            labels[i] = "O"
            if e > s:
                prev_span = None
            continue
        prefix = "B" if sid != prev_span else "I"
        labels[i] = f"{prefix}-{ent}"
        prev_span = sid

    return labels


def extract_entity_types(records: list[dict[str, Any]], label_key: str) -> list[str]:
    """Collect unique entity type names from records and return them sorted."""
    ents = set()
    for row in records:
        for span in row.get("spans", []):
            label = (
                span.get(label_key)
                or span.get("label")
                or span.get("Category")
                or span.get("type")
            )
            if label is not None:
                ents.add(str(label))
    return sorted(ents)


# Canonical entity label set = the project DEID schema for the entity head.
# PINNING it (instead of deriving the vocabulary from whatever labels happen to
# appear in a given split) keeps every trained model's classifier head identical
# in BOTH size and index order -- so checkpoints are interchangeable, comparable,
# and a label with no examples in a given split still gets a consistent, reserved
# head slot instead of silently shrinking the head.
# Single source of truth: the shared 14-label model-head subset from meddeid-core (the 15
# canonical labels minus the Anonymize_Other catch-all, which is not a trainable NER
# entity). Identical alphabetical order -> checkpoints
# stay interchangeable. Change the schema for ALL future models at once via
# meddeid_core.BERT_ENTITY_LABELS (or pass --entity-labels).
CANONICAL_ENTITY_TYPES = list(BERT_ENTITY_LABELS)


def resolve_entity_types(args, records: list[dict[str, Any]]) -> list[str]:
    """Return the FIXED canonical entity label set (NOT data-derived) so the head
    is identical across models. Validates against the data: warns about labels in
    the data but outside the schema (their spans become O), and notes canonical
    labels with no support in this split (kept as zero-support head slots)."""
    raw = (getattr(args, "entity_labels", "") or "").strip()
    canonical = [s.strip() for s in raw.split(",") if s.strip()] if raw else list(CANONICAL_ENTITY_TYPES)
    observed = set(extract_entity_types(records, label_key=args.label_key))
    unknown = sorted(observed - set(canonical))
    if unknown:
        print(
            f"[labels] WARNING: {len(unknown)} data label(s) not in the canonical "
            f"schema -> their spans are treated as O: {unknown}",
            flush=True,
        )
    unobserved = [e for e in canonical if e not in observed]
    if unobserved:
        print(
            f"[labels] note: {len(unobserved)} canonical label(s) absent from this data, "
            f"kept as zero-support head slots for schema consistency: {unobserved}",
            flush=True,
        )
    return canonical


def split_typed_bio_tag(tag: str) -> tuple[str, str | None]:
    """
    Split a typed BIO tag into BIO prefix and entity name.

    Returns:
        ("O", None) for outside tokens or malformed tags.
        ("B" or "I", entity_type) for entity tags.
    """
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return "O", None
    prefix, ent = tag.split("-", 1)
    if prefix not in {"B", "I"} or not ent:
        return "O", None
    return prefix, ent


@dataclass
class DualHeadOutput:
    """Container for dual-head forward outputs and losses."""

    loss: torch.Tensor | None
    bio_loss: torch.Tensor | None
    label_loss: torch.Tensor | None
    bio_logits: torch.Tensor
    label_logits: torch.Tensor


@dataclass(frozen=True)
class ModelInitialization:
    """Resolved encoder, tokenizer, weights, and provenance for one training run."""

    mode: str
    encoder_source: str
    tokenizer_source: str
    initialize_encoder_from_pretrained: bool
    local_files_only: bool
    state_dict: dict[str, torch.Tensor] | None
    base_encoder: str
    base_revision: str | None
    initial_model_revision: str | None
    transformer_revision: str | None
    bio_labels: tuple[str, ...] | None = None
    entity_labels: tuple[str, ...] | None = None
    profile_id: str | None = None
    profile_version: str | None = None


def resolve_model_initialization(
    model_name: str,
    *,
    from_base_encoder: bool,
    model_revision: str | None = None,
) -> ModelInitialization:
    """Resolve either an existing MedDeID bundle or an explicitly requested encoder."""
    def base_encoder_initialization() -> ModelInitialization:
        return ModelInitialization(
            mode="base_encoder",
            encoder_source=model_name,
            tokenizer_source=model_name,
            initialize_encoder_from_pretrained=True,
            local_files_only=False,
            state_dict=None,
            base_encoder=model_name,
            base_revision=model_revision,
            initial_model_revision=None,
            transformer_revision=model_revision,
        )

    if from_base_encoder:
        return base_encoder_initialization()

    try:
        from meddeid.bundle import load_model_bundle
        from meddeid.model import load_checkpoint
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError(
            "fine-tuning an existing model requires meddeid-training[train]"
        ) from exc

    root = Path(model_name).expanduser()
    if root.is_file():
        if root.name != "bundle.json":
            raise ValueError(
                "existing-model initialization requires a bundle directory or bundle.json"
            )
        root = root.parent
    elif root.exists() and not root.is_dir():
        raise ValueError(f"existing model is not a directory: {root}")
    elif not root.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - declared by the train extra
            raise RuntimeError(
                "fine-tuning a Hub model requires meddeid-training[train]"
            ) from exc
        try:
            root = Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=model_revision,
                    allow_patterns=[
                        "bundle.json",
                        "config.json",
                        "model.safetensors",
                        "model.pt",
                        "tokenizer*",
                        "special_tokens_map.json",
                        "vocab.json",
                        "merges.txt",
                    ],
                )
            )
        except Exception as exc:
            if model_name != DEFAULT_INITIAL_MODEL:
                return base_encoder_initialization()
            raise RuntimeError(
                f"existing MedDeID model {model_name!r} could not be downloaded; "
                "verify access or pass a local bundle directory"
            ) from exc

    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        if model_name != DEFAULT_INITIAL_MODEL:
            return base_encoder_initialization()
        raise ValueError(
            f"default existing-model initialization requires bundle.json in {root}"
        )
    bundle = load_model_bundle(manifest_path, validate_package=True)
    if bundle.weights_format not in {"safetensors", "pt"}:
        raise ValueError(
            f"training cannot initialize from {bundle.weights_format} model weights"
        )
    _, state = load_checkpoint(bundle.checkpoint_path)
    encoder_source = (
        str(bundle.encoder_config_path.parent)
        if bundle.encoder_config_path is not None
        else bundle.base_encoder
    )
    tokenizer_source = (
        str(bundle.tokenizer_path)
        if bundle.tokenizer_path is not None
        else bundle.base_encoder
    )
    local_assets = bundle.encoder_config_path is not None
    resolved_revision = (
        root.name
        if root.parent.name == "snapshots" and len(root.name) >= 7
        else None
    )
    return ModelInitialization(
        mode="existing_model",
        encoder_source=encoder_source,
        tokenizer_source=tokenizer_source,
        initialize_encoder_from_pretrained=not local_assets,
        local_files_only=local_assets,
        state_dict=state,
        base_encoder=bundle.base_encoder,
        base_revision=bundle.base_encoder_revision,
        initial_model_revision=resolved_revision,
        transformer_revision=None,
        bio_labels=tuple(bundle.bio_labels),
        entity_labels=tuple(bundle.entity_labels),
        profile_id=bundle.postprocess.profile_id,
        profile_version=bundle.postprocess.profile_version,
    )


class DualHeadTokenClassifier(torch.nn.Module):
    """
    Shared encoder with two token-classification heads.

    - BIO head predicts O/B/I for every token.
    - Label head predicts entity type for every token, but loss is applied only on
      tokens with an entity target.
    """

    def __init__(
        self,
        model_name: str,
        num_bio_labels: int,
        num_entity_labels: int,
        attn_implementation: str | None = None,
        initialize_from_pretrained: bool = True,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        model_kwargs: dict[str, Any] = {"config": config}
        if revision is not None:
            model_kwargs["revision"] = revision
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if initialize_from_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            model_kwargs.pop("config")
            self.encoder = AutoModel.from_config(config, **model_kwargs)
        self.config = config

        hidden_size = int(config.hidden_size)
        dropout_p = float(getattr(config, "hidden_dropout_prob", 0.1))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.bio_classifier = torch.nn.Linear(hidden_size, num_bio_labels)
        self.label_classifier = torch.nn.Linear(hidden_size, num_entity_labels)
        self._init_classifier(self.bio_classifier)
        self._init_classifier(self.label_classifier)

    def _init_classifier(self, layer: torch.nn.Linear) -> None:
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        torch.nn.init.normal_(layer.weight, mean=0.0, std=init_std)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_bio: torch.Tensor | None = None,
        labels_entity: torch.Tensor | None = None,
    ) -> DualHeadOutput:
        encoder_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden = self.dropout(encoder_out.last_hidden_state)
        bio_logits = self.bio_classifier(hidden)
        label_logits = self.label_classifier(hidden)

        bio_loss = None
        label_loss = None
        if labels_bio is not None:
            bio_loss = torch.nn.functional.cross_entropy(
                bio_logits.reshape(-1, bio_logits.shape[-1]),
                labels_bio.reshape(-1),
                ignore_index=-100,
            )

        if labels_entity is not None:
            flat_labels = labels_entity.reshape(-1)
            valid = flat_labels != -100
            if bool(valid.any()):
                flat_logits = label_logits.reshape(-1, label_logits.shape[-1])[valid]
                label_loss = torch.nn.functional.cross_entropy(
                    flat_logits, flat_labels[valid]
                )
            else:
                label_loss = label_logits.new_zeros(())

        if bio_loss is not None and label_loss is not None:
            total_loss = bio_loss + label_loss
        elif bio_loss is not None:
            total_loss = bio_loss
        else:
            total_loss = label_loss

        return DualHeadOutput(
            loss=total_loss,
            bio_loss=bio_loss,
            label_loss=label_loss,
            bio_logits=bio_logits,
            label_logits=label_logits,
        )


@dataclass
class Example:
    """
    One training/evaluation window derived from a single source document.

    Attributes:
        input_ids: Token ids for the window including special tokens expected by the model.
        attention_mask: Mask aligned to ``input_ids`` (1 for real tokens, 0 for padding).
        labels_bio: BIO class ids aligned to ``input_ids``. Positions for special/pad
            tokens use ignore index (typically ``-100``).
        labels_entity: Entity-type class ids aligned to ``input_ids``. Non-entity and
            special/pad tokens use ignore index (``-100``).
        doc_idx: Zero-based index of the source document in the dataset split.
        begin: Inclusive begin token index of this window in the source document tokenization
            (without special tokens).
        end: Exclusive end token index of this window in the source document tokenization
            (without special tokens).
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels_bio: list[int]
    labels_entity: list[int]
    doc_idx: int
    begin: int
    end: int


class WindowedTokenDataset(Dataset):
    """Token-classification dataset that slices each document into overlapping windows."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        typed_label2id: dict[str, int],
        bio_label2id: dict[str, int],
        entity_label2id: dict[str, int],
        label_key: str,
        max_length: int,
        overlap: int,
    ):
        """
        Build windowed examples with model special tokens and loss-aligned labels.

        For each document, this stores full-document gold targets for typed BIO, BIO,
        and entity labels and creates per-window ``Example`` instances.
        """
        self.records = records
        self.tokenizer = tokenizer
        self.typed_label2id = typed_label2id
        self.bio_label2id = bio_label2id
        self.entity_label2id = entity_label2id
        self.examples: list[Example] = []

        max_seq = max_length
        usable = max_seq - tokenizer.num_special_tokens_to_add(pair=False)
        if usable <= 8:
            raise ValueError("max_length too small after accounting for special tokens")

        self.doc_gold_typed_ids: list[list[int]] = []
        self.doc_gold_bio_ids: list[list[int]] = []
        self.doc_gold_entity_ids: list[list[int]] = []
        self.doc_offsets: list[list[tuple[int, int]]] = []
        self.doc_texts: list[str] = []
        self.doc_gold_char_spans: list[list[dict[str, Any]]] = []

        for doc_idx, row in enumerate(records):
            text = row["text"]
            spans = row.get("spans", [])

            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                verbose=False,
            )
            input_ids = enc["input_ids"]
            offsets = [tuple(x) for x in enc["offset_mapping"]]
            self.doc_offsets.append(offsets)
            self.doc_texts.append(text)
            self.doc_gold_char_spans.append(
                normalize_spans_for_edit_evaluation(
                    spans=spans, text=text, label_key=label_key
                )
            )

            gold_typed_tags = spans_to_bio(
                offsets=offsets, spans=spans, label_key=label_key
            )
            gold_typed_ids = [typed_label2id[tag] for tag in gold_typed_tags]
            gold_bio_ids: list[int] = []
            gold_entity_ids: list[int] = []
            for tag in gold_typed_tags:
                bio_tag, ent = split_typed_bio_tag(tag)
                gold_bio_ids.append(bio_label2id[bio_tag])
                if ent is None:
                    gold_entity_ids.append(-100)
                else:
                    gold_entity_ids.append(entity_label2id[ent])

            self.doc_gold_typed_ids.append(gold_typed_ids)
            self.doc_gold_bio_ids.append(gold_bio_ids)
            self.doc_gold_entity_ids.append(gold_entity_ids)

            for s, e in windows_full_tail(
                len(input_ids), usable=usable, overlap=overlap
            ):
                ids_slice = input_ids[s:e]
                labels_bio_slice = gold_bio_ids[s:e]
                labels_entity_slice = gold_entity_ids[s:e]

                packed_ids = None
                special_mask = None

                prepare_for_model = getattr(tokenizer, "prepare_for_model", None)
                if callable(prepare_for_model):
                    try:
                        prepared = prepare_for_model(
                            ids_slice,
                            add_special_tokens=True,
                            return_token_type_ids=False,
                            return_attention_mask=False,
                            return_special_tokens_mask=True,
                        )
                        packed_ids = prepared["input_ids"]
                        special_mask = prepared["special_tokens_mask"]
                    except (AssertionError, NotImplementedError, TypeError, ValueError):
                        packed_ids = None
                        special_mask = None

                if packed_ids is None or special_mask is None:
                    build_inputs = getattr(
                        tokenizer, "build_inputs_with_special_tokens", None
                    )
                    if callable(build_inputs):
                        packed_ids = build_inputs(ids_slice)
                    else:
                        cls_id = getattr(tokenizer, "cls_token_id", None)
                        sep_id = getattr(tokenizer, "sep_token_id", None)
                        if sep_id is None:
                            sep_id = getattr(tokenizer, "eos_token_id", None)
                        if cls_id is None or sep_id is None:
                            raise AttributeError(
                                f"{tokenizer.__class__.__name__} cannot add special tokens: "
                                "missing build_inputs_with_special_tokens and cls/sep token ids"
                            )
                        packed_ids = [cls_id, *ids_slice, sep_id]

                    get_special_mask = getattr(
                        tokenizer, "get_special_tokens_mask", None
                    )
                    if callable(get_special_mask):
                        try:
                            special_mask = get_special_mask(
                                packed_ids, already_has_special_tokens=True
                            )
                        except (
                            AssertionError,
                            NotImplementedError,
                            TypeError,
                            ValueError,
                        ):
                            special_mask = [1] + [0] * len(ids_slice) + [1]
                    else:
                        special_mask = [1] + [0] * len(ids_slice) + [1]

                attention = [1] * len(packed_ids)

                labels_bio_with_special = []
                labels_entity_with_special = []
                j = 0
                for is_special in special_mask:
                    if is_special:
                        labels_bio_with_special.append(-100)
                        labels_entity_with_special.append(-100)
                    else:
                        labels_bio_with_special.append(labels_bio_slice[j])
                        labels_entity_with_special.append(labels_entity_slice[j])
                        j += 1

                if j != len(labels_bio_slice):
                    raise RuntimeError(
                        "Label alignment failure while preparing model inputs"
                    )

                self.examples.append(
                    Example(
                        input_ids=packed_ids,
                        attention_mask=attention,
                        labels_bio=labels_bio_with_special,
                        labels_entity=labels_entity_with_special,
                        doc_idx=doc_idx,
                        begin=s,
                        end=e,
                    )
                )

    def __len__(self) -> int:
        """Return the number of windowed examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single example as a dictionary expected by the collator."""
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels_bio": ex.labels_bio,
            "labels_entity": ex.labels_entity,
            "doc_idx": ex.doc_idx,
            "begin": ex.begin,
            "end": ex.end,
        }


class TokenCollator:
    """Pad variable-length token windows and merge metadata into a batch."""

    def __init__(self, tokenizer):
        """Initialize with the tokenizer used for dynamic batch padding."""
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id or 0)
        self.padding_side = getattr(tokenizer, "padding_side", "right")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate examples into tensors for model input.

        Pads ``input_ids``/``attention_mask`` with tokenizer rules and pads both label
        targets with ``-100`` so padded positions are ignored by losses.
        """
        labels_bio = [item["labels_bio"] for item in batch]
        labels_entity = [item["labels_entity"] for item in batch]
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_size = len(batch)
        padded_input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        padded_attention = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention,
        }

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids_tensor = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_tensor = torch.tensor(item["attention_mask"], dtype=torch.long)
            if self.padding_side == "left":
                start = max_len - seq_len
                padded_input_ids[i, start:] = input_ids_tensor
                padded_attention[i, start:] = attention_tensor
            else:
                padded_input_ids[i, :seq_len] = input_ids_tensor
                padded_attention[i, :seq_len] = attention_tensor

        padded_bio: list[list[int]] = []
        padded_entity: list[list[int]] = []
        for item_bio, item_entity in zip(labels_bio, labels_entity):
            if len(item_bio) > max_len or len(item_entity) > max_len:
                raise RuntimeError("labels length exceeds padded input length")
            if self.padding_side == "left":
                padded_bio.append(([-100] * (max_len - len(item_bio))) + item_bio)
                padded_entity.append(([-100] * (max_len - len(item_entity))) + item_entity)
            else:
                padded_bio.append(item_bio + ([-100] * (max_len - len(item_bio))))
                padded_entity.append(item_entity + ([-100] * (max_len - len(item_entity))))
        padded["labels_bio"] = torch.tensor(padded_bio, dtype=torch.long)
        padded["labels_entity"] = torch.tensor(padded_entity, dtype=torch.long)
        padded["doc_idx"] = torch.tensor(
            [item["doc_idx"] for item in batch], dtype=torch.long
        )
        padded["begin"] = torch.tensor(
            [item["begin"] for item in batch], dtype=torch.long
        )
        padded["end"] = torch.tensor([item["end"] for item in batch], dtype=torch.long)
        return padded


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the transformer's base model parameters."""
    backbone = getattr(model, "encoder", None)
    if backbone is None:
        backbone = getattr(model, "base_model", None)
    if backbone is None:
        raise AttributeError("Model does not expose an encoder/base_model to freeze")
    for p in backbone.parameters():
        p.requires_grad = trainable


def make_optimizer(
    model: torch.nn.Module,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create an optimizer with separate parameter groups for encoder and classifier head.

    Parameters whose names start with ``bio_classifier``, ``label_classifier``, or
    ``classifier`` use ``head_lr``; all other trainable parameters use ``encoder_lr``.
    """
    encoder_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("bio_classifier", "label_classifier", "classifier")):
            head_params.append(p)
        else:
            encoder_params.append(p)

    groups = []
    if encoder_params:
        groups.append(
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    if head_params:
        groups.append(
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        )

    if optimizer_name == "adafactor":
        return Adafactor(
            groups,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(groups)


def bio_tags_to_spans(
    tag_ids: list[int], id2label: dict[int, str]
) -> set[tuple[str, int, int]]:
    """
    Convert token-level BIO label ids to entity spans.

    Returns spans as ``(entity_type, start_idx, end_idx)`` with inclusive token bounds.
    """
    spans: set[tuple[str, int, int]] = set()
    cur_label = None
    cur_start = None

    for i, tag_id in enumerate(tag_ids):
        tag = id2label[int(tag_id)]

        if tag == "O":
            if cur_label is not None and cur_start is not None:
                spans.add((cur_label, cur_start, i - 1))
            cur_label = None
            cur_start = None
            continue

        if "-" not in tag:
            if cur_label is not None and cur_start is not None:
                spans.add((cur_label, cur_start, i - 1))
            cur_label = None
            cur_start = None
            continue

        prefix, ent = tag.split("-", 1)
        if prefix == "B":
            if cur_label is not None and cur_start is not None:
                spans.add((cur_label, cur_start, i - 1))
            cur_label = ent
            cur_start = i
        elif prefix == "I":
            if cur_label == ent and cur_start is not None:
                pass
            else:
                if cur_label is not None and cur_start is not None:
                    spans.add((cur_label, cur_start, i - 1))
                cur_label = ent
                cur_start = i
        else:
            if cur_label is not None and cur_start is not None:
                spans.add((cur_label, cur_start, i - 1))
            cur_label = None
            cur_start = None

    if cur_label is not None and cur_start is not None:
        spans.add((cur_label, cur_start, len(tag_ids) - 1))

    return spans


def typed_ids_to_char_spans(
    tag_ids: list[int],
    id2label: dict[int, str],
    offsets: list[tuple[int, int]],
    text: str,
) -> list[dict[str, Any]]:
    """Convert typed BIO ids into character-level spans with labels and text slices."""
    if len(tag_ids) != len(offsets):
        raise RuntimeError(
            "Mismatch between typed tag sequence and token offsets for span decoding"
        )

    spans: list[dict[str, Any]] = []
    seen_boundaries: set[tuple[int, int]] = set()
    cur_label: str | None = None
    cur_start: int | None = None

    def flush(end_idx: int) -> None:
        nonlocal cur_label, cur_start
        if cur_label is None or cur_start is None:
            return
        if end_idx < cur_start:
            return
        begin = int(offsets[cur_start][0])
        end = int(offsets[end_idx][1])
        if end <= begin:
            return
        boundary = (begin, end)
        if boundary in seen_boundaries:
            # Byte-level tokenizers can assign the same character offsets to
            # more than one token. Pathological BIO output (especially from an
            # untrained model) can then decode to multiple token spans at one
            # character boundary. Canonical spans are boundary-unique, so keep
            # the first decoded prediction deterministically.
            return
        seen_boundaries.add(boundary)
        spans.append(
            {
                "begin": begin,
                "end": end,
                "label": cur_label,
                "text": text[begin:end],
            }
        )

    for idx, tag_id in enumerate(tag_ids):
        tag = id2label[int(tag_id)]
        prefix, ent = split_typed_bio_tag(tag)
        if prefix == "O" or ent is None:
            flush(idx - 1)
            cur_label = None
            cur_start = None
            continue

        if cur_label is None:
            cur_label = ent
            cur_start = idx
            continue

        if prefix == "B" or ent != cur_label:
            flush(idx - 1)
            cur_label = ent
            cur_start = idx

    flush(len(tag_ids) - 1)
    return spans


def evaluate_span_edits_counts(
    pred_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
) -> dict[str, int]:
    """Return dependency-free span-edit operation totals from meddeid-eval."""
    result = evaluate_span_edits(pred_spans, gold_spans, label_key="label")

    counts = result.get("counts", {})
    return {
        "total_ops": int(counts.get("total_ops", 0)),
        "add_ops": int(counts.get("Addition", 0)),
        "edit_ops": int(counts.get("Edit", 0)),
        "delete_ops": int(counts.get("Deletion", 0)),
    }


def evaluate(
    model,
    loader,
    dataset: WindowedTokenDataset,
    typed_id2label: dict[int, str],
    typed_label2id: dict[str, int],
    bio_id2label: dict[int, str],
    bio_label2id: dict[str, int],
    entity_id2label: dict[int, str],
    device: torch.device,
    split_name: str = "eval",
    log_interval: int = 0,
    use_tqdm: bool = True,
    compute_span_edit_metrics: bool = False,
) -> dict[str, Any]:
    """
    Evaluate model performance with dual heads on a windowed dataset.

    Overlapping window logits are averaged per original token. Final typed BIO
    predictions are reconstructed from:
    - BIO head: O/B/I
    - label head: entity type
    """
    model.eval()
    total_batches = len(loader)
    print(
        f"[eval:{split_name}] start batches={total_batches} windows={len(dataset)} docs={len(dataset.doc_gold_typed_ids)}",
        flush=True,
    )

    num_bio_labels = len(bio_id2label)
    num_entity_labels = len(entity_id2label)
    bio_o_id = int(bio_label2id["O"])
    doc_bio_logits: list[np.ndarray] = []
    doc_entity_logits: list[np.ndarray] = []
    doc_counts: list[np.ndarray] = []

    for gold in dataset.doc_gold_typed_ids:
        n = len(gold)
        doc_bio_logits.append(np.zeros((n, num_bio_labels), dtype=np.float64))
        doc_entity_logits.append(np.zeros((n, num_entity_labels), dtype=np.float64))
        doc_counts.append(np.zeros(n, dtype=np.int64))

    losses = []
    bio_losses = []
    label_losses = []

    eval_bar = tqdm(
        loader,
        total=total_batches,
        desc=f"eval:{split_name}",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not use_tqdm,
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_bar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_bio = batch["labels_bio"].to(device, non_blocking=True)
            labels_entity = batch["labels_entity"].to(device, non_blocking=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_bio=labels_bio,
                labels_entity=labels_entity,
            )
            if out.loss is not None:
                losses.append(float(out.loss.detach().cpu().item()))
            if out.bio_loss is not None:
                bio_losses.append(float(out.bio_loss.detach().cpu().item()))
            if out.label_loss is not None:
                label_losses.append(float(out.label_loss.detach().cpu().item()))

            bio_logits = out.bio_logits.detach().cpu().numpy()
            label_logits = out.label_logits.detach().cpu().numpy()
            labels_bio_np = labels_bio.detach().cpu().numpy()
            doc_idx_np = batch["doc_idx"].numpy()
            begin_np = batch["begin"].numpy()
            end_np = batch["end"].numpy()

            for i in range(bio_logits.shape[0]):
                valid_mask = labels_bio_np[i] != -100
                token_bio_logits = bio_logits[i][valid_mask]
                token_label_logits = label_logits[i][valid_mask]
                doc_idx = int(doc_idx_np[i])
                begin = int(begin_np[i])
                end = int(end_np[i])

                if token_bio_logits.shape[0] != (end - begin):
                    raise RuntimeError(
                        "Mismatch between BIO logits and window span length"
                    )

                doc_bio_logits[doc_idx][begin:end] += token_bio_logits
                doc_entity_logits[doc_idx][begin:end] += token_label_logits
                doc_counts[doc_idx][begin:end] += 1

            if log_interval > 0 and (
                batch_idx == 1
                or batch_idx % log_interval == 0
                or batch_idx == total_batches
            ):
                progress = 100.0 * batch_idx / max(1, total_batches)
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                mean_bio_loss = (
                    float(np.mean(bio_losses)) if bio_losses else float("nan")
                )
                mean_label_loss = (
                    float(np.mean(label_losses)) if label_losses else float("nan")
                )
                print(
                    f"[eval:{split_name}] {batch_idx}/{total_batches} ({progress:.1f}%) "
                    f"running_loss={mean_loss:.4f} running_bio_loss={mean_bio_loss:.4f} "
                    f"running_entity_type_loss={mean_label_loss:.4f}",
                    flush=True,
                )
            if use_tqdm and losses:
                eval_bar.set_postfix(
                    {
                        "loss": f"{np.mean(losses):.4f}",
                        "bio": f"{(np.mean(bio_losses) if bio_losses else float('nan')):.4f}",
                        "label": f"{(np.mean(label_losses) if label_losses else float('nan')):.4f}",
                    }
                )
    if use_tqdm:
        eval_bar.close()

    typed_gold_all: list[int] = []
    typed_pred_all: list[int] = []
    bio_gold_all: list[int] = []
    bio_pred_all: list[int] = []
    entity_gold_all: list[int] = []
    entity_pred_all: list[int] = []

    true_spans_total = 0
    pred_spans_total = 0
    correct_spans_total = 0
    span_edit_total_ops = 0
    span_edit_add_ops = 0
    span_edit_edit_ops = 0
    span_edit_delete_ops = 0

    for doc_idx, gold_typed_ids in enumerate(dataset.doc_gold_typed_ids):
        counts = doc_counts[doc_idx]
        if np.any(counts == 0):
            missing = int(np.sum(counts == 0))
            raise RuntimeError(
                f"{missing} tokens received no predictions in doc index {doc_idx}"
            )

        avg_bio_logits = doc_bio_logits[doc_idx] / counts[:, None]
        avg_entity_logits = doc_entity_logits[doc_idx] / counts[:, None]
        pred_bio_ids = avg_bio_logits.argmax(axis=1).tolist()
        pred_entity_ids = avg_entity_logits.argmax(axis=1).tolist()

        gold_bio_ids = dataset.doc_gold_bio_ids[doc_idx]
        gold_entity_ids = dataset.doc_gold_entity_ids[doc_idx]
        bio_gold_all.extend(gold_bio_ids)
        bio_pred_all.extend(pred_bio_ids)

        pred_typed_ids: list[int] = []
        for pb, pe in zip(pred_bio_ids, pred_entity_ids):
            bio_tag = bio_id2label[int(pb)]
            if int(pb) == bio_o_id:
                pred_typed_ids.append(typed_label2id["O"])
            else:
                ent_name = entity_id2label[int(pe)]
                typed_tag = f"{bio_tag}-{ent_name}"
                pred_typed_ids.append(
                    typed_label2id.get(typed_tag, typed_label2id["O"])
                )

        typed_gold_all.extend(gold_typed_ids)
        typed_pred_all.extend(pred_typed_ids)

        for ge, pe in zip(gold_entity_ids, pred_entity_ids):
            if ge == -100:
                continue
            entity_gold_all.append(int(ge))
            entity_pred_all.append(int(pe))

        gold_spans = bio_tags_to_spans(gold_typed_ids, typed_id2label)
        pred_spans = bio_tags_to_spans(pred_typed_ids, typed_id2label)

        true_spans_total += len(gold_spans)
        pred_spans_total += len(pred_spans)
        correct_spans_total += len(gold_spans & pred_spans)

        if compute_span_edit_metrics:
            pred_char_spans = typed_ids_to_char_spans(
                tag_ids=pred_typed_ids,
                id2label=typed_id2label,
                offsets=dataset.doc_offsets[doc_idx],
                text=dataset.doc_texts[doc_idx],
            )
            span_edit_counts = evaluate_span_edits_counts(
                pred_spans=pred_char_spans,
                gold_spans=dataset.doc_gold_char_spans[doc_idx],
            )
            span_edit_total_ops += span_edit_counts["total_ops"]
            span_edit_add_ops += span_edit_counts["add_ops"]
            span_edit_edit_ops += span_edit_counts["edit_ops"]
            span_edit_delete_ops += span_edit_counts["delete_ops"]

    token_acc = accuracy_score(typed_gold_all, typed_pred_all)

    valid_ids = [i for i, name in typed_id2label.items() if name != "O"]
    if valid_ids:
        p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
            typed_gold_all,
            typed_pred_all,
            labels=valid_ids,
            average="micro",
            zero_division=0,
        )
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            typed_gold_all,
            typed_pred_all,
            labels=valid_ids,
            average="macro",
            zero_division=0,
        )
    else:
        p_micro = r_micro = f_micro = 0.0
        p_macro = r_macro = f_macro = 0.0

    bio_token_acc = accuracy_score(bio_gold_all, bio_pred_all)
    bio_valid_ids = [i for i, name in bio_id2label.items() if name != "O"]
    if bio_valid_ids:
        _, _, bio_f_macro, _ = precision_recall_fscore_support(
            bio_gold_all,
            bio_pred_all,
            labels=bio_valid_ids,
            average="macro",
            zero_division=0,
        )
    else:
        bio_f_macro = 0.0

    if entity_gold_all:
        label_token_acc = accuracy_score(entity_gold_all, entity_pred_all)
        entity_label_ids = sorted(entity_id2label.keys())
        _, _, label_f_macro, _ = precision_recall_fscore_support(
            entity_gold_all,
            entity_pred_all,
            labels=entity_label_ids,
            average="macro",
            zero_division=0,
        )
    else:
        label_token_acc = 0.0
        label_f_macro = 0.0

    entity_precision = (
        (correct_spans_total / pred_spans_total) if pred_spans_total > 0 else 0.0
    )
    entity_recall = (
        (correct_spans_total / true_spans_total) if true_spans_total > 0 else 0.0
    )
    if entity_precision + entity_recall > 0:
        entity_f1 = (
            2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        )
    else:
        entity_f1 = 0.0

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bio_loss": float(np.mean(bio_losses)) if bio_losses else float("nan"),
        "label_loss": float(np.mean(label_losses)) if label_losses else float("nan"),
        # Alias for clarity: second-stage entity-type loss (no third head).
        "entity_type_loss": float(np.mean(label_losses))
        if label_losses
        else float("nan"),
        "token_accuracy": float(token_acc),
        "token_precision_micro": float(p_micro),
        "token_recall_micro": float(r_micro),
        "token_f1_micro": float(f_micro),
        "token_precision_macro": float(p_macro),
        "token_recall_macro": float(r_macro),
        "token_f1_macro": float(f_macro),
        "bio_token_accuracy": float(bio_token_acc),
        "bio_token_f1_macro": float(bio_f_macro),
        "label_token_accuracy": float(label_token_acc),
        "label_token_f1_macro": float(label_f_macro),
        "entity_precision": float(entity_precision),
        "entity_recall": float(entity_recall),
        "entity_f1": float(entity_f1),
        "entity_spans_true": int(true_spans_total),
        "entity_spans_pred": int(pred_spans_total),
        "entity_spans_correct": int(correct_spans_total),
    }
    if compute_span_edit_metrics:
        metrics.update(
            {
                "span_edit_total_ops": int(span_edit_total_ops),
                "span_edit_add_ops": int(span_edit_add_ops),
                "span_edit_edit_ops": int(span_edit_edit_ops),
                "span_edit_delete_ops": int(span_edit_delete_ops),
            }
        )

    log_line = (
        f"[eval:{split_name}] done loss={metrics['loss']:.4f} "
        f"bio_loss={metrics['bio_loss']:.4f} entity_type_loss={metrics['entity_type_loss']:.4f} "
        f"token_f1_macro={metrics['token_f1_macro']:.4f} entity_f1={metrics['entity_f1']:.4f}"
    )
    if compute_span_edit_metrics:
        log_line += (
            f" span_edit_total={metrics['span_edit_total_ops']} "
            f"add={metrics['span_edit_add_ops']} "
            f"edit={metrics['span_edit_edit_ops']} "
            f"delete={metrics['span_edit_delete_ops']}"
        )
    print(
        log_line,
        flush=True,
    )
    return metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_accum_steps: int,
    scaler: torch.amp.GradScaler | None,
    use_fp16: bool,
    epoch: int,
    total_epochs: int,
    phase: str,
    log_interval: int,
    use_tqdm: bool,
) -> dict[str, float]:
    """
    Run one training epoch with gradient accumulation and optional fp16 AMP on CUDA.

    Returns epoch-average optimization losses.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    bio_loss_sum = 0.0
    label_loss_sum = 0.0
    steps = 0
    optimizer_steps = 0
    num_batches = len(loader)
    start_time = time.time()
    print(
        f"[train] epoch={epoch}/{total_epochs} phase={phase} start batches={num_batches} grad_accum={grad_accum_steps}",
        flush=True,
    )

    train_bar = tqdm(
        loader,
        total=num_batches,
        desc=f"train e{epoch}/{total_epochs} {phase}",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not use_tqdm,
    )
    for step_idx, batch in enumerate(train_bar, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels_bio = batch["labels_bio"].to(device, non_blocking=True)
        labels_entity = batch["labels_entity"].to(device, non_blocking=True)

        if scaler is not None and use_fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_bio=labels_bio,
                    labels_entity=labels_entity,
                )
                if out.loss is None:
                    raise RuntimeError("Training received no loss from model output")
                loss = out.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_bio=labels_bio,
                labels_entity=labels_entity,
            )
            if out.loss is None:
                raise RuntimeError("Training received no loss from model output")
            loss = out.loss / grad_accum_steps
            loss.backward()

        loss_sum += float(out.loss.detach().cpu().item())
        if out.bio_loss is not None:
            bio_loss_sum += float(out.bio_loss.detach().cpu().item())
        if out.label_loss is not None:
            label_loss_sum += float(out.label_loss.detach().cpu().item())
        steps += 1

        if step_idx % grad_accum_steps == 0:
            if scaler is not None and use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if log_interval > 0 and (
            step_idx == 1 or step_idx % log_interval == 0 or step_idx == num_batches
        ):
            progress = 100.0 * step_idx / max(1, num_batches)
            mean_loss = loss_sum / max(1, steps)
            mean_bio_loss = bio_loss_sum / max(1, steps)
            mean_label_loss = label_loss_sum / max(1, steps)
            batch_loss = float(out.loss.detach().cpu().item())
            lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
            elapsed = time.time() - start_time
            print(
                f"[train] epoch={epoch}/{total_epochs} phase={phase} "
                f"batch={step_idx}/{num_batches} ({progress:.1f}%) "
                f"batch_loss={batch_loss:.4f} running_loss={mean_loss:.4f} "
                f"running_bio_loss={mean_bio_loss:.4f} running_entity_type_loss={mean_label_loss:.4f} "
                f"opt_steps={optimizer_steps} lrs={lrs} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if use_tqdm:
            train_bar.set_postfix(
                {
                    "loss": f"{(loss_sum / max(1, steps)):.4f}",
                    "bio": f"{(bio_loss_sum / max(1, steps)):.4f}",
                    "label": f"{(label_loss_sum / max(1, steps)):.4f}",
                    "opt": optimizer_steps,
                }
            )

    remainder = steps % grad_accum_steps
    if remainder != 0:
        if scaler is not None and use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if use_tqdm:
        train_bar.close()
    mean_loss = loss_sum / max(1, steps)
    mean_bio_loss = bio_loss_sum / max(1, steps)
    mean_label_loss = label_loss_sum / max(1, steps)
    total_time = time.time() - start_time
    print(
        f"[train] epoch={epoch}/{total_epochs} phase={phase} done "
        f"mean_loss={mean_loss:.4f} mean_bio_loss={mean_bio_loss:.4f} "
        f"mean_entity_type_loss={mean_label_loss:.4f} batches={steps} "
        f"opt_steps={optimizer_steps} elapsed={total_time:.1f}s",
        flush=True,
    )
    return {
        "loss": float(mean_loss),
        "bio_loss": float(mean_bio_loss),
        "label_loss": float(mean_label_loss),
        # Alias for clarity: second-stage entity-type loss (no third head).
        "entity_type_loss": float(mean_label_loss),
    }


def metric_for_selection(metrics: dict[str, Any], metric_name: str) -> float:
    """Map a metrics dict to a single score where larger is always better."""
    if metric_name == "val_loss":
        return -float(metrics["loss"])
    return float(metrics[metric_name])


def save_checkpoint(
    path: Path, model, args: argparse.Namespace, epoch: int, metrics: dict[str, Any]
) -> None:
    """Serialize model state, epoch, metrics, and run configuration to disk."""
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def plot_history(history: list[dict[str, Any]], plot_dir: Path) -> None:
    """Save readable loss and task-specific F1 curves as PNG and vector PDF."""
    if not history:
        return

    # Plotting is optional; training should not fail due to matplotlib/numpy conflicts.
    try:
        import matplotlib.pyplot as plt
        from matplotlib import ticker
        from meddeid_eval.plotting import clean_axes, plotting_style, save_figure
    except Exception as exc:  # noqa: BLE001 - diagnostics must never abort training
        print(f"[plots] skipped: plotting support unavailable ({exc})", flush=True)
        return
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    tr_loss = [row["train_loss"] for row in history]
    tr_eval_loss = [row.get("train_eval_loss", float("nan")) for row in history]
    val_loss = [row["val_loss"] for row in history]

    train_bio_f1 = [
        row.get("train_bio_token_f1_macro", float("nan")) for row in history
    ]
    val_bio_f1 = [row.get("val_bio_token_f1_macro", float("nan")) for row in history]
    train_label_f1 = [
        row.get("train_label_token_f1_macro", float("nan")) for row in history
    ]
    val_label_f1 = [
        row.get("val_label_token_f1_macro", float("nan")) for row in history
    ]
    train_entity_f1 = [row.get("train_entity_f1", float("nan")) for row in history]
    val_entity_f1 = [row.get("val_entity_f1", float("nan")) for row in history]

    with plotting_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
        ax.plot(
            epochs,
            tr_loss,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color="#2C7593",
            label="Training objective",
        )
        ax.plot(
            epochs,
            tr_eval_loss,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color="#56B4E9",
            label="Training-set evaluation",
        )
        ax.plot(
            epochs,
            val_loss,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color="#E67924",
            label="Validation",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (lower is better)")
        ax.set_title("Training and validation loss", loc="left", pad=10)
        clean_axes(ax, grid_axis="both")
        ax.legend(frameon=False)
        save_figure(fig, plot_dir, "loss_curve", dpi=300)

        fig, axes = plt.subplots(
            ncols=3,
            figsize=(10.2, 3.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        panels = [
            ("BIO token macro F1", train_bio_f1, val_bio_f1),
            ("Entity-label token macro F1", train_label_f1, val_label_f1),
            ("Entity span F1", train_entity_f1, val_entity_f1),
        ]
        for index, (title, training_values, validation_values) in enumerate(panels):
            ax = axes[index]
            ax.plot(
                epochs,
                training_values,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                color="#2C7593",
                label="Training",
            )
            ax.plot(
                epochs,
                validation_values,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                color="#E67924",
                label="Validation",
            )
            ax.set_title(title, loc="left", fontsize=10, pad=8)
            ax.set_xlabel("Epoch")
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylim(0, 1.02)
            clean_axes(ax, grid_axis="both")
        axes[0].set_ylabel("F1 (higher is better)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        fig.suptitle("Training and validation F1", fontsize=12, fontweight="bold")
        save_figure(fig, plot_dir, "f1_curve", dpi=300)


def main() -> None:
    """
    End-to-end training entrypoint.

    Loads data, builds labels/tokenizer/model, runs warmup + finetuning, evaluates,
    and applies early stopping on the selected validation metric.
    """
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if args.early_stopping_min_epochs <= 0:
        raise ValueError("early_stopping_min_epochs must be > 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if args.final_epoch_is_best and args.early_stopping_patience > 0:
        print(
            "[init] --final-epoch-is-best set: forcing early stopping off "
            "(the final epoch is the deliverable, so there is nothing to select).",
            flush=True,
        )
        args.early_stopping_patience = 0
    early_stopping_start_epoch = min(args.early_stopping_min_epochs, args.epochs)
    initialization_mode = (
        "base_encoder" if args.from_base_encoder else "existing_model"
    )
    print(
        f"[init] model={args.model_name} initialization={initialization_mode} "
        f"data_dir={args.data_dir} output_dir={args.output_dir} ckpt_dir={args.ckpt_dir}",
        flush=True,
    )
    print(
        f"[init] epochs={args.epochs} head_warmup_epochs={args.head_warmup_epochs} optimizer={args.optimizer} early_stopping_patience={args.early_stopping_patience} early_stopping_min_delta={args.early_stopping_min_delta} early_stopping_start_epoch={early_stopping_start_epoch} train_bs={args.train_batch_size} eval_bs={args.eval_batch_size} grad_accum={args.grad_accum_steps} max_length={args.max_length} overlap={args.overlap}",
        flush=True,
    )
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    test_rows = (
        []
        if args.skip_test_evaluation
        else read_jsonl(data_dir / "test.jsonl")
    )
    print(
        f"[data] docs train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )
    check_split_disjointness(
        train_rows,
        val_rows,
        test_rows,
        early_stopping_enabled=args.early_stopping_patience > 0,
    )

    all_rows = train_rows + val_rows + test_rows
    entity_types = resolve_entity_types(args, all_rows)
    if not entity_types:
        raise ValueError(
            "Empty entity label schema; set --entity-labels or CANONICAL_ENTITY_TYPES."
        )

    # Lock the head to the canonical schema. Without an explicit --entity-labels
    # override the head MUST equal meddeid_core.BERT_ENTITY_LABELS, or trained
    # checkpoints drift out of interchangeability (this is exactly how the retired
    # Organization:Patient label produced a 15-label head). Fail loudly at the start.
    canonical_head = list(BERT_ENTITY_LABELS)
    if not (getattr(args, "entity_labels", "") or "").strip():
        if entity_types != canonical_head:
            raise RuntimeError(
                "Resolved entity head does not match meddeid_core.BERT_ENTITY_LABELS "
                f"({len(entity_types)} vs {len(canonical_head)} labels). Refusing to "
                "train a non-canonical head; fix CANONICAL_ENTITY_TYPES/meddeid_core, or "
                "pass --entity-labels explicitly if this is intentional."
            )
    elif entity_types != canonical_head:
        print(
            "[labels] WARNING: --entity-labels differs from canonical "
            "meddeid_core.BERT_ENTITY_LABELS; this checkpoint will NOT be "
            "interchangeable with canonical models.",
            flush=True,
        )

    typed_labels = ["O"]
    for ent in entity_types:
        typed_labels.append(f"B-{ent}")
        typed_labels.append(f"I-{ent}")

    typed_label2id = {name: i for i, name in enumerate(typed_labels)}
    typed_id2label = {i: name for name, i in typed_label2id.items()}
    bio_labels = ["O", "B", "I"]
    bio_label2id = {name: i for i, name in enumerate(bio_labels)}
    bio_id2label = {i: name for name, i in bio_label2id.items()}
    entity_label2id = {name: i for i, name in enumerate(entity_types)}
    entity_id2label = {i: name for name, i in entity_label2id.items()}
    print(
        f"[data] discovered entity types={len(entity_types)} typed_labels={len(typed_labels)}",
        flush=True,
    )

    device = select_device(args.device)
    attn_implementation = args.attn_implementation
    if attn_implementation == "auto":
        attn_implementation = "eager" if device.type == "mps" else None
    initialization = resolve_model_initialization(
        args.model_name,
        from_base_encoder=args.from_base_encoder,
        model_revision=args.model_revision,
    )
    if initialization.mode == "existing_model":
        if initialization.bio_labels != tuple(bio_labels):
            raise ValueError(
                "initial model BIO labels do not match the training head contract"
            )
        if initialization.entity_labels != tuple(entity_types):
            raise ValueError(
                "initial model entity labels do not match the configured training head"
            )
        if initialization.profile_id != args.language_profile:
            raise ValueError(
                "initial model language profile does not match --language-profile"
            )
        if initialization.profile_version != args.language_profile_version:
            raise ValueError(
                "initial model language profile version does not match "
                "--language-profile-version"
            )
    args.initialization_mode = initialization.mode
    args.base_encoder = initialization.base_encoder
    args.base_revision = initialization.base_revision
    args.initial_model_revision = initialization.initial_model_revision
    tokenizer = load_tokenizer(
        initialization.tokenizer_source,
        revision=initialization.transformer_revision,
        local_files_only=initialization.local_files_only,
    )
    print("[model] tokenizer loaded", flush=True)
    model = DualHeadTokenClassifier(
        model_name=initialization.encoder_source,
        num_bio_labels=len(bio_labels),
        num_entity_labels=len(entity_types),
        attn_implementation=attn_implementation,
        initialize_from_pretrained=initialization.initialize_encoder_from_pretrained,
        revision=initialization.transformer_revision,
        local_files_only=initialization.local_files_only,
    )
    if initialization.state_dict is not None:
        model.load_state_dict(initialization.state_dict, strict=True)
        print("[model] existing encoder and classifier heads loaded", flush=True)
    else:
        print("[model] base encoder loaded; classifier heads initialized", flush=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[model] gradient checkpointing enabled", flush=True)

    train_dataset = WindowedTokenDataset(
        records=train_rows,
        tokenizer=tokenizer,
        typed_label2id=typed_label2id,
        bio_label2id=bio_label2id,
        entity_label2id=entity_label2id,
        label_key=args.label_key,
        max_length=args.max_length,
        overlap=args.overlap,
    )
    val_dataset = WindowedTokenDataset(
        records=val_rows,
        tokenizer=tokenizer,
        typed_label2id=typed_label2id,
        bio_label2id=bio_label2id,
        entity_label2id=entity_label2id,
        label_key=args.label_key,
        max_length=args.max_length,
        overlap=args.overlap,
    )
    test_dataset = WindowedTokenDataset(
        records=test_rows,
        tokenizer=tokenizer,
        typed_label2id=typed_label2id,
        bio_label2id=bio_label2id,
        entity_label2id=entity_label2id,
        label_key=args.label_key,
        max_length=args.max_length,
        overlap=args.overlap,
    )
    print(
        f"[data] windows train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )

    collator = TokenCollator(tokenizer)
    dataloader_num_workers = max(0, int(args.dataloader_num_workers))
    dataloader_prefetch_factor = max(2, int(args.dataloader_prefetch_factor))
    pin_memory = bool(device.type == "cuda" and not args.disable_pin_memory)
    persistent_workers = bool(
        dataloader_num_workers > 0 and not args.disable_persistent_workers
    )
    loader_kwargs: dict[str, Any] = {
        "collate_fn": collator,
        "num_workers": dataloader_num_workers,
        "pin_memory": pin_memory,
    }
    if dataloader_num_workers > 0:
        loader_kwargs["prefetch_factor"] = dataloader_prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model.to(device)
    print(f"[device] using {device}", flush=True)
    print(
        "[data] loader "
        f"num_workers={dataloader_num_workers} "
        f"pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers} "
        f"prefetch_factor={(dataloader_prefetch_factor if dataloader_num_workers > 0 else 'n/a')}",
        flush=True,
    )

    use_fp16 = bool(args.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    print(f"[device] fp16_enabled={use_fp16}", flush=True)

    warmup_epochs = min(args.head_warmup_epochs, args.epochs)
    finetune_epochs = args.epochs - warmup_epochs

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    print(
        f"[train] planned warmup_epochs={warmup_epochs} finetune_epochs={finetune_epochs} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )

    history = []
    best_payload = None
    best_score = -float("inf")
    global_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    early_stop_reason = None

    if warmup_epochs > 0:
        set_backbone_trainable(model, False)
        optimizer = make_optimizer(
            model,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
        )
        warmup_total_steps = warmup_epochs * steps_per_epoch
        warmup_warmup_steps = int(args.warmup_ratio * warmup_total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_warmup_steps, max(1, warmup_total_steps)
        )

        for _ in range(warmup_epochs):
            global_epoch += 1
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                scaler=scaler,
                use_fp16=use_fp16,
                epoch=global_epoch,
                total_epochs=args.epochs,
                phase="head_warmup",
                log_interval=args.log_interval,
                use_tqdm=not args.disable_tqdm,
            )
            train_eval_metrics = evaluate(
                model,
                train_eval_loader,
                train_dataset,
                typed_id2label=typed_id2label,
                typed_label2id=typed_label2id,
                bio_id2label=bio_id2label,
                bio_label2id=bio_label2id,
                entity_id2label=entity_id2label,
                device=device,
                split_name="train",
                log_interval=0,
                use_tqdm=not args.disable_tqdm,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                val_dataset,
                typed_id2label=typed_id2label,
                typed_label2id=typed_label2id,
                bio_id2label=bio_id2label,
                bio_label2id=bio_label2id,
                entity_id2label=entity_id2label,
                device=device,
                split_name="val",
                log_interval=args.log_interval,
                use_tqdm=not args.disable_tqdm,
            )
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": "head_warmup",
                    "train_loss": train_stats["loss"],
                    "train_bio_loss": train_stats["bio_loss"],
                    "train_label_loss": train_stats["label_loss"],
                    "train_entity_type_loss": train_stats["entity_type_loss"],
                    "train_eval_loss": train_eval_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_bio_token_f1_macro": train_eval_metrics[
                        "bio_token_f1_macro"
                    ],
                    "val_bio_token_f1_macro": val_metrics["bio_token_f1_macro"],
                    "train_label_token_f1_macro": train_eval_metrics[
                        "label_token_f1_macro"
                    ],
                    "val_label_token_f1_macro": val_metrics["label_token_f1_macro"],
                    "train_entity_f1": train_eval_metrics["entity_f1"],
                    "val_entity_f1": val_metrics["entity_f1"],
                    "train_token_f1_macro": train_eval_metrics["token_f1_macro"],
                    "val_token_f1_macro": val_metrics["token_f1_macro"],
                }
            )

            score = metric_for_selection(val_metrics, args.save_best_metric)
            score_is_finite = math.isfinite(score)
            if not score_is_finite:
                print(
                    f"[warn] non-finite score for metric {args.save_best_metric} at epoch {global_epoch}: {score}",
                    flush=True,
                )
            improved = False
            if not args.final_epoch_is_best:
                if best_payload is None:
                    improved = True
                elif score_is_finite and score > (best_score + args.early_stopping_min_delta):
                    improved = True
            if improved:
                if score_is_finite:
                    best_score = score
                epochs_without_improvement = 0
                best_payload = {
                    "epoch": global_epoch,
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "val_metrics": val_metrics,
                }
                save_checkpoint(
                    ckpt_dir / "best.pt", model, args, global_epoch, val_metrics
                )
            elif global_epoch >= early_stopping_start_epoch:
                epochs_without_improvement += 1

            print(
                f"[epoch {global_epoch}/{args.epochs}] phase=head_warmup "
                f"train_loss={train_stats['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"train_bio_f1={train_eval_metrics['bio_token_f1_macro']:.4f} "
                f"val_bio_f1={val_metrics['bio_token_f1_macro']:.4f} "
                f"train_label_f1={train_eval_metrics['label_token_f1_macro']:.4f} "
                f"val_label_f1={val_metrics['label_token_f1_macro']:.4f} "
                f"train_entity_f1={train_eval_metrics['entity_f1']:.4f} "
                f"val_entity_f1={val_metrics['entity_f1']:.4f} "
                f"no_improve_count={epochs_without_improvement}"
            )
            if (
                args.early_stopping_patience > 0
                and global_epoch >= early_stopping_start_epoch
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                early_stop_reason = (
                    f"No improvement in {args.save_best_metric} for "
                    f"{epochs_without_improvement} consecutive epoch(s) "
                    f"(starting from epoch {early_stopping_start_epoch})."
                )
                print(
                    f"[early-stop] triggered at epoch {global_epoch}/{args.epochs}: {early_stop_reason}",
                    flush=True,
                )
                break

    if finetune_epochs > 0 and not stopped_early:
        empty_device_cache(device)
        set_backbone_trainable(model, True)
        optimizer = make_optimizer(
            model,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
        )
        ft_total_steps = finetune_epochs * steps_per_epoch
        ft_warmup_steps = int(args.warmup_ratio * ft_total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, ft_warmup_steps, max(1, ft_total_steps)
        )

        for _ in range(finetune_epochs):
            global_epoch += 1
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                scaler=scaler,
                use_fp16=use_fp16,
                epoch=global_epoch,
                total_epochs=args.epochs,
                phase="full_finetune",
                log_interval=args.log_interval,
                use_tqdm=not args.disable_tqdm,
            )
            train_eval_metrics = evaluate(
                model,
                train_eval_loader,
                train_dataset,
                typed_id2label=typed_id2label,
                typed_label2id=typed_label2id,
                bio_id2label=bio_id2label,
                bio_label2id=bio_label2id,
                entity_id2label=entity_id2label,
                device=device,
                split_name="train",
                log_interval=0,
                use_tqdm=not args.disable_tqdm,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                val_dataset,
                typed_id2label=typed_id2label,
                typed_label2id=typed_label2id,
                bio_id2label=bio_id2label,
                bio_label2id=bio_label2id,
                entity_id2label=entity_id2label,
                device=device,
                split_name="val",
                log_interval=args.log_interval,
                use_tqdm=not args.disable_tqdm,
            )
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": "full_finetune",
                    "train_loss": train_stats["loss"],
                    "train_bio_loss": train_stats["bio_loss"],
                    "train_label_loss": train_stats["label_loss"],
                    "train_entity_type_loss": train_stats["entity_type_loss"],
                    "train_eval_loss": train_eval_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_bio_token_f1_macro": train_eval_metrics[
                        "bio_token_f1_macro"
                    ],
                    "val_bio_token_f1_macro": val_metrics["bio_token_f1_macro"],
                    "train_label_token_f1_macro": train_eval_metrics[
                        "label_token_f1_macro"
                    ],
                    "val_label_token_f1_macro": val_metrics["label_token_f1_macro"],
                    "train_entity_f1": train_eval_metrics["entity_f1"],
                    "val_entity_f1": val_metrics["entity_f1"],
                    "train_token_f1_macro": train_eval_metrics["token_f1_macro"],
                    "val_token_f1_macro": val_metrics["token_f1_macro"],
                }
            )

            score = metric_for_selection(val_metrics, args.save_best_metric)
            score_is_finite = math.isfinite(score)
            if not score_is_finite:
                print(
                    f"[warn] non-finite score for metric {args.save_best_metric} at epoch {global_epoch}: {score}",
                    flush=True,
                )
            improved = False
            if not args.final_epoch_is_best:
                if best_payload is None:
                    improved = True
                elif score_is_finite and score > (best_score + args.early_stopping_min_delta):
                    improved = True
            if improved:
                if score_is_finite:
                    best_score = score
                epochs_without_improvement = 0
                best_payload = {
                    "epoch": global_epoch,
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "val_metrics": val_metrics,
                }
                save_checkpoint(
                    ckpt_dir / "best.pt", model, args, global_epoch, val_metrics
                )
            elif global_epoch >= early_stopping_start_epoch:
                epochs_without_improvement += 1

            print(
                f"[epoch {global_epoch}/{args.epochs}] phase=full_finetune "
                f"train_loss={train_stats['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"train_bio_f1={train_eval_metrics['bio_token_f1_macro']:.4f} "
                f"val_bio_f1={val_metrics['bio_token_f1_macro']:.4f} "
                f"train_label_f1={train_eval_metrics['label_token_f1_macro']:.4f} "
                f"val_label_f1={val_metrics['label_token_f1_macro']:.4f} "
                f"train_entity_f1={train_eval_metrics['entity_f1']:.4f} "
                f"val_entity_f1={val_metrics['entity_f1']:.4f} "
                f"no_improve_count={epochs_without_improvement}"
            )
            if (
                args.early_stopping_patience > 0
                and global_epoch >= early_stopping_start_epoch
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                early_stop_reason = (
                    f"No improvement in {args.save_best_metric} for "
                    f"{epochs_without_improvement} consecutive epoch(s) "
                    f"(starting from epoch {early_stopping_start_epoch})."
                )
                print(
                    f"[early-stop] triggered at epoch {global_epoch}/{args.epochs}: {early_stop_reason}",
                    flush=True,
                )
                break

    final_train_metrics = evaluate(
        model,
        train_eval_loader,
        train_dataset,
        typed_id2label=typed_id2label,
        typed_label2id=typed_label2id,
        bio_id2label=bio_id2label,
        bio_label2id=bio_label2id,
        entity_id2label=entity_id2label,
        device=device,
        split_name="train_final",
        log_interval=0,
        use_tqdm=not args.disable_tqdm,
        compute_span_edit_metrics=True,
    )
    final_val_metrics = evaluate(
        model,
        val_loader,
        val_dataset,
        typed_id2label=typed_id2label,
        typed_label2id=typed_label2id,
        bio_id2label=bio_id2label,
        bio_label2id=bio_label2id,
        entity_id2label=entity_id2label,
        device=device,
        split_name="val_final",
        log_interval=args.log_interval,
        use_tqdm=not args.disable_tqdm,
        compute_span_edit_metrics=True,
    )
    final_test_metrics = None
    if not args.skip_test_evaluation:
        final_test_metrics = evaluate(
            model,
            test_loader,
            test_dataset,
            typed_id2label=typed_id2label,
            typed_label2id=typed_label2id,
            bio_id2label=bio_id2label,
            bio_label2id=bio_label2id,
            entity_id2label=entity_id2label,
            device=device,
            split_name="test_final",
            log_interval=args.log_interval,
            use_tqdm=not args.disable_tqdm,
            compute_span_edit_metrics=True,
        )

    if not args.final_epoch_is_best:
        save_checkpoint(
            ckpt_dir / "last.pt", model, args, global_epoch, final_val_metrics
        )

    if args.final_epoch_is_best:
        # Refit mode: the final-epoch model IS the deliverable, so emit a SINGLE checkpoint
        # -- best.pt only. We deliberately do NOT also write last.pt (it would be a
        # byte-identical duplicate and the only source of best/last ambiguity). The reported
        # best_* / test metrics equal the final-epoch metrics, so best.pt and the `test`
        # field are unambiguously the model and score to report.
        best_payload = {
            "epoch": global_epoch,
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "val_metrics": final_val_metrics,
        }
        save_checkpoint(
            ckpt_dir / "best.pt", model, args, global_epoch, final_val_metrics
        )
        final_selection_score = metric_for_selection(
            final_val_metrics, args.save_best_metric
        )
        if math.isfinite(final_selection_score):
            best_score = final_selection_score

    if best_payload is not None:
        model.load_state_dict(best_payload["state_dict"])
    best_train_metrics = evaluate(
        model,
        train_eval_loader,
        train_dataset,
        typed_id2label=typed_id2label,
        typed_label2id=typed_label2id,
        bio_id2label=bio_id2label,
        bio_label2id=bio_label2id,
        entity_id2label=entity_id2label,
        device=device,
        split_name="train_best",
        log_interval=0,
        use_tqdm=not args.disable_tqdm,
    )
    best_test_metrics = final_test_metrics if args.final_epoch_is_best else None
    if not args.skip_test_evaluation and not args.final_epoch_is_best:
        best_test_metrics = evaluate(
            model,
            test_loader,
            test_dataset,
            typed_id2label=typed_id2label,
            typed_label2id=typed_label2id,
            bio_id2label=bio_id2label,
            bio_label2id=bio_label2id,
            entity_id2label=entity_id2label,
            device=device,
            split_name="test_best",
            log_interval=args.log_interval,
            use_tqdm=not args.disable_tqdm,
        )

    train_metrics = {
        "model_name": args.model_name,
        "device": str(device),
        "fp16": use_fp16,
        "num_typed_labels": len(typed_labels),
        "num_bio_labels": len(bio_labels),
        "num_entity_labels": len(entity_types),
        "labels": typed_labels,
        "typed_labels": typed_labels,
        "bio_labels": bio_labels,
        "entity_labels": entity_types,
        "epochs": args.epochs,
        "epochs_requested": args.epochs,
        "epochs_completed": global_epoch,
        "head_warmup_epochs": warmup_epochs,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "split_docs": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "history": history,
        "best_metric": args.save_best_metric,
        "best_score": best_score,
        "best_epoch": best_payload["epoch"] if best_payload is not None else None,
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "min_epochs": args.early_stopping_min_epochs,
            "start_epoch": early_stopping_start_epoch,
            "triggered": stopped_early,
            "reason": early_stop_reason,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "final_train": final_train_metrics,
        "final_val": final_val_metrics,
        "final_test": final_test_metrics,
        "best_train": best_train_metrics,
        "best_test": best_test_metrics,
        "config": vars(args),
    }

    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    eval_payload = {
        "best_epoch": best_payload["epoch"] if best_payload is not None else None,
        "selection_metric": args.save_best_metric,
        "train": best_train_metrics,
        "validation": (
            best_payload["val_metrics"]
            if best_payload is not None
            else final_val_metrics
        ),
        "test": best_test_metrics,
        "final": {
            "train": final_train_metrics,
            "validation": final_val_metrics,
            "test": final_test_metrics,
        },
    }

    with (output_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    plot_history(history=history, plot_dir=plots_dir)

    print(f"[train] wrote {(output_dir / 'train_metrics.json')}")
    print(f"[eval] wrote {(output_dir / 'eval.json')}")
    print(f"[plots] wrote {plots_dir}")
    if args.final_epoch_is_best:
        print(f"[ckpt] wrote {(ckpt_dir / 'best.pt')} (single refit checkpoint)")
    else:
        print(f"[ckpt] wrote {(ckpt_dir / 'best.pt')} and {(ckpt_dir / 'last.pt')}")


if __name__ == "__main__":
    main()
