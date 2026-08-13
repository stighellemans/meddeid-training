from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from . import DEFAULT_INITIAL_MODEL
from .export import export_bundle


def _append_value(command: list[str], flag: str, config: dict, key: str, default) -> None:
    command.extend([flag, str(config.get(key, default))])


def training_command(
    config: dict,
    data: Path,
    run: Path,
    *,
    mode: str,
    epochs: int | None = None,
) -> list[str]:
    if mode not in {"fit", "select-epochs", "refit"}:
        raise ValueError(f"unsupported training mode: {mode}")
    command = [
        sys.executable,
        "-m",
        "meddeid_training.train_script",
        "--data-dir", str(data.resolve()),
        "--output-dir", str(run.resolve()),
        "--ckpt-dir", str((run / "checkpoints").resolve()),
        "--model-name", str(config.get("model_name", DEFAULT_INITIAL_MODEL)),
        "--language-profile", str(config["language_profile"]),
        "--language-profile-version", str(config["language_profile_version"]),
        "--max-length", str(config.get("max_length", 512)),
        "--overlap", str(config.get("overlap", 64)),
        "--train-batch-size", str(config.get("train_batch_size", 8)),
        "--eval-batch-size", str(config.get("eval_batch_size", 8)),
        "--grad-accum-steps", str(config.get("grad_accum_steps", 2)),
        "--epochs", str(epochs if epochs is not None else config.get("epochs", 8)),
        "--optimizer", str(config.get("optimizer", "adamw")),
        "--seed", str(config.get("seed", 42)),
        "--label-key", str(config.get("label_key", "label")),
        "--device", str(config.get("device", "auto")),
        "--attn-implementation", str(config.get("attn_implementation", "auto")),
    ]
    if config.get("model_revision"):
        command.extend(["--model-revision", str(config["model_revision"])])
    _append_value(command, "--head-warmup-epochs", config, "head_warmup_epochs", 0)
    _append_value(command, "--encoder-lr", config, "encoder_lr", 2e-5)
    _append_value(command, "--head-lr", config, "head_lr", 1e-4)
    _append_value(command, "--weight-decay", config, "weight_decay", 0.01)
    _append_value(command, "--warmup-ratio", config, "warmup_ratio", 0.1)
    _append_value(command, "--save-best-metric", config, "save_best_metric", "entity_f1")
    _append_value(
        command,
        "--early-stopping-min-delta",
        config,
        "early_stopping_min_delta",
        0.001,
    )
    _append_value(
        command,
        "--early-stopping-min-epochs",
        config,
        "early_stopping_min_epochs",
        3,
    )
    _append_value(
        command,
        "--dataloader-num-workers",
        config,
        "dataloader_num_workers",
        4,
    )
    _append_value(
        command,
        "--dataloader-prefetch-factor",
        config,
        "dataloader_prefetch_factor",
        4,
    )
    if config.get("fp16", False):
        command.append("--fp16")
    if config.get("gradient_checkpointing", False):
        command.append("--gradient-checkpointing")
    if config.get("from_base_encoder", False):
        command.append("--from-base-encoder")
    if config.get("disable_pin_memory", False):
        command.append("--disable-pin-memory")
    if config.get("disable_persistent_workers", False):
        command.append("--disable-persistent-workers")
    if config.get("disable_tqdm", False):
        command.append("--disable-tqdm")

    if mode in {"fit", "select-epochs"}:
        _append_value(
            command,
            "--early-stopping-patience",
            config,
            "early_stopping_patience",
            3,
        )
        if mode == "select-epochs":
            command.append("--skip-test-evaluation")
    else:
        command.extend(["--early-stopping-patience", "0", "--final-epoch-is-best"])
    return command


def selected_epochs(metrics: dict) -> int:
    """Return the trainer's one-based best epoch without applying an offset."""
    value = metrics.get("best_epoch")
    if value is None:
        raise ValueError("training metrics do not contain best_epoch")
    selected = int(value)
    if selected < 1:
        raise ValueError("best_epoch must be one-based and positive")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meddeid-train",
        description=(
            "Fit a MedDeID model once, or run the stricter epoch-selection and "
            "full-data-refit publication protocol."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    training_help = {
        "fit": "run one ordinary train/validation/test fit",
        "select-epochs": "select an epoch count without reading the test set",
        "refit": "restart and fit all development data for the selected epochs",
    }
    for name in ("fit", "select-epochs", "refit"):
        command = sub.add_parser(name, help=training_help[name])
        command.add_argument("--config", required=True)
        command.add_argument("--data", required=True)
        command.add_argument("--run", required=True)
        if name == "refit":
            command.add_argument("--selection")
        elif name == "select-epochs":
            command.add_argument("--selection-output")
        if name == "fit":
            command.add_argument(
                "--epochs",
                type=int,
                help="override the maximum epoch count from the configuration",
            )
        command.add_argument("--dry-run", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--run-metadata")
    export.add_argument("--base-encoder")
    export.add_argument("--base-revision")
    export.add_argument("--unsafe-override", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "export":
        print(export_bundle(
            args.checkpoint,
            args.output,
            run_metadata=args.run_metadata,
            base_encoder=args.base_encoder,
            base_revision=args.base_revision,
            unsafe_override=args.unsafe_override,
        ))
        return 0

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    epochs = args.epochs if args.command == "fit" else None
    if args.command == "refit":
        if not args.selection:
            parser.error("refit requires --selection")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        epochs = int(selection["selected_epochs"])
    command = training_command(
        config,
        Path(args.data),
        Path(args.run),
        mode=args.command,
        epochs=epochs,
    )
    if args.dry_run:
        print(json.dumps(command))
        return 0
    subprocess.run(command, check=True)
    metrics = json.loads((Path(args.run) / "train_metrics.json").read_text(encoding="utf-8"))
    if args.command == "select-epochs":
        selected = selected_epochs(metrics)
        output = Path(args.selection_output or (Path(args.run) / "run.json"))
        output.write_text(json.dumps({"selected_epochs": selected, "selection_metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    return 0
