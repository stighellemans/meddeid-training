#!/usr/bin/env python3
"""Point each English benchmark battery at one completed MedDeID checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


MODEL_ID = "meddeid-english-synth"
BATTERIES = ("meddeid-english-synthetic", "asq-phi", "technetium-i")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_value(path: Path, engine_root: Path) -> str:
    return Path(os.path.relpath(path, engine_root)).as_posix()


def _replace_model_block(text: str, values: dict[str, str], config: Path) -> str:
    lines = text.splitlines(keepends=True)
    marker = f"  - id: {MODEL_ID}"
    starts = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == marker]
    if len(starts) != 1:
        raise ValueError(f"expected one {MODEL_ID!r} model block in {config}, found {len(starts)}")

    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].rstrip("\r\n")
        if stripped.startswith("  - id: ") or (stripped and not stripped.startswith(" ")):
            end = index
            break

    for key, value in values.items():
        prefix = f"      {key}:"
        matches = [index for index in range(start, end) if lines[index].startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {key!r} field in the {MODEL_ID!r} block of {config}, "
                f"found {len(matches)}"
            )
        newline = "\r\n" if lines[matches[0]].endswith("\r\n") else "\n"
        lines[matches[0]] = f"{prefix} {value}{newline}"
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-metrics", required=True, type=Path)
    parser.add_argument("--benchmarks-root", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    train_metrics = args.train_metrics.expanduser().resolve()
    benchmarks_root = args.benchmarks_root.expanduser().resolve()
    engine_root = benchmarks_root.parent / "deid-battery"
    for required in (checkpoint, train_metrics):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not engine_root.is_dir():
        raise FileNotFoundError(engine_root)

    values = {
        "checkpoint": _config_value(checkpoint, engine_root),
        "checkpoint_sha256": _sha256(checkpoint),
        "train_metrics": _config_value(train_metrics, engine_root),
    }
    for battery in BATTERIES:
        config = benchmarks_root / "batteries" / battery / "config.yaml"
        original = config.read_text(encoding="utf-8")
        updated = _replace_model_block(original, values, config)
        config.write_text(updated, encoding="utf-8")
        print(f"updated {config}")
    print(f"checkpoint sha256: {values['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
