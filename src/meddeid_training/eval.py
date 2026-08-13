#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def checkpoint_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"checkpoint_read_error": str(exc)}
    if not isinstance(payload, dict):
        return {}
    out = {}
    if payload.get("epoch") is not None:
        out["checkpoint_epoch"] = payload.get("epoch")
    if isinstance(payload.get("metrics"), dict):
        out["checkpoint_metrics"] = payload["metrics"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_eval = {}
    if out_path.exists():
        try:
            existing_eval = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_eval = {}

    train_metrics_path = out_path.parent / "train_metrics.json"
    metrics_from_train = {}
    metadata_from_checkpoint = checkpoint_metadata(model_path)
    if train_metrics_path.exists():
        try:
            train_payload = json.loads(train_metrics_path.read_text(encoding="utf-8"))
            checkpoint_metrics = metadata_from_checkpoint.get("checkpoint_metrics")
            metrics_from_train = {
                "best_epoch": train_payload.get("best_epoch"),
                "selection_metric": train_payload.get("best_metric"),
                "selection_score": train_payload.get("best_score"),
                "checkpoint_epoch": metadata_from_checkpoint.get("checkpoint_epoch"),
                "checkpoint_metrics": checkpoint_metrics,
                "test": train_payload.get("best_test", train_payload.get("final_test", {})),
                "validation": checkpoint_metrics or train_payload.get("final_val", {}),
                "final": {
                    "train": train_payload.get("final_train", {}),
                    "validation": train_payload.get("final_val", {}),
                    "test": train_payload.get("final_test", {}),
                },
            }
        except json.JSONDecodeError:
            metrics_from_train = {}

    out_payload = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "model_found": model_path.exists(),
    }
    out_payload.update(existing_eval)
    out_payload.update(metrics_from_train)

    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
