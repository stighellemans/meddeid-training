#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
training_root="${script_dir:h}"
repos_root="${training_root:h}"
suite_root="${repos_root:h}"
local_root="${suite_root:h}"
data_root="$repos_root/meddeid-data/data/english-production-v2/training-views"
benchmarks_root="${MEDDEID_BENCHMARKS_ROOT:-$local_root/english-deid-benchmarks}"
python_bin="${MEDDEID_PYTHON:-$suite_root/workspaces/manual-validation/.venv/bin/python}"
run_root="$training_root/runs/english-gb-us-roberta-base-data-v2-final-20260826"
config="$training_root/configs/english-gb-us.yaml"
selection_data="$data_root/selection"
refit_data="$data_root/refit"
checkpoint="$run_root/refit/checkpoints/best.pt"
train_metrics="$run_root/refit/train_metrics.json"
log_file="$run_root/pipeline.log"
status_file="$run_root/pipeline-status.txt"

if [[ -e "$run_root" ]]; then
  if [[ -f "$status_file" ]] \
    && grep -q 'failed (exit ' "$status_file" \
    && [[ ! -e "$run_root/selection/run.json" ]] \
    && [[ ! -e "$run_root/selection/checkpoints/best.pt" ]]; then
    print -r -- "Restarting the preflight-failed run in place: $run_root"
  else
    print -u2 "Refusing to overwrite existing or partially trained run directory: $run_root"
    exit 1
  fi
fi
mkdir -p "$run_root"
exec > >(tee -a "$log_file") 2>&1

pipeline_succeeded=0
set_status() {
  local state="$1"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" > "$status_file"
  print -r -- "[pipeline] $state"
}
finish() {
  local exit_code=$?
  if (( pipeline_succeeded )); then
    set_status "complete"
  else
    set_status "failed (exit $exit_code)"
  fi
}
trap finish EXIT

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$training_root/src:$repos_root/meddeid-core/src:$repos_root/meddeid-eval/src:$repos_root/meddeid-language-en/src"

set_status "validating improved English training views"
"$python_bin" - "$selection_data" "$refit_data" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_view(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profiles") != ["en-GB", "en-US"]:
        raise SystemExit(
            f"{root}/manifest.json must use the unversioned profiles en-GB and en-US"
        )
    documents = []
    for split, info in manifest["files"].items():
        path = root / info["filename"]
        actual_hash = sha256(path)
        if actual_hash != info["sha256"]:
            raise SystemExit(f"checksum mismatch for {path}: {actual_hash} != {info['sha256']}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != info["documents"]:
            raise SystemExit(f"document-count mismatch for {path}: {len(rows)} != {info['documents']}")
        if split != "test" or rows:
            documents.extend(rows)
    return documents


selection = validate_view(Path(sys.argv[1]))
refit = validate_view(Path(sys.argv[2]))
docs = {row["document_id"]: row for row in selection + refit}.values()
if any("@example.test" in row["text"].lower() for row in docs):
    raise SystemExit("legacy @example.test address remains in the training views")
for row in docs:
    metadata = row.get("metadata") or {}
    language = str(metadata.get("lang") or "").replace("_", "-")
    profile = str(metadata.get("generation_profile") or "").replace("_", "-")
    if "@" in profile or profile not in {"en-GB", "en-US"}:
        raise SystemExit(
            f"{row['document_id']} has invalid versioned/unsupported generation profile {profile!r}"
        )
    if profile != language:
        raise SystemExit(
            f"{row['document_id']} generation profile {profile!r} conflicts with lang {language!r}"
        )

patient_ids = []
national_ids = []
for row in docs:
    for span in row.get("spans", []):
        if span.get("label") != "ID:Patient":
            continue
        value = str(span.get("text", ""))
        patient_ids.append(value)
        if span.get("source_slot") == "patient.national_id":
            national_ids.append(value)
if not national_ids:
    raise SystemExit("no patient national identifiers found in the training views")
if not any(value.isdigit() for value in patient_ids):
    raise SystemExit("no numeric-only patient identifiers found in the training views")
if not any(re.search(r"[- /]", value) for value in patient_ids):
    raise SystemExit("no grouped patient identifiers found in the training views")
print(
    f"validated {len(docs)} unique documents, {len(patient_ids)} patient identifiers, "
    f"and {len(national_ids)} national identifiers"
)
PY

set_status "selecting epoch count"
cd "$training_root"
"$python_bin" -m meddeid_training.cli select-epochs \
  --config "$config" \
  --data "$selection_data" \
  --run "$run_root/selection"
test -s "$run_root/selection/run.json"

set_status "refitting on complete development data"
"$python_bin" -m meddeid_training.cli refit \
  --config "$config" \
  --selection "$run_root/selection/run.json" \
  --data "$refit_data" \
  --run "$run_root/refit"
test -s "$checkpoint"
test -s "$train_metrics"

set_status "exporting unversioned English model bundle"
"$python_bin" -m meddeid_training.cli export \
  --checkpoint "$checkpoint" \
  --run-metadata "$train_metrics" \
  --name meddeid-english-synth \
  --output "$run_root/export/meddeid-english-synth"

set_status "updating English benchmark configs"
"$python_bin" "$training_root/scripts/update_english_benchmark_checkpoint.py" \
  --checkpoint "$checkpoint" \
  --train-metrics "$train_metrics" \
  --benchmarks-root "$benchmarks_root"

for battery in meddeid-english-synthetic asq-phi technetium-i; do
  battery_root="$benchmarks_root/batteries/$battery"
  results_root="$battery_root/results"
  typeset -a model_ids run_arguments verification_outputs
  if [[ "$battery" == "meddeid-english-synthetic" ]]; then
    results_root="$results_root/human-validated-v1"
    # The synthetic input changed, so every comparison model must be refreshed.
    model_ids=(
      openmed-superclinical
      obi-i2b2-roberta
      openai-privacy-filter
      openmed-multilingual
      gliner-multilingual-pii
      philter-ucsf
      meddeid-english-synth
    )
    run_arguments=()
  else
    # The external inputs and comparator models did not change. Refresh only
    # the newly trained MedDeID checkpoint and retain comparator inference.
    model_ids=(meddeid-english-synth)
    run_arguments=(--only meddeid-english-synth)
    for comparator_id in \
      openmed-superclinical \
      obi-i2b2-roberta \
      openai-privacy-filter \
      openmed-multilingual \
      gliner-multilingual-pii \
      philter-ucsf; do
      test -s "$results_root/runs/$comparator_id/raw.jsonl"
    done
  fi
  verification_outputs=()

  for model_id in $model_ids; do
    model_run="$results_root/runs/$model_id"
    prediction="$results_root/work/predictions/$model_id.jsonl"
    case "$model_run" in
      "$benchmarks_root"/batteries/*/results/runs/*|\
      "$benchmarks_root"/batteries/*/results/human-validated-v1/runs/*) ;;
      *) print -u2 "Unsafe benchmark cleanup target: $model_run"; exit 1 ;;
    esac
    rm -rf -- "$model_run"
    rm -f -- "$prediction"
    verification_outputs+=("$model_run/raw.jsonl")
    if [[ "$battery" == "meddeid-english-synthetic" ]]; then
      verification_outputs+=(
        "$model_run/by_doc.nometa.jsonl"
        "$model_run/by_doc.meta.jsonl"
      )
    else
      verification_outputs+=("$model_run/by_doc.jsonl")
    fi
  done

  set_status "benchmarking $battery"
  PATH="${python_bin:h}:$PATH" \
    "$benchmarks_root/batteries/run_battery.sh" "$battery" \
      "${run_arguments[@]}" \
      --device mps

  "$python_bin" - "$battery_root/input.jsonl" "${verification_outputs[@]}" <<'PY'
import json
import sys
from pathlib import Path


def ids(path: Path, key: str) -> list[str]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(str(json.loads(line)[key]))
    return values


input_ids = ids(Path(sys.argv[1]), "document_id")
for output_name in sys.argv[2:]:
    output_ids = ids(Path(output_name), "doc_id")
    if output_ids != input_ids:
        raise SystemExit(
            f"benchmark output does not exactly cover its input: {output_name} "
            f"({len(output_ids)} output rows versus {len(input_ids)} inputs)"
        )
print(f"verified {len(input_ids)} predictions")
PY
done

pipeline_succeeded=1
set_status "all training and benchmark stages completed"
