#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
training_root="${script_dir:h}"
repos_root="${training_root:h}"
suite_root="${repos_root:h}"
local_root="${suite_root:h}"
benchmarks_root="${MEDDEID_BENCHMARKS_ROOT:-$local_root/english-deid-benchmarks}"
training_run="$training_root/runs/english-gb-us-roberta-base-data-v2-final-20260826"
python_bin="${MEDDEID_PYTHON:-$suite_root/workspaces/manual-validation/.venv/bin/python}"
status_file="$training_run/pipeline-status.txt"
log_file="$training_run/benchmark-resume.log"
all_model_ids=(
  openmed-superclinical
  obi-i2b2-roberta
  openai-privacy-filter
  openmed-multilingual
  gliner-multilingual-pii
  philter-ucsf
  meddeid-english-synth
)
comparator_ids=(
  openmed-superclinical
  obi-i2b2-roberta
  openai-privacy-filter
  openmed-multilingual
  gliner-multilingual-pii
  philter-ucsf
)

test -s "$training_run/selection/run.json"
test -s "$training_run/refit/checkpoints/best.pt"
test -s "$training_run/refit/train_metrics.json"
test -s "$training_run/export/meddeid-english-synth/bundle.json"
exec > >(tee -a "$log_file") 2>&1

completion_succeeded=0
set_status() {
  local state="$1"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" > "$status_file"
  print -r -- "[benchmark-resume] $state"
}
finish() {
  local exit_code=$?
  if (( completion_succeeded )); then
    set_status "changed-result benchmarks complete; manuscript update pending"
  else
    set_status "changed-result benchmark resume failed (exit $exit_code)"
  fi
}
trap finish EXIT

verify_outputs() {
  local battery="$1"
  local results_root="$2"
  shift 2
  typeset -a outputs
  outputs=()
  for model_id in "$@"; do
    outputs+=("$results_root/runs/$model_id/raw.jsonl")
    if [[ "$battery" == "meddeid-english-synthetic" ]]; then
      outputs+=(
        "$results_root/runs/$model_id/by_doc.nometa.jsonl"
        "$results_root/runs/$model_id/by_doc.meta.jsonl"
      )
    else
      outputs+=("$results_root/runs/$model_id/by_doc.jsonl")
    fi
  done
  "$python_bin" - "$benchmarks_root/batteries/$battery/input.jsonl" "${outputs[@]}" <<'PY'
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
print(f"verified {len(input_ids)} documents across {len(sys.argv) - 2} output files")
PY
}

clear_model() {
  local results_root="$1"
  local model_id="$2"
  local model_run="$results_root/runs/$model_id"
  local prediction="$results_root/work/predictions/$model_id.jsonl"
  case "$model_run" in
    "$benchmarks_root"/batteries/*/results/runs/*|\
    "$benchmarks_root"/batteries/*/results/human-validated-v1/runs/*) ;;
    *) print -u2 "Unsafe benchmark cleanup target: $model_run"; exit 1 ;;
  esac
  rm -rf -- "$model_run"
  rm -f -- "$prediction"
}

export PATH="${python_bin:h}:$PATH"

# The improved synthetic benchmark has already been rerun for every model.
synthetic_root="$benchmarks_root/batteries/meddeid-english-synthetic/results/human-validated-v1"
set_status "verifying completed all-model synthetic benchmark"
verify_outputs meddeid-english-synthetic "$synthetic_root" "${all_model_ids[@]}"

# An earlier broad recovery was stopped after it had cleared ASQ-PHI comparator
# runs. Reuse every surviving artifact, restore the unchanged Philter result
# from its existing staging copy, and recompute only comparator raw outputs that
# are now genuinely absent. This repair is specific to the accidental clearing;
# it is not part of the normal changed-results benchmark policy.
asq_root="$benchmarks_root/batteries/asq-phi/results"
philter_staging="$benchmarks_root/batteries/asq-phi/results-philter-staging/runs/philter-ucsf"
if [[ ! -s "$asq_root/runs/philter-ucsf/raw.jsonl" ]]; then
  set_status "restoring unchanged ASQ-PHI Philter artifacts from staging"
  test -s "$philter_staging/raw.jsonl"
  mkdir -p "$asq_root/runs/philter-ucsf"
  cp -p "$philter_staging/raw.jsonl" "$asq_root/runs/philter-ucsf/raw.jsonl"
  cp -p "$philter_staging/by_doc.jsonl" "$asq_root/runs/philter-ucsf/by_doc.jsonl"
  cp -p "$philter_staging/runner_timing.json" "$asq_root/runs/philter-ucsf/runner_timing.json"
fi

typeset -a asq_run_ids
asq_run_ids=()
for comparator_id in $comparator_ids; do
  if [[ ! -s "$asq_root/runs/$comparator_id/raw.jsonl" ]]; then
    asq_run_ids+=("$comparator_id")
  fi
done
clear_model "$asq_root" meddeid-english-synth
asq_run_ids+=(meddeid-english-synth)
set_status "repairing missing ASQ-PHI artifacts and refreshing MedDeID only"
"$benchmarks_root/batteries/run_battery.sh" asq-phi \
  --only "${(j:,:)asq_run_ids}" \
  --device mps
verify_outputs asq-phi "$asq_root" "${all_model_ids[@]}"

# Technetium-I comparator inference is intact. Refresh only MedDeID; the
# orchestrator reuses the retained comparator raw outputs in the combined report.
technetium_root="$benchmarks_root/batteries/technetium-i/results"
for comparator_id in $comparator_ids; do
  test -s "$technetium_root/runs/$comparator_id/raw.jsonl"
done
clear_model "$technetium_root" meddeid-english-synth
set_status "refreshing MedDeID only on Technetium-I"
"$benchmarks_root/batteries/run_battery.sh" technetium-i \
  --only meddeid-english-synth \
  --device mps
verify_outputs technetium-i "$technetium_root" "${all_model_ids[@]}"

completion_succeeded=1
