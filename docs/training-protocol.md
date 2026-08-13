# Release training protocol

This repository has one supported training route for `meddeid-dutch-synth`.
It consumes explicit, checksum-pinned dataset directories and immutable model
revisions.

## Inputs

Download the unsplit synthetic corpus and independent synthetic benchmark identified in
their Hugging Face cards. The corpus publisher does not choose a
training/validation split. Prepare two directories with
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `manifest.json`:

- `prepared/selection`: a user-chosen training subset, a disjoint held-out
  validation subset, and an empty `test.jsonl` placeholder;
- `prepared/refit`: all 6,493 training documents, with the same 300-document
  synthetic benchmark. Its validation view is for reporting only, not model selection.

The reference configuration uses seed 42 and produces 5,844 selection-training
and 649 validation documents. This partition is generated locally and is not a
separate published dataset split. Manifests and checksums—not directory names—
determine the data used by a run.

The default initial state is the complete existing model bundle
`stighellemans/meddeid-dutch-synth`: encoder, BIO head, and entity head. To
intentionally initialize new heads from a plain encoder, set that encoder as
`model_name`; this is already an explicit departure from the trained default.
`from_base_encoder: true` is an optional mode assertion rather than a required
second opt-in. `model_revision` pins either remote source so selection and refit
cannot resolve different model states.

## Stage 1: select the epoch count

```bash
meddeid-train select-epochs \
  --config configs/release.yaml \
  --data prepared/selection \
  --run runs/selection
```

Early stopping uses only the held-out validation subset. The synthetic benchmark is not
used to select a checkpoint. The command writes `runs/selection/run.json` with
the selected epoch count and the selection metrics.

## Stage 2: refit and export

```bash
meddeid-train refit \
  --config configs/release.yaml \
  --selection runs/selection/run.json \
  --data prepared/refit \
  --run runs/refit

meddeid-train export \
  --checkpoint runs/refit/checkpoints/best.pt \
  --run-metadata runs/refit/train_metrics.json \
  --output release/meddeid-dutch-synth
```

The refit restarts from the same configured initial MedDeID bundle as stage 1,
trains on all 6,493 training documents for the fixed epoch count, and evaluates
once on the 300-document synthetic benchmark. It does not continue from the selection
checkpoint. When `from_base_encoder: true` is explicitly configured, both
stages instead restart from that encoder with newly initialized heads.
Export writes a self-contained model directory: `model.safetensors`, local
encoder configuration and tokenizer files, plus `bundle.json` with the
canonical ordered 14-label entity head. The release process records the output
checksums.

Export reads the resolved encoder, window length, overlap, label order, and
split counts from the run metadata. Missing metadata, a non-canonical head, or
conflicting manual values causes the export to fail. `--unsafe-override` is
reserved for explicitly reviewed recovery work.

## Invariants

- No document ID or normalized text may overlap between training and the synthetic benchmark.
- Stage 1 validation must not overlap its training split.
- Stage 1 never reads or scores the synthetic benchmark.
- Stage 2 disables early stopping and treats the selected epoch count as fixed.
- Existing-model initialization restores the encoder and both classifier heads.
- A non-default plain `model_name` initializes from that base encoder; the
  optional `from_base_encoder: true` setting makes the mode explicit.
- Entity labels must equal `meddeid_core.BERT_ENTITY_LABELS`, including order.
- Dataset and model files stay outside Git and are identified by manifests.
- Refit results are accepted against declared metric tolerances; deterministic
  preprocessing, bundle validation, and scoring require exact equality.

Use `--dry-run` on either training command to inspect the exact invocation
without starting a training job.
