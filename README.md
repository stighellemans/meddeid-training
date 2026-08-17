# meddeid-training

Training and export tools for MedDeID sequence-labeling models. Researchers can
run one ordinary fit, while release workflows can separate epoch selection from
the final full-data fit so a benchmark remains sealed during model selection.

The [training and evaluation workflow](https://meddeid.github.io/workflows/train-and-evaluate/)
shows the cross-suite handoffs. This repository and its training protocol remain
authoritative for configuration, fitting, refitting, and export.

## Installation

```bash
python -m pip install 'meddeid-training[train]'
```

## One-time training

For an ordinary research run, fit once using separate train, validation, and
test files. Validation chooses the best checkpoint and the test set is evaluated
after training:

```bash
meddeid-train fit \
  --config configs/release.yaml \
  --data prepared/fit \
  --run runs/fit
```

Use `--epochs N` to override the maximum configured epoch count. The resulting
checkpoint is `runs/fit/checkpoints/best.pt`.

## Publication protocol

1. Create a held-out validation subset from the 6,493-document synthetic
   development corpus.
2. Select an epoch count using only that validation subset.
3. Restart from the configured initial model and refit on the complete corpus
   for the selected number of epochs.
4. Evaluate once on the independent 300-document synthetic benchmark.
5. Export a self-contained Safetensors model bundle.

```bash
meddeid-train select-epochs \
  --config configs/release.yaml \
  --data prepared/selection \
  --run runs/selection

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

`configs/release.yaml` initializes both stages from the complete
`stighellemans/meddeid-dutch-synth` bundle, including its encoder and trained
classifier heads. To train new heads from a base encoder, set that encoder as
`model_name` and pin its immutable `model_revision`.

The `select-epochs` and `refit` commands deliberately name the two stages of
this stricter protocol; they are not required for an ordinary one-time fit.
Selection and refit always restart independently from the configured initial
model; refit never continues from the selection checkpoint. Dataset manifests,
model revisions, ordered labels, run configuration, and output checksums are
recorded for reproducibility.

See [Release training protocol](docs/training-protocol.md) for input layout,
stage invariants, configuration, and export requirements.

## Development

```bash
pip install -e '.[dev]'
pytest
```

## Licence

AGPL-3.0-only. Datasets and model artifacts are distributed separately under
the terms stated with each artifact.
