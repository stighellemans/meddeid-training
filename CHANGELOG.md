# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

## [Unreleased]

## [0.3.0] - 2026-09-08

- Updated the supported dependency line to MedDeID 0.4 and MedDeID Eval 0.5
  for the coordinated suite release.

## [0.2.1] - 2026-09-05

- Made the English retraining and resume scripts portable across local suite
  checkouts while retaining their existing experiment protocol.
- Extended compatible dependency ranges to the coordinated MedDeID 0.3 and
  evaluation 0.4 releases.

## [0.2.0] - 2026-08-27

- Standardized profile-aware epoch selection and clean full-development refit,
  with an explicit 30-epoch ceiling, absolute-best checkpoint policy, separate
  early-stopping tolerance, and single benchmark access accounting.

## [0.1.2] - 2026-08-18

- Reworked training plots into readable loss and task-specific F1 figures.
- Added shared MedDeID plotting conventions and searchable vector PDF output.
- Added an explicit `plots` installation extra and plot regression coverage.

## [0.1.1] - 2026-08-17

- Published the first externally supported MedDeID training release.
- Added public installation, compatibility, licensing, and verification
  metadata.
- Established independent CI and immutable release artifacts.

For earlier migration history, consult the repository's Git history.
