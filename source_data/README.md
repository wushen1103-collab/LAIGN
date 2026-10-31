# Source data

`derived/` contains the compact numerical tables used by the publication
figure scripts. These tables contain benchmark identifiers and numerical
results only; raw structures, licensed datasets, trained checkpoints and
machine-specific paths are intentionally excluded.

Run the evaluation workflows under `scripts/evaluation/` to regenerate these
tables from newly trained models. The plotting scripts read this directory
without requiring access to the original compute environment.

`validation/platinum_assayed_site_scores.csv` contains the compact table of
476 experimentally assayed sites used by the distance-conditioned Platinum
analysis. It contains public benchmark identifiers, measured endpoints and
model scores, without local paths or personal metadata.

The four `validation/plinder200_*` tables and
`validation/trainable_fusion_selection.csv` record the pocket-restricted
PLINDER-200 summaries, paired graph-AP effects, distance-matched residue-pair
analyses and internal-validation selection trace for the trainable two-input
fusion control. They contain aggregate statistics only and exclude prediction
archives, local paths and machine metadata.
