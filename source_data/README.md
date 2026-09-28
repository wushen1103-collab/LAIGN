# Source data

`derived/` contains the compact numerical tables used by the publication
figure scripts. These tables contain benchmark identifiers and numerical
results only; raw structures, licensed datasets, trained checkpoints and
machine-specific paths are intentionally excluded.

Run the evaluation workflows under `scripts/evaluation/` to regenerate these
tables from newly trained models. The plotting scripts read this directory
without requiring access to the original compute environment.
