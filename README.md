# LAIGN

LAIGN (Ligand-Aware Interaction-Grounded Network) ranks every receptor residue
in a resolved protein-ligand complex. It first predicts typed residue-atom
interaction probabilities with a coordinate-aware graph network, averages
three independently trained scorers, and then combines the consensus map with
explicit geometry through a typed multi-scale residue localizer.

## Repository contents

- `scripts/data/`: BioLiP2 and PLINDER preparation, PLIP labels, graph
  construction, post-freeze cohort construction and overlap purging.
- `scripts/model/`: interaction-scorer and residue-localizer training,
  calibration, inference and fixed-input controls.
- `scripts/evaluation/`: matched controls, chemical shift, pose perturbation,
  post-freeze evaluation, Platinum validation and statistical analysis.
- `scripts/baselines/`: adapters for official third-party baseline outputs.
- `figures/`: publication data-figure generation.
- `source_data/derived/`: compact publication-level numerical source tables.
- `configs/reproduction.yaml`: final model and evaluation settings.

Raw datasets and model checkpoints are not committed because of size and
upstream licensing. Every workflow accepts repository-relative paths or
explicit command-line paths; no machine-specific path is required.

## Environment

```bash
conda env create -f environment.yml
conda activate laign
python -m scripts.tools.check_environment
```

CUDA builds of PyTorch and PyG must match the local NVIDIA driver. A Docker
definition is also provided. Dataset preparation additionally requires the
PLIP command-line interface, MMseqs2 and the wwPDB/RCSB download endpoints.

## Data preparation

Download BioLiP2, PLINDER and Platinum only from their official sources and
respect their licenses. The preparation scripts expose complete command-line
interfaces:

```bash
python -m scripts.data.build_biolip2_candidates --help
python -m scripts.data.generate_biolip2_plip_labels --help
python -m scripts.data.extract_plinder_selected_systems --help
python -m scripts.data.generate_plinder_plip_labels --help
python -m scripts.data.build_structure_supervision_manifest --help
python -m scripts.data.build_structure_supervision_splits --help
python -m scripts.data.build_overlap_purged_training --help
```

Construct the coordinate graphs used by the interaction scorer:

```bash
python -m scripts.data.build_interaction_graph_dataset \
  --out-dir data/processed/structure_supervision/interaction_graphs \
  --negative-mode hard --negative-ratio 10 \
  --max-negatives-per-complex 128 --seed 2401
```

## Training

Train three interaction scorers with seeds 2401, 2402 and 2403. The example
below shows one run; repeat it with the other two seeds and output directories.

```bash
python -m scripts.model.train_interaction_scorer \
  --graph-dir data/processed/structure_supervision/interaction_graphs \
  --out-dir outputs/interaction_scorer_seed2401 \
  --checkpoint-dir checkpoints/interaction_scorer_seed2401 \
  --table outputs/tables/interaction_scorer_seed2401.md \
  --epochs 10 --batch-size 8 --hidden 64 --layers 3 --heads 8 \
  --num-rbf 16 --cutoff 5.0 --max-neighbors 64 --seed 2401
```

Train the typed multi-scale residue localizer with the three-scorer consensus:

```bash
python -m scripts.model.train_residue_localizer \
  --checkpoint checkpoints/interaction_scorer_seed2401/best.pt \
  --ensemble-checkpoints checkpoints/interaction_scorer_seed2402/best.pt \
                         checkpoints/interaction_scorer_seed2403/best.pt \
  --interaction-backbone visnet \
  --stage2-feature-mode typed_multiscale \
  --stage2-classifier logreg \
  --candidate-radius 999 \
  --max-train-graphs 5000 --max-internal-val-graphs 1000 \
  --out-dir outputs/residue_localizer \
  --predictions-output outputs/residue_localizer/predictions.npz \
  --table outputs/tables/residue_localizer.md --seed 2401
```

## Evaluation

The principal robustness and biological-validation workflows are available as
modules and document all required inputs through `--help`:

```bash
python -m scripts.evaluation.evaluate_matched_controls --help
python -m scripts.evaluation.analyze_chemical_shift --help
python -m scripts.evaluation.evaluate_pose_robustness --help
python -m scripts.evaluation.summarize_postfreeze_holdout --help
python -m scripts.evaluation.analyze_platinum_validation --help
python -m scripts.evaluation.analyze_paired_statistics --help
```

External baseline adapters require outputs from the corresponding official
implementations. See `scripts/baselines/README.md` for the evaluation boundary.

## Figures

The numerical figure panels can be regenerated from the committed source data:

```bash
python figures/plot_fig2.py
python figures/plot_main_quantitative.py
python figures/plot_supplementary.py
```

Generated files are written to ignored `main/` and `supplementary/`
directories. Molecular case renderings require the corresponding public PDB
structures and PyMOL and are therefore not bundled as binary assets.

## Reproducibility notes

- Training, validation and external-test partitions are defined before model
  fitting; post-freeze and Platinum workflows include overlap audits.
- The final consensus uses scorer seeds 2401, 2402 and 2403.
- Statistics operate at the complex level unless a workflow explicitly states
  a site-level endpoint.
- Compact source tables are included, while raw structures and checkpoints are
  regenerated locally to keep the repository small and license compliant.

## License

The repository code is released under the MIT License. Upstream datasets,
baseline implementations and model weights retain their original licenses.
