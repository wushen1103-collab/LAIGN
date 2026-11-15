# Result-level reproduction index

Run commands from the repository root. This index distinguishes regenerating
an analysis from prediction archives, recomputing a statistic from compact
source data, and rendering a figure from a numerical source table. A rendered
summary table is not a substitute for independently regenerated predictions.

## Shared inputs and order

1. Install `environment.yml` and run `python -m scripts.tools.check_environment`.
2. Use PLINDER `2024-06/v2`, the frozen cohort files in `source_data/cohorts/`,
   and the exact parent-complex splits. The README lists the preparation and
   scorer/localizer training commands; the full-data graph limits and archive
   schema are documented in [FUSION_REPRODUCTION.md](FUSION_REPRODUCTION.md).
3. Generate the scorer checkpoints and residue prediction archives before
   fitting alternative heads or running prediction-level evaluation. The
   per-run train/validation membership is in `localizer_runs.csv.gz`.
4. Select models only on internal validation. Keep original and disjoint-stage
   fitting protocols separate, and retain all sites of a complex in resampling.

The table below uses repository-relative paths. Modules with required raw or
prediction inputs expose them through `python -m MODULE --help`; those are
input interfaces, not standalone reproduction commands. The fusion guide gives
complete staged commands once its explicit input locations have been filled.

## Main results

| Result | Inputs and preprocessing | Analysis/configuration | Intermediate output and numerical source | Figure command |
| --- | --- | --- | --- | --- |
| Residue ranking; Table 1 and Fig. 2 | Frozen PLINDER-200 membership; aligned LAIGN and official baseline residue predictions; native-output adapters | `scripts.evaluation.evaluate_matched_controls`; complex-level paired statistics in `scripts.evaluation.analyze_paired_statistics`; `configs/reproduction.yaml` | Aligned residue predictions and per-complex ranking metrics; `source_data/derived/fig2_paired_graph_ap_source.csv` preserves paired graph AP. Other Table 1 metrics require residue predictions. | `python figures/plot_fig2.py` |
| Within-pocket and distance-matched ranking; Table S6 | Original three localizer training/prediction archives and PLINDER-200 membership; retain residues within 6/8 A | `scripts.evaluation.analyze_plinder_pocket_and_trainable_fusion`; training-only fitting of two-input controls; complex-blocked inference | `source_data/validation/plinder200_pocket_summary.csv`, `plinder200_pocket_paired_effects.csv`, `plinder200_distance_matched_pair_summary.csv`, `plinder200_distance_matched_pair_effects.csv`, `trainable_fusion_selection.csv` | Numerical table endpoint, no separate figure |
| Post-freeze, chemical and pose analyses; Table 2 and Fig. 3 | Three complete retraining pipelines; fixed 70-complex holdout; receptor groups; CCD structures and training chemistry; frozen pose labels | `scripts.evaluation.summarize_postfreeze_holdout`, `analyze_chemical_shift`, `evaluate_pose_robustness`; retain run-mean and probability-ensemble endpoints separately | `source_data/derived/fig3_temporal_scatter.csv`, `fig3_absolute.csv`, `fig3_effect.csv`, `figs2_pose.csv`; cohort receptor sequences in `source_data/cohorts/postfreeze_receptors.csv` | `python figures/plot_main_quantitative.py` (Figs. 3 and 4) |
| Component and alignment controls; Table 3 and Fig. S1 | Matched localizer splits; component predictions; within-complex shuffled pair scores; validation-only calibration fit | `scripts.model.train_residue_localizer`, `scripts.evaluation.summarize_calibration`; fixed seeds and feature settings in `configs/reproduction.yaml` | `source_data/derived/figs1_components.csv`, `figs1_route.csv`, `figs1_shuffle.csv`, `figs1_calibration.csv` | `python figures/plot_supplementary.py` (numerical supplementary panels) |
| Experimental mutation endpoints; Fig. 4 and Table S3 | Strict Platinum records; mapped wild-type sites; overlap-purged training; retain all receptor candidates for top-K | `scripts.evaluation.analyze_platinum_validation`, `analyze_platinum_distance_conditioned`; complex-blocked resampling; grouped held-out incremental prediction | `source_data/validation/platinum_assayed_site_scores.csv`; `source_data/derived/fig4_auprc.csv`, `fig4_effect.csv`, `fig4_topk.csv`, `blocked_bootstrap.csv`, `budget_bootstrap.csv`, `topk_metrics.csv` | `python figures/plot_main_quantitative.py` |
| Molecular cases; Fig. 5 and Table S4 | Public 1HNN/SKF and 2DOR/ORO structures; verify chain, residue identity and ligand instance against mapped mutation records | `source_data/derived/case_candidates.csv` and the strict assayed-site table define the numerical comparisons | PDB coordinates plus verified residue, distance, model-score and mutation-effect records. Exact publication scene/camera assets are not included in this code repository. | Molecular views require PyMOL; numerical plotting commands do not generate these scenes |

Fig. 1 is a method schematic, not a statistical result; it is not regenerated
by a numerical evaluation script. Fig. S7 is likewise a molecular comparison
view requiring the corresponding public 6IMJ/ATP coordinates and scene setup.

## Remaining supplementary results

| Result | Source data | Analysis/rendering entry |
| --- | --- | --- |
| Fig. S2: pose-coordinate perturbation | `source_data/derived/figs2_pose.csv` | `scripts.evaluation.evaluate_pose_robustness`; `figures/plot_supplementary.py` |
| Fig. S3 and Table S2: ligand shifts | `figs3_similarity_summary.csv`, `figs3_similarity_effect.csv`, `figs3_chemical_rows.csv`, `post_freeze_chemical_strata_three_pipeline.csv` under `source_data/derived/` | `scripts.evaluation.analyze_chemical_shift`; `figures/plot_supplementary.py` |
| Fig. S4: Platinum filtering and chemical composition | `figs4_flow.csv`, `figs4_sites_per_complex.csv`, `figs4_mutations_per_site.csv`, `figs4_overlap.csv`, `figs4_purge.csv`, `figs3_scaffold_counts.csv` under `source_data/derived/` | Data-preparation overlap checks; `figures/plot_supplementary.py` |
| Figs. S5/S6: mutation sensitivity and budgeted retrieval | `figs5_secondary_metrics.csv`, `figs5_threshold.csv`, `figs5_homology.csv`, `figs6_hit.csv`, `figs6_budget_effect.csv`, `figs6_per_complex_recall10.csv` under `source_data/derived/` | `scripts.evaluation.analyze_platinum_validation`; `figures/plot_supplementary.py` |
| Tables S7-S10: matched heads, typed channels, disjoint stages and annotation transfer | All 17 CSVs under `source_data/fusion_validation/`; frozen sample/group files under `source_data/cohorts/` | Complete command sequence and input archive schema in [FUSION_REPRODUCTION.md](FUSION_REPRODUCTION.md) |

## Verify Fig. 3a without plotting

```bash
python figures/temporal_statistics.py
python -m pytest -q tests/test_temporal_figure_statistics.py
```

The helper reads the same paired per-complex AP columns as the scatter and
uses 10,000 whole-complex bootstrap resamples with seed 8241. It returns
`delta = 0.009714178402794501`, 95% CI
`[-0.006374299913225312, 0.025256214840576555]`. The displayed annotation is
therefore +0.0097 and [-0.0064, 0.0253]. The mean of three runs' AP is a
different statistic; it must not replace the scatter endpoint.

These two verification commands do not create or alter any image file.
Running the figure commands above is a separate, explicit rendering step.
