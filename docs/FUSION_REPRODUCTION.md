# Reproducing Supplementary Tables S7-S10

## Frozen dataset snapshot

Use PLINDER release `2024-06`, iteration `v2`. The retained upstream files are
`index/annotation_table.parquet` and `splits/split.parquet`; their exact byte
sizes and SHA-256 hashes are in `source_data/cohorts/plinder_release.json`.
The download source is the public `storage.googleapis.com/plinder/2024-06/v2/`
tree. Do not substitute the current monthly release or construct a new split.

The upstream split contains 832 validation and 1,036 test systems. After the
paper's preprocessing, 551 validation and 785 test complexes are retained;
these are not the upstream unfiltered counts. Every retained assignment was
checked against the archived official split, with zero mismatches.
`plinder_cohorts.csv` preserves system IDs, ligand IDs, split assignments and
PLINDER-200 membership (100 validation and 100 test complexes).

`localizer_runs.csv.gz` records the exact per-run train/validation/evaluation
sample IDs. `structure_splits.csv.gz` records the structural split groups.
`disjoint_protocol.json` preserves the originally frozen stage memberships;
reuse it rather than selecting another holdout. Similarity strata and the
70-complex temporal receptor sequences are included separately. These files
contain public biological identifiers, not machine-specific paths.

## Inputs and environment

Install the root environment, including LightGBM 4.6.0. Run commands from the
repository root as Python modules. `LAIGN_WORKSPACE` optionally selects a data
workspace with the same relative directory structure; `LAIGN_FUSION_OUTPUT`
optionally selects an output directory. By default both are repository-relative.
`LAIGN_DEVICE=cpu` supports a small CPU smoke test; CUDA is used when available.
Each training process uses at most eight CPU threads. Choose concurrency with
available resources in mind.

Large raw structures, pair caches and trained checkpoints are regenerated,
not committed. The following are the expected relative input locations:

| Input | Location |
| --- | --- |
| Preprocessed structures and labels | `data/processed/` (as used by `scripts.model.train_residue_localizer`) |
| Hard-negative scorer graph shards | `data/processed/structure_supervision/interaction_graphs/` |
| Three original scorer checkpoints | `checkpoints/scorer_seed2401/best.pt` (also 2402 and 2403) |
| Original PLINDER localizer archives | `outputs/reference/plinder/seed2401/predictions.npz` (also 2402 and 2403) |
| Temporal localizer archives | `outputs/reference/postfreeze/replicate_A/predictions.npz` (also B and C) |
| Temporal preprocessing inputs | `outputs/reference/postfreeze/temporal_blind_ccd_manifest.csv.gz`, `temporal_blind_plip_interactions.csv.gz`, `temporal_blind_site_labels.csv.gz` |

Prediction archives use the existing Stage 2 schema:
`<split>__sample_ids`, `<split>__sample_index`, `<split>__labels`, and scores
`raw_visnet`, `raw_distance`, `trained_stage2`, `fixed_hybrid` under
`<split>__score__<name>`. Splits are `train`, `internal_val`, `external_val`
and `external_test`. `raw_distance` is **negative** nearest-atom distance.
Archive rows must agree exactly with regenerated cache labels and group order.
The cache builder checks this alignment downstream; it does not silently join
unmatched residue arrays. Each cache has labels, nearest distance, 73 features,
five typed pair probabilities, pair distances, residue-to-pair mapping and
explicit residue keys.

The frozen per-run sample IDs must be used when regenerating localizer
archives. The public archives list IDs rather than raw structure paths;
preprocessing manifests supplied at run time must use workspace-relative
paths or valid paths on the reproducer's own machine. Temporal checkpoint
metadata similarly points to the reproducer's scorer triplets.

The graph builder defaults to a smoke-sized dataset. For full scorer training,
set every graph-limit option to zero and write the shards to the directory
used by the disjoint wrapper:

```bash
python -m scripts.data.build_interaction_graph_dataset \
  --out-dir data/processed/structure_supervision/interaction_graphs \
  --max-train-complexes 0 --max-internal-val-complexes 0 \
  --max-external-val-complexes 0 --max-external-test-complexes 0 \
  --negative-mode hard --negative-ratio 4 --max-negatives-per-complex 128 \
  --num-workers 8 --seed 2401
```

## S7: matched-information fusion controls

```bash
python -m scripts.fusion_validation.cache_pair_evidence --mode original
python -m scripts.fusion_validation.run_twoinput_fusion --cohort plinder
python -m scripts.fusion_validation.run_feature_heads --run 0
python -m scripts.fusion_validation.run_deepsets_fusion --run 0
python -m scripts.fusion_validation.run_refit_reference --run 0
python -m scripts.fusion_validation.run_branch_heads --run 0
```

Repeat the four per-run commands for runs 1 and 2, then aggregate:

```bash
python -m scripts.fusion_validation.aggregate_fusion_runs --cohort plinder
```

The original cache is shared because the three original localizers use the
same frozen scorer triplet. For temporal controls, build a separate cache for
each pipeline, then fit matching heads:

```bash
python -m scripts.fusion_validation.cache_pair_evidence --mode postfreeze --pipeline A
python -m scripts.fusion_validation.run_twoinput_fusion --cohort postfreeze
python -m scripts.fusion_validation.run_feature_heads --cohort postfreeze --run 0
python -m scripts.fusion_validation.run_deepsets_fusion --cohort postfreeze --run 0
python -m scripts.fusion_validation.run_refit_reference --cohort postfreeze --run 0
```

Repeat with B/run 1 and C/run 2, and aggregate `--cohort postfreeze`.
`--smoke` is available on the cache builder, two-input control, full-feature
head and DeepSets. Smoke output is separate from full results and must never
be used to populate the submitted tables. Selection traces expose the
internal-validation metric and every searched hyperparameter.

## S8: typed channels, refitted removals and branch selection

`run_feature_heads` performs the five removal refits unless `--skip-ablation`
is supplied. Removal drops the three channel-specific columns and recomputes
every shared any-channel aggregate, including cross terms; it does not merely
zero named features. `run_refit_reference` supplies the same-cache 73-feature
reference for these differences. `run_branch_heads` fits the 34-feature
geometry, 21-feature chemistry and 55-feature no-cross-term controls.

```bash
python -m scripts.fusion_validation.analyze_channels
python -m scripts.fusion_validation.evaluate_external_protocols
```

Typed pair diagnostics retain all pairs within 6 A, without negative
downsampling. Residue-ranking metrics always use all specified candidates.
Recall@10 divides recovered positives by **all reference-positive residues**,
not by ten. Paired bootstrap and sign-flip tests use complexes; temporal
intervals group all complexes sharing an exact receptor sequence. Holm
adjustment covers five removal comparisons within each cohort/metric family.

## S9: disjoint-stage control

Copy the frozen `source_data/cohorts/disjoint_protocol.json` to the fusion
output directory before training. `prepare_disjoint_protocol` exposes the
deterministic group-selection procedure for verification; do not replace the
published membership with a fresh sample.

```bash
python -m scripts.fusion_validation.train_scorer_disjoint --seed 2401
python -m scripts.fusion_validation.train_scorer_disjoint --seed 2402
python -m scripts.fusion_validation.train_scorer_disjoint --seed 2403
python -m scripts.fusion_validation.cache_pair_evidence --mode disjoint
python -m scripts.fusion_validation.cache_pair_evidence --mode disjoint_postfreeze
python -m scripts.fusion_validation.train_disjoint_heads
```

The held localizer set has 5,000 structures; 4,999 had an original scorer graph.
The original scorer had 40,057 graphs, so the disjoint scorer has **35,058**.
One held structure is built directly from the structural manifest. The
scorer wrapper records the graph IDs actually loaded. There are three scorer
fits, averaged into one pair map, and one fitted head per family in this
protocol; it is not three independent disjoint Stage 2 repetitions.

## S10 and table source data

`evaluate_external_protocols` also recomputes annotation-transfer endpoints,
checks explicit chain/residue number/residue type alignment, reports missing
archives and separates external validation from external test. The compact
196-complex ProLIF residue-score archive is supplied in
`source_data/validation/prolif_residue_predictions.csv.gz`; four unavailable
validation complexes remain explicit exclusions. ProLIF labels are a
rule-based annotation endpoint, not experimental biological validation.

All 17 submitted supplementary source CSVs are in
`source_data/fusion_validation/`, including the ProLIF distance advantage and
the matched-feature LightGBM advantage. Per-fit mean/SD and probability-
ensemble scores are distinct outputs and must not be substituted for one
another. Existing Platinum source data remain in `source_data/validation/`.

## Tests

```bash
python -m pytest -q
python scripts/tools/audit_release.py
```

Tests check ranking denominators, paired and receptor-grouped resampling,
shared-feature recomputation on removal, cohort integrity and cache alignment.
Small synthetic inputs test executable training paths; they are not scientific
benchmark results. Reproducing the complete numerical tables additionally
requires the regenerated structures, scorer checkpoints and prediction archives
described above.
