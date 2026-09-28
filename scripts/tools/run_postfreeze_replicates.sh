#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MANIFEST="outputs/postfreeze_holdout/postfreeze_split_manifest.csv.gz"
INTERACTIONS="outputs/postfreeze_holdout/postfreeze_plip_interactions.csv.gz"
SITE_LABELS="outputs/postfreeze_holdout/postfreeze_site_labels.csv.gz"

run_replicate() {
    local gpu="$1"
    local replicate="$2"
    local localizer_seed="$3"
    local seed_a="$4"
    local seed_b="$5"
    local seed_c="$6"
    local out_dir="outputs/postfreeze_holdout/replicate_${replicate}"
    local checkpoint_root="checkpoints/interaction_scorer_seed"

    mkdir -p "$out_dir" logs outputs/tables
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PYTHON" -m scripts.model.train_residue_localizer \
        --split-manifest "$MANIFEST" \
        --plinder-interactions "$INTERACTIONS" \
        --plinder-site-labels "$SITE_LABELS" \
        --checkpoint "${checkpoint_root}${seed_a}/best.pt" \
        --ensemble-checkpoints \
            "${checkpoint_root}${seed_b}/best.pt" \
            "${checkpoint_root}${seed_c}/best.pt" \
        --interaction-backbone visnet \
        --label-mode active_interaction \
        --positive-interactions \
            hydrophobic_contact hydrogen_bond salt_bridge pi_stacking pi_cation \
        --score-interactions \
            hydrophobic_contact hydrogen_bond salt_bridge pi_stacking pi_cation \
        --candidate-radius 999 \
        --stage2-feature-mode typed_multiscale \
        --stage2-classifier logreg \
        --seed "$localizer_seed" \
        --max-train-graphs 5000 \
        --max-internal-val-graphs 1000 \
        --max-external-val-graphs 0 \
        --max-external-test-graphs 0 \
        --predictions-output "$out_dir/predictions.npz" \
        --out-dir "$out_dir" \
        --table "outputs/tables/table_postfreeze_replicate_${replicate}.md" \
        --device cuda \
        --finite-representation-guard \
        > "logs/postfreeze_replicate_${replicate}.log" 2>&1
}

run_replicate 0 A 3401 2401 2402 2403 &
pid_a=$!
run_replicate 2 B 3402 2404 2405 2406 &
pid_b=$!
run_replicate 4 C 3403 2407 2408 2409 &
pid_c=$!

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
wait "$pid_c" || status=1

echo "replicate_A_exit=$(test -f outputs/postfreeze_holdout/replicate_A/metrics.json && echo 0 || echo 1)"
echo "replicate_B_exit=$(test -f outputs/postfreeze_holdout/replicate_B/metrics.json && echo 0 || echo 1)"
echo "replicate_C_exit=$(test -f outputs/postfreeze_holdout/replicate_C/metrics.json && echo 0 || echo 1)"
exit "$status"
