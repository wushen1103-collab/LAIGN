"""Strong two-scalar nonlinear fusion; tune only on internal validation."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "8")
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from .fusion_metrics import per_complex, mean_summary, paired_summary

from .runtime import ROOT, BASE, load_archive, canonical_id


def feature(z, split):
    return np.column_stack((z[f"{split}__score__raw_visnet"],
                            np.exp(z[f"{split}__score__raw_distance"] / 12))).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cohort", choices=["plinder", "postfreeze"], default="plinder")
    a = p.parse_args()
    out = BASE / (("smoke_" if a.smoke else "full_") + "twoinput_" + a.cohort)
    out.mkdir(parents=True, exist_ok=True)
    paths = ([ROOT / f"outputs/reference/plinder/seed{seed}/predictions.npz" for seed in (2401, 2402, 2403)]
             if a.cohort == "plinder" else [ROOT / f"outputs/reference/postfreeze/replicate_{r}/predictions.npz" for r in "ABC"])
    if a.smoke:
        paths = paths[:1]
    grid = [dict(num_leaves=7, min_child_samples=100, reg_lambda=1.0),
            dict(num_leaves=15, min_child_samples=100, reg_lambda=1.0),
            dict(num_leaves=31, min_child_samples=100, reg_lambda=1.0),
            dict(num_leaves=15, min_child_samples=300, reg_lambda=10.0)]
    if a.smoke:
        grid = grid[:2]
    predictions, trace = {}, []
    start = time.time()
    reference = None
    for run, path in enumerate(paths):
        z = load_archive(path)
        xtr, xva = feature(z, "train"), feature(z, "internal_val")
        ytr, yva = z["train__labels"], z["internal_val__labels"]
        if a.smoke:
            train_mask = z["train__sample_index"] < 50
            val_mask = z["internal_val__sample_index"] < 30
            xtr, ytr, xva, yva = xtr[train_mask], ytr[train_mask], xva[val_mask], yva[val_mask]
        best = None
        for cfg in grid:
            for weight in (None, "balanced"):
                model = LGBMClassifier(**cfg, n_estimators=30 if a.smoke else 200, learning_rate=0.05,
                                       n_jobs=8, random_state=2401 + run, verbosity=-1,
                                       class_weight=weight, deterministic=True, force_col_wise=True)
                model.fit(xtr, ytr)
                score = model.predict_proba(xva)[:, 1]
                value = float(average_precision_score(yva, score))
                row = dict(run=run, config=cfg, class_weight=weight,
                           internal_validation_pooled_AP=value, training_residues=len(ytr),
                           validation_residues=len(yva))
                trace.append(row)
                print(json.dumps(row), flush=True)
                if best is None or value > best[0]:
                    best = (value, model, row)
        joblib.dump(best[1], out / f"model_run{run}.joblib")
        (out / f"selection_run{run}.json").write_text(json.dumps(best[2], indent=2))
        for split in ("external_val", "external_test"):
            if not len(z[f"{split}__labels"]):
                continue
            if reference is None:
                reference = {}
            for suffix in ("labels", "sample_index", "sample_ids", "score__raw_distance"):
                key = split + "__" + suffix
                if key not in reference:
                    reference[key] = z[key]
                elif not np.array_equal(reference[key], z[key]):
                    raise ValueError("Nonaligned external archives: " + key)
            for name, score in {
                "Nonlinear two-input fusion": best[1].predict_proba(feature(z, split))[:, 1],
                "LAIGN": z[f"{split}__score__trained_stage2"],
                "Fixed fusion": z[f"{split}__score__fixed_hybrid"],
                "Distance": z[f"{split}__score__raw_distance"],
            }.items():
                predictions.setdefault((split, name), []).append(score.astype(np.float64))
        z.close()
    rows, arrays = [], dict(reference)
    for (split, method), values in predictions.items():
        score = np.mean(values, axis=0)
        arrays[split + "__score__" + method] = score
        y, g, ids = [reference[split + "__" + k] for k in ("labels", "sample_index", "sample_ids")]
        cohort = ("PLINDER-785 test" if split == "external_test" else "PLINDER-551 validation") if a.cohort == "plinder" else "Post-freeze-70"
        rows.extend(per_complex(y, score, g, ids, cohort, method))
        rows.extend(per_complex(y, score, g, ids, cohort + " / 6A", method,
                                -reference[split + "__score__raw_distance"], 6.0))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_complex.csv", index=False)
    summary = mean_summary(frame)
    summary.to_csv(out / "summary.csv", index=False)
    paired_summary(frame, bootstrap=200 if a.smoke else 10000).to_csv(out / "paired_effects.csv", index=False)
    np.savez_compressed(out / "external_predictions.npz", **arrays)
    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    (out / "completion.json").write_text(json.dumps({"status": "complete", "smoke": a.smoke,
                                                    "runs": len(paths), "seconds": time.time() - start,
                                                    "selection_endpoint": "internal_validation_pooled_AP",
                                                    "external_tuning": False}, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
