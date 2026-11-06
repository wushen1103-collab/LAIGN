"""Unified frozen-cohort, annotation-transfer and fusion-conflict evaluations."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "8")
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .fusion_metrics import graph_metrics, mean_summary, paired_summary,clustered_paired_summary
from .feature_data import cache_path

from .runtime import ROOT, BASE, load_archive, canonical_id


def add_methods(z, extra):
    for key in extra.files:
        if "__score__" in key:
            if key not in z:
                z[key] = extra[key]
            elif not np.array_equal(z[key], extra[key]):
                # Original frozen scores are authoritative in this evaluation.
                continue
        elif key in z and not np.array_equal(z[key], extra[key]):
            raise ValueError("Evidence alignment differs: " + key)


def main():
    out = BASE / "unified_external_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(ROOT / "source_data/cohorts/plinder200.csv", usecols=["sample_id", "split"])
    selected_ids = set(selected.sample_id.astype(str))
    chemistry = pd.read_csv(ROOT / "source_data/cohorts/chemical_strata.csv")
    low_ids = set(chemistry.loc[chemistry.dataset.eq("PLINDER") & chemistry.max_train_identity_percent.lt(30), "sample_id"])
    joint_ids = set(chemistry.loc[chemistry.dataset.eq("PLINDER") & chemistry.max_train_identity_percent.lt(30)
                     & chemistry.max_train_tanimoto.lt(.5)
                     & (chemistry.scaffold_unseen.eq(True) | chemistry.ringless.eq(True)), "sample_id"])
    counts = {"PLINDER-200": len(selected_ids), "low_identity": len(low_ids), "joint_OOD": len(joint_ids)}
    assert len(selected_ids) == 200 and len(low_ids) == 54 and len(joint_ids) == 25, counts
    prolif = pd.read_csv(ROOT / "source_data/validation/prolif_residue_predictions.csv.gz")
    by_prolif = {str(k): v for k, v in prolif.groupby("sample_id", sort=False)}
    metrics, annotations, overlaps, conflicts, branch_states = [], [], [], [], []
    for cohort in ("plinder", "postfreeze"):
        path = BASE / ("full_twoinput_" + cohort) / "external_predictions.npz"
        z = dict(load_archive(path))
        original_paths = ([ROOT / f'outputs/reference/plinder/seed{seed}/predictions.npz' for seed in (2401,2402,2403)]
                          if cohort == 'plinder' else [ROOT / f'outputs/reference/postfreeze/replicate_{r}/predictions.npz' for r in 'ABC'])
        scalar_runs = {}
        for source in original_paths:
            with load_archive(source) as original:
                for split in ('external_val','external_test'):
                    if split+'__labels' not in z:continue
                    for suffix in ('labels','sample_index','sample_ids'):
                        assert np.array_equal(z[split+'__'+suffix],original[split+'__'+suffix])
                    scalar_runs.setdefault(split,[]).append(original[split+'__score__raw_visnet'].astype(np.float64))
        for split, values in scalar_runs.items():
            z[split+'__score__Scalar ViSNet (max probability)']=np.mean(values,axis=0)
        for folder in ("full_feature_original", "deepsets_original", "branch_original", "refit_original") if cohort == "plinder" else ("full_feature_postfreeze", "deepsets_postfreeze", "refit_postfreeze"):
            added = BASE / folder / "external_predictions.npz"
            if added.exists():
                add_methods(z, load_archive(added))
        for split in ("external_val", "external_test"):
            if split + "__labels" not in z:
                continue
            labels, groups, ids = [z[split + "__" + k] for k in ("labels", "sample_index", "sample_ids")]
            distances = -z[split + "__score__raw_distance"]
            methods = {k.split("__score__",1)[1]: v for k, v in z.items()
                       if k.startswith(split + "__score__") and not k.endswith('__score__raw_distance')}
            boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                sid = str(ids[int(groups[start])])
                y, d = labels[start:stop], distances[start:stop]
                tests = (["PLINDER-785 test"] if split == "external_test" else []) if cohort == "plinder" else ["Post-freeze-70"]
                if cohort == "plinder" and sid in selected_ids:
                    tests += ["PLINDER-200", "PLINDER-200 / 6A"]
                if cohort == "plinder" and sid in low_ids:
                    tests += ["Receptor identity <30%"]
                if cohort == "plinder" and sid in joint_ids:
                    tests += ["Joint protein-ligand OOD"]
                for test in tests:
                    mask = d <= 6 if test.endswith("6A") else np.ones(len(y), dtype=bool)
                    for method, values in methods.items():
                        metrics.append(dict(cohort=test, sample_id=sid, method=method,
                                            residues=int(mask.sum()), positives=int(y[mask].sum()),
                                            **graph_metrics(y[mask],values[start:stop][mask])))
                if cohort == "plinder" and sid in selected_ids and sid in by_prolif:
                    alternative = by_prolif[sid]
                    if len(alternative) != len(y) or not np.array_equal(alternative.label.to_numpy(), y):
                        raise ValueError("ProLIF residue alignment failed: " + sid)
                    with load_archive(cache_path(BASE/'cache_original',sid)) as cached:
                        keys=cached['residue_keys'].astype(str)
                    actual=alternative[['chain','residue_number','residue_type']].fillna('').astype(str).to_numpy()
                    def normalized_residue_number(value):
                        try:
                            number=float(value)
                            if np.isfinite(number) and number.is_integer():return str(int(number))
                        except ValueError:pass
                        return value.strip()
                    actual[:,1]=[normalized_residue_number(v) for v in actual[:,1]]
                    if not np.array_equal(actual,keys):
                        raise ValueError('ProLIF explicit residue keys differ: '+sid)
                    yy = (alternative.score.to_numpy() > 0).astype(np.int8)
                    union = np.count_nonzero((y > 0) | (yy > 0))
                    overlap = np.count_nonzero((y > 0) & (yy > 0))
                    overlaps.append(dict(sample_id=sid, split=split, residues=len(y),
                                         plip_positives=int(y.sum()), prolif_positives=int(yy.sum()),
                                         jaccard=overlap / union if union else 1.0,
                                         disagreement_residues=int(np.count_nonzero(y != yy))))
                    for endpoint, truth in [("PLIP-defined", y), ("ProLIF-defined", yy)]:
                        for radius in (None,6.):
                            mask = d <= radius if radius else np.ones(len(y), dtype=bool)
                            name = endpoint + (" / 6A" if radius else " / all residues")
                            suffix = " / external test" if split == "external_test" else " / external validation"
                            for method, values in methods.items():
                                annotations.append(dict(cohort=name+suffix, sample_id=sid, method=method,
                                                        residues=int(mask.sum()), positives=int(truth[mask].sum()),
                                                        **graph_metrics(truth[mask],values[start:stop][mask])))
                            if endpoint == "ProLIF-defined":
                                annotations.append(dict(cohort=name+suffix, sample_id=sid,
                                                        method="PLIP label transfer reference", residues=int(mask.sum()),
                                                        positives=int(truth[mask].sum()), **graph_metrics(truth[mask],y[mask].astype(float))))
                if cohort == "plinder" and split == "external_test":
                    # Top-P lists expose whether the chemistry signal corrects or introduces proximity errors.
                    p = int(y.sum())
                    near = set(np.argsort(d,kind="stable")[:p])
                    fused = set(np.argsort(-methods["LAIGN"][start:stop],kind="stable")[:p])
                    scalar_key = "raw_visnet"
                    for item in fused - near:
                        conflicts.append(dict(sample_id=sid, event="added_by_LAIGN", positive=int(y[item]),distance=float(d[item])))
                    for item in near - fused:
                        conflicts.append(dict(sample_id=sid, event="removed_by_LAIGN", positive=int(y[item]),distance=float(d[item])))
                    geometry_key='Geometry only (34 features)'
                    chemistry_key='Chemistry only (21 features)'
                    if geometry_key in methods and chemistry_key in methods:
                        geometry=np.zeros(len(y),dtype=bool);chemistry_selected=np.zeros(len(y),dtype=bool)
                        geometry[np.argsort(-methods[geometry_key][start:stop],kind='stable')[:p]]=True
                        chemistry_selected[np.argsort(-methods[chemistry_key][start:stop],kind='stable')[:p]]=True
                        selected_mask=np.zeros(len(y),dtype=bool);selected_mask[list(fused)]=True
                        for name,state_mask in [('both',geometry & chemistry_selected),('geometry_only',geometry & ~chemistry_selected),
                                                ('chemistry_only',chemistry_selected & ~geometry),('neither',~geometry & ~chemistry_selected)]:
                            branch_states.append(dict(sample_id=sid,state=name,residues=int(state_mask.sum()),
                                positives=int(y[state_mask].sum()),selected_by_LAIGN=int(selected_mask[state_mask].sum()),
                                selected_positive_by_LAIGN=int((selected_mask & state_mask & (y>0)).sum())))
    frame = pd.DataFrame(metrics)
    frame.to_csv(out/"per_complex.csv",index=False)
    mean_summary(frame).to_csv(out/"summary.csv",index=False)
    paired_summary(frame).to_csv(out/"paired_effects.csv",index=False)
    if frame.method.eq('LAIGN refit (73 features)').any():
        refit_effects=paired_summary(frame,reference='LAIGN refit (73 features)')
        refit_effects.to_csv(out/'same_cache_reference_paired_effects.csv',index=False)
        removal=refit_effects[refit_effects.comparator.str.startswith('Without ')].copy()
        removal['holm_adjusted_signflip_p']=np.nan
        for _,part in removal.groupby(['cohort','metric']):
            order=part.signflip_p.sort_values().index
            adjusted=np.minimum(1.,np.maximum.accumulate(removal.loc[order,'signflip_p'].to_numpy()*np.arange(len(order),0,-1)))
            removal.loc[order,'holm_adjusted_signflip_p']=adjusted
        removal.to_csv(out/'type_removal_effects.csv',index=False)
    temporal_manifest=pd.read_csv(ROOT/'source_data/cohorts/postfreeze_receptors.csv',low_memory=False)
    temporal_manifest=temporal_manifest[temporal_manifest.supervision_split.eq('external_test')]
    receptor_groups=temporal_manifest.set_index('sample_id').receptor_sequence.astype(str)
    assert receptor_groups.nunique()==56
    clustered_paired_summary(frame[frame.cohort.eq('Post-freeze-70')],receptor_groups).to_csv(out/'postfreeze_receptor_grouped_effects.csv',index=False)
    annotation = pd.DataFrame(annotations)
    annotation.to_csv(out/"annotation_per_complex.csv",index=False)
    mean_summary(annotation).to_csv(out/"annotation_summary.csv",index=False)
    paired_summary(annotation).to_csv(out/"annotation_paired_effects.csv",index=False)
    pd.DataFrame(overlaps).to_csv(out/"annotation_label_overlap.csv",index=False)
    missing=selected_ids-set(row['sample_id'] for row in overlaps)
    pd.DataFrame([dict(sample_id=sid,reason='No existing ProLIF prediction archive for this frozen sample')
                  for sid in sorted(missing)],columns=['sample_id','reason']).to_csv(out/'annotation_exclusions.csv',index=False)
    conflict = pd.DataFrame(conflicts)
    conflict.to_csv(out/"fusion_topP_conflicts.csv",index=False)
    conflict.groupby("event").agg(residues=("positive","size"), positives=("positive","sum"),
                                 positive_fraction=("positive","mean")).reset_index().to_csv(out/"fusion_conflict_summary.csv",index=False)
    if branch_states:
        state_frame=pd.DataFrame(branch_states)
        state_frame.to_csv(out/'fusion_branch_states_per_complex.csv',index=False)
        aggregate=state_frame.groupby('state')[['residues','positives','selected_by_LAIGN','selected_positive_by_LAIGN']].sum().reset_index()
        aggregate['positive_fraction']=aggregate.positives/aggregate.residues
        aggregate['LAIGN_selected_positive_fraction']=aggregate.selected_positive_by_LAIGN/aggregate.selected_by_LAIGN.replace(0,np.nan)
        aggregate.to_csv(out/'fusion_branch_agreement_summary.csv',index=False)
    identity_columns = ["dataset","sample_id","max_train_identity_percent","max_train_tanimoto","similarity_bin",
                        "structure_resolved","scaffold_unseen","ringless"]
    chemistry[identity_columns].to_csv(out/"frozen_ood_metadata.csv",index=False)
    temporal = chemistry[chemistry.dataset.eq("Temporal")]
    distribution = temporal[["max_train_identity_percent","max_train_tanimoto"]].describe(percentiles=[.1,.25,.5,.75,.9]).to_dict()
    (out/"completion.json").write_text(json.dumps(dict(status="complete", counts=counts,
                                                       annotation_complexes=len(overlaps),
                                                       annotation_requested_complexes=len(selected_ids),
                                                       missing_annotation_complexes=len(missing),
                                                       annotation_source="ProLIF rule-based coordinate annotations; not experimental biological ground truth",
                                                       annotation_alignment='Explicit chain, residue number and residue type plus PLIP labels',
                                                       temporal_distribution=distribution,
                                                       bootstrap_unit="complex",bootstrap_replicates=10000),indent=2))
    print(mean_summary(frame).to_string(index=False))
    print(mean_summary(annotation).to_string(index=False))


if __name__ == "__main__":
    main()
