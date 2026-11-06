"""Cache exact original evidence once for fusion controls and channel diagnostics."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "8")
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from .runtime import ROOT, BASE, load_archive, canonical_id


def relocated(value):
    path = Path(str(value))
    return str(path if path.is_absolute() else ROOT / path)


def cache_name(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24] + ".npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["original", "disjoint", "postfreeze", "disjoint_postfreeze"], default="original")
    p.add_argument("--pipeline", choices=["A","B","C"], default="A")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.model import train_residue_localizer as stage2
    torch.set_num_threads(8)
    device = torch.device(os.environ.get("LAIGN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    names = stage2.DEFAULT_ACTIVE_INTERACTIONS
    indices = stage2.contact_eval.active_indices(names)
    args = SimpleNamespace(
        split_manifest=str(ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"),
        biolip_interactions=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_interaction_labels.csv.gz"),
        plinder_interactions=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
        biolip_site_labels=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_site_labels.csv.gz"),
        plinder_site_labels=str(ROOT / "data/processed/plinder/plinder_plip_site_labels.csv.gz"),
        positive_interactions=names, score_interactions=names, label_mode="active_interaction",
        max_train_graphs=5000, max_internal_val_graphs=1000,
        max_external_val_graphs=0, max_external_test_graphs=0, seed=2401,
        candidate_radius=999.0, stage2_feature_mode="typed_multiscale",
        ensemble_uncertainty_features=False, finite_representation_guard=False,
        representation_clip=0.0, visnet_control="normal", control_seed=0,
    )
    if a.mode in ("postfreeze", "disjoint_postfreeze"):
        args.split_manifest = str(ROOT / "outputs/reference/postfreeze/temporal_blind_ccd_manifest.csv.gz")
        args.plinder_interactions = str(ROOT / "outputs/reference/postfreeze/temporal_blind_plip_interactions.csv.gz")
        args.plinder_site_labels = str(ROOT / "outputs/reference/postfreeze/temporal_blind_site_labels.csv.gz")
        args.seed = 3401 if a.mode == "postfreeze" else 2401
        args.finite_representation_guard = True
    manifest = stage2.select_manifest(args)
    if a.mode.startswith("disjoint"):
        config = json.loads((BASE / "disjoint_protocol.json").read_text())
        all_rows = pd.read_csv(args.split_manifest, low_memory=False)
        held = all_rows[all_rows.sample_id.isin(config["stage2_train_ids"])].copy()
        held["supervision_split"] = "train"
        manifest = pd.concat([manifest[~manifest.supervision_split.eq("train")], held]).sort_values(["supervision_split", "sample_id"])
    else:
        # Reuse the union of the three existing fit subsets to support paired runs.
        all_rows = pd.read_csv(args.split_manifest, low_memory=False)
        if a.mode == "original":
            archives = [ROOT / f"outputs/reference/plinder/seed{s}/predictions.npz" for s in (2401,2402,2403)]
        else:
            archives = [ROOT / f"outputs/reference/postfreeze/replicate_{a.pipeline}/predictions.npz"]
        chosen = set()
        for archive in archives:
            with load_archive(archive) as z:
                for split in ("train", "internal_val"):
                    chosen.update(z[f"{split}__sample_ids"].astype(str))
        manifest = pd.concat([all_rows[all_rows.sample_id.isin(chosen)],
                              manifest[manifest.supervision_split.isin(["external_val", "external_test"])]])
        manifest = manifest.drop_duplicates("sample_id").sort_values(["supervision_split", "sample_id"])
    for key in ("plip_pdb", "plip_report", "receptor_path", "ligand_path"):
        if key in manifest:
            manifest[key] = manifest[key].map(relocated)
    if a.smoke:
        manifest = manifest.groupby("supervision_split", sort=False).head(3)
    manifest = manifest.iloc[a.shard::a.shards]
    positive, serials = stage2.load_label_context(args)
    pair_lookup, _ = stage2.pair_features.group_interactions(pd.read_csv(args.plinder_interactions), ["system_id", "ligand_id"])
    if a.mode.startswith("disjoint"):
        checkpoints = [BASE / f"full_scorer_seed{s}/checkpoints/best.pt" for s in (2401,2402,2403)]
    elif a.mode == "postfreeze":
        metrics=json.loads((ROOT/f"outputs/reference/postfreeze/replicate_{a.pipeline}/metrics.json").read_text())
        checkpoints=[ROOT/relocated(p) if not Path(relocated(p)).is_absolute() else Path(relocated(p))
                     for p in [metrics['checkpoint'],*metrics['ensemble_checkpoints']]]
    else:
        checkpoints = [ROOT / f"checkpoints/scorer_seed{s}/best.pt" for s in (2401,2402,2403)]
    models = [stage2.contact_eval.load_model(str(path), args, device)[0] for path in checkpoints]
    cache_mode='disjoint' if a.mode.startswith('disjoint') else a.mode
    out = BASE / ("cache_" + cache_mode + ('_'+a.pipeline if a.mode=='postfreeze' else ''))
    out.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    start = time.time()
    for index, row in enumerate(manifest.to_dict("records")):
        public_id = canonical_id(row["sample_id"])
        target = out / cache_name(public_id)
        if target.exists():
            try:
                with load_archive(target) as existing:
                    assert str(existing['sample_id']) == public_id
                    existing['x']
                records.append({"sample_id": public_id, "split": row["supervision_split"], "file": target.name, "cached": True})
                continue
            except (OSError,ValueError,EOFError,zipfile.BadZipFile):
                print('Rebuilding incomplete generated cache '+target.name,flush=True)
        record, error = stage2.build_candidate_record(row, positive, serials, 999.0)
        if error:
            errors.append({"sample_id": public_id, "error": error})
            continue
        data = record["data"].clone().to(device)
        with torch.no_grad():
            pieces = [torch.sigmoid(model(data)[:, indices]).cpu().numpy() for model in models]
        probs = np.mean(pieces, axis=0)
        if not np.isfinite(probs).all():
            raise FloatingPointError(row["sample_id"])
        residue_index = record["residue_for_pair"]
        scalar = np.zeros(len(record["labels"]), dtype=np.float64)
        np.maximum.at(scalar, residue_index, probs.max(axis=1))
        extra, extra_names = stage2.typed_multiscale_features(record, probs, names)
        split_data = dict(distances=record["distances"], visnet=scalar,
                          extra_features=extra, extra_feature_names=extra_names)
        features, feature_names = stage2.feature_matrix(split_data, "stage2", "typed_multiscale")
        assert features.shape[1] == 73
        residue_atoms, ligand_atoms = stage2.selected_ligand_atoms(row, serials)
        occupied, keys = set(), []
        for key in stage2.unique_standard_residues(residue_atoms):
            atoms = stage2.graph_builder.get_residue_atoms(residue_atoms, key)
            if not atoms:
                continue
            xyz = np.stack([atom["xyz"] for atom in atoms]).mean(axis=0).astype(np.float32)
            coord = tuple(xyz.tolist())
            if coord in occupied:
                continue
            occupied.add(coord)
            keys.append(key)
        assert len(keys) == len(record["labels"])
        pair_positions = data.pos[data.pair_index[1]].detach().cpu().numpy()
        by_xyz = {tuple(np.asarray(atom["xyz"],dtype=np.float32).tolist()): atom["serial"] for atom in ligand_atoms}
        pair_serials = np.asarray([by_xyz[tuple(x.tolist())] for x in pair_positions], dtype=np.int32)
        ypair = np.zeros(probs.shape, dtype=np.int8)
        if row["source"] == "plinder":
            prefix = (row["system_id"], row["ligand_id"])
            zero = np.zeros(len(stage2.train_visnet.INTERACTION_TYPES), dtype=np.int8)
            for pair_number, (res, serial) in enumerate(zip(residue_index, pair_serials)):
                lookup_key = prefix + tuple(keys[res]) + (int(serial),)
                ypair[pair_number] = pair_lookup.get(lookup_key, zero)[indices]
            y_reconstructed = np.zeros(len(keys), dtype=np.int8)
            np.maximum.at(y_reconstructed, residue_index, ypair.max(axis=1))
            if not np.array_equal(y_reconstructed, record["labels"]):
                raise ValueError("Pair-to-residue label mapping differs: " + str(row["sample_id"]))
        temporary=target.with_name(target.stem+f'.tmp.{os.getpid()}.npz')
        np.savez_compressed(temporary, sample_id=np.asarray(public_id), split=np.asarray(row["supervision_split"]),
                            y=record["labels"], distance=record["distances"], x=features,
                            pair_prob=probs.astype(np.float32), pair_distance=record["pair_distances"],
                            residue_for_pair=residue_index.astype(np.int32), pair_label=ypair,
                            residue_keys=np.asarray(keys, dtype=str), pair_serial=pair_serials,
                            feature_names=np.asarray(feature_names), interaction_names=np.asarray(names))
        os.replace(temporary,target)
        records.append({"sample_id": public_id, "split": row["supervision_split"], "file": target.name,
                        "residues": len(record["labels"]), "pairs": len(probs)})
        if index % 25 == 0:
            print(json.dumps({"processed": index + 1, "total": len(manifest), "seconds": time.time() - start}), flush=True)
    suffix = f"{'smoke_' if a.smoke else ''}shard{a.shard}of{a.shards}"
    pd.DataFrame(records).to_csv(out / (suffix + "_index.csv"), index=False)
    (out / (suffix + "_completion.json")).write_text(json.dumps({"status": "complete", "records": len(records),
                                                               "errors": errors, "seconds": time.time() - start,
                                                               "checkpoints": [str(p) for p in checkpoints]}, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} requested graphs failed to cache")


if __name__ == "__main__":
    main()
