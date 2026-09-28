#!/usr/bin/env python3
"""Audit ligand chemical OOD for the frozen LAIGN evaluation sets.

The training chemical reference is the set of BioLiP2 ligand CCD codes in the
structure-supervision training split. Chemical identity comes from the wwPDB
Chemical Component Dictionary rather than bond-order-free PDB ligand files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVES = [
    ROOT / "outputs/stage2_statistics_statistics_laign_full_seed2401/predictions.npz",
    ROOT / "outputs/stage2_statistics_statistics_laign_full_seed2402/predictions.npz",
    ROOT / "outputs/stage2_statistics_statistics_laign_full_seed2403/predictions.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"),
    )
    parser.add_argument(
        "--ccd-sdf",
        default=str(
            ROOT
            / "experiments/validation/data/raw/ccd/components-pub.sdf.gz"
        ),
    )
    parser.add_argument(
        "--mmseqs",
        default=str(ROOT / "outputs/leakage_audit_leakage/query_vs_train_mmseqs.tsv"),
    )
    parser.add_argument("--archives", nargs="+", default=[str(path) for path in DEFAULT_ARCHIVES])
    parser.add_argument(
        "--temporal-predictions",
        default=str(ROOT / "outputs/postfreeze_holdout/laign_temporal_predictions.npz"),
    )
    parser.add_argument(
        "--temporal-manifest",
        default=str(ROOT / "outputs/postfreeze_holdout/postfreeze_split_manifest.csv.gz"),
    )
    parser.add_argument(
        "--temporal-identity",
        default=str(ROOT / "outputs/postfreeze_holdout/sequence_identity_audit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/validation/results/chemical_ood"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8241)
    return parser.parse_args()


def canonical_molecule(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def load_ccd(path: Path, wanted: set[str]) -> tuple[dict[str, Chem.Mol], dict[str, int]]:
    molecules: dict[str, Chem.Mol] = {}
    seen = 0
    failed = 0
    with gzip.open(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=True)
        for mol in supplier:
            seen += 1
            if mol is None:
                failed += 1
                continue
            code = mol.GetProp("_Name").strip().upper() if mol.HasProp("_Name") else ""
            if code not in wanted or code in molecules:
                continue
            cleaned = canonical_molecule(mol)
            if cleaned is None:
                failed += 1
                continue
            molecules[code] = cleaned
    return molecules, {"records_seen": seen, "parse_failures": failed, "wanted_loaded": len(molecules)}


def fingerprint(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def parse_mmseqs_max_identity(path: Path) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            header = fields[0].split("|", 2)
            if len(header) < 2:
                continue
            values[header[1]] = max(values[header[1]], float(fields[2]))
    return dict(values)


def graph_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.sum() == 0:
        return math.nan
    return float(average_precision_score(labels, scores))


def load_plinder_graph_metrics(archives: list[Path], split: str = "external_test") -> pd.DataFrame:
    loaded = [np.load(path, allow_pickle=True) for path in archives]
    sample_ids = loaded[0][f"{split}__sample_ids"].astype(str)
    sample_index = loaded[0][f"{split}__sample_index"].astype(np.int32)
    labels = loaded[0][f"{split}__labels"].astype(np.int8)
    for archive in loaded[1:]:
        if not np.array_equal(sample_ids, archive[f"{split}__sample_ids"].astype(str)):
            raise RuntimeError("Prediction archives have different sample ordering")
        if not np.array_equal(labels, archive[f"{split}__labels"].astype(np.int8)):
            raise RuntimeError("Prediction archives have different labels")
    methods = {
        "distance": np.mean(
            [archive[f"{split}__score__raw_distance"].astype(np.float64) for archive in loaded], axis=0
        ),
        "fixed_fusion": np.mean(
            [archive[f"{split}__score__fixed_hybrid"].astype(np.float64) for archive in loaded], axis=0
        ),
        "laign": np.mean(
            [archive[f"{split}__score__trained_stage2"].astype(np.float64) for archive in loaded], axis=0
        ),
    }
    rows = []
    for index, sample_id in enumerate(sample_ids):
        mask = sample_index == index
        row = {"dataset": "PLINDER", "sample_id": sample_id, "n_residues": int(mask.sum()), "n_positive": int(labels[mask].sum())}
        for method, score in methods.items():
            row[f"graph_ap_{method}"] = graph_ap(labels[mask], score[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def load_temporal_graph_metrics(path: Path) -> pd.DataFrame:
    archive = np.load(path, allow_pickle=True)
    sample_ids = archive["sample_ids"].astype(str)
    sample_index = archive["sample_index"].astype(np.int32)
    labels = archive["label"].astype(np.int8)
    methods = {
        "distance": archive["raw_distance"].astype(np.float64),
        "fixed_fusion": np.mean(
            [archive[f"fixed_fusion_{name}"].astype(np.float64) for name in "ABC"], axis=0
        ),
        "laign": np.mean([archive[f"laign_{name}"].astype(np.float64) for name in "ABC"], axis=0),
    }
    rows = []
    for index, sample_id in enumerate(sample_ids):
        mask = sample_index == index
        row = {"dataset": "Temporal", "sample_id": sample_id, "n_residues": int(mask.sum()), "n_positive": int(labels[mask].sum())}
        for method, score in methods.items():
            row[f"graph_ap_{method}"] = graph_ap(labels[mask], score[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def assign_similarity_bin(value: float) -> str:
    if not math.isfinite(value):
        return "unresolved"
    if value < 0.30:
        return "[0.00,0.30)"
    if value < 0.50:
        return "[0.30,0.50)"
    if value < 0.70:
        return "[0.50,0.70)"
    return "[0.70,1.00]"


def bootstrap_summary(frame: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float | int]:
    frame = frame.dropna(subset=["graph_ap_laign", "graph_ap_fixed_fusion", "graph_ap_distance"])
    n = len(frame)
    if n == 0:
        return {"n": 0}
    values = frame[["graph_ap_laign", "graph_ap_fixed_fusion", "graph_ap_distance"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = values[draws].mean(axis=1)
    delta_fixed = boot[:, 0] - boot[:, 1]
    delta_distance = boot[:, 0] - boot[:, 2]
    return {
        "n": n,
        "laign_mean": float(values[:, 0].mean()),
        "laign_ci_low": float(np.quantile(boot[:, 0], 0.025)),
        "laign_ci_high": float(np.quantile(boot[:, 0], 0.975)),
        "fixed_fusion_mean": float(values[:, 1].mean()),
        "distance_mean": float(values[:, 2].mean()),
        "delta_fixed_mean": float((values[:, 0] - values[:, 1]).mean()),
        "delta_fixed_ci_low": float(np.quantile(delta_fixed, 0.025)),
        "delta_fixed_ci_high": float(np.quantile(delta_fixed, 0.975)),
        "delta_distance_mean": float((values[:, 0] - values[:, 2]).mean()),
        "delta_distance_ci_low": float(np.quantile(delta_distance, 0.025)),
        "delta_distance_ci_high": float(np.quantile(delta_distance, 0.975)),
    }


def summarize_strata(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    definitions: list[tuple[str, str, pd.Series]] = []
    for dataset in ["PLINDER", "Temporal", "Combined"]:
        base = pd.Series(True, index=frame.index) if dataset == "Combined" else frame["dataset"].eq(dataset)
        definitions.append((dataset, "all_resolved", base & frame["max_train_tanimoto"].notna()))
        for name in ["ccd_seen", "ccd_unseen", "scaffold_seen", "scaffold_unseen", "ringless"]:
            definitions.append((dataset, name, base & frame[name].eq(True)))
        for label in ["[0.00,0.30)", "[0.30,0.50)", "[0.50,0.70)", "[0.70,1.00]"]:
            definitions.append((dataset, f"tanimoto_{label}", base & frame["similarity_bin"].eq(label)))
        definitions.append((dataset, "chemical_ood_tanimoto_lt_0.5", base & frame["max_train_tanimoto"].lt(0.5)))
        definitions.append(
            (
                dataset,
                "joint_ood_seq_lt30_tanimoto_lt0.5_scaffold_unseen",
                base
                & frame["max_train_identity_percent"].lt(30.0)
                & frame["max_train_tanimoto"].lt(0.5)
                & (frame["scaffold_unseen"].eq(True) | frame["ringless"].eq(True)),
            )
        )
    rows = []
    for offset, (dataset, stratum, mask) in enumerate(definitions):
        stats = bootstrap_summary(frame.loc[mask], n_boot, seed + offset)
        rows.append({"dataset": dataset, "stratum": stratum, **stats})
    return pd.DataFrame(rows)


def fmt_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def write_markdown(frame: pd.DataFrame, strata: pd.DataFrame, audit: dict[str, object], path: Path) -> None:
    lines = [
        "# Validation Ligand Chemical and Scaffold OOD Audit",
        "",
        "## Protocol",
        "",
        "- Training chemistry: all BioLiP2 ligands in the frozen structure-supervision training split.",
        "- Molecular identity: wwPDB Chemical Component Dictionary structures.",
        "- Similarity: ECFP4/Morgan radius 2, 2048 bits, Tanimoto coefficient.",
        "- Scaffold: canonical Bemis-Murcko scaffold; acyclic ligands are reported separately.",
        "- Performance: per-complex AP followed by a complex-level mean; intervals use paired complex bootstrap.",
        "- Joint OOD: max train protein identity <30%, max train-ligand Tanimoto <0.5, and an unseen non-empty scaffold or an acyclic ligand.",
        "",
        "## Data Audit",
        "",
        f"- RDKit: `{audit['rdkit_version']}`",
        f"- Training ligand codes: {audit['training_codes']}",
        f"- Training structures resolved: {audit['training_structures_resolved']}",
        f"- Evaluation complexes: {audit['evaluation_complexes']}",
        f"- Evaluation structures resolved: {audit['evaluation_structures_resolved']}",
        "",
        "## Main Results",
        "",
        "Values are mean graph AP with 95% complex-bootstrap CI. Delta is LAIGN minus fixed fusion.",
        "",
        "| Dataset | Stratum | N | LAIGN graph AP | Fixed fusion | Distance | Delta vs fixed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    keep = {
        "all_resolved",
        "ccd_unseen",
        "scaffold_unseen",
        "chemical_ood_tanimoto_lt_0.5",
        "joint_ood_seq_lt30_tanimoto_lt0.5_scaffold_unseen",
    }
    for row in strata[strata["stratum"].isin(keep)].to_dict("records"):
        if int(row.get("n", 0) or 0) == 0:
            continue
        lines.append(
            f"| {row['dataset']} | {row['stratum']} | {int(row['n'])} | "
            f"{fmt_ci(row['laign_mean'], row['laign_ci_low'], row['laign_ci_high'])} | "
            f"{row['fixed_fusion_mean']:.4f} | {row['distance_mean']:.4f} | "
            f"{fmt_ci(row['delta_fixed_mean'], row['delta_fixed_ci_low'], row['delta_fixed_ci_high'])} |"
        )
    lines.extend(
        [
            "",
            "## Similarity-Bin Results",
            "",
            "| Dataset | Maximum train-ligand Tanimoto | N | LAIGN graph AP | Delta vs fixed |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in strata[strata["stratum"].str.startswith("tanimoto_")].to_dict("records"):
        if int(row.get("n", 0) or 0) == 0:
            continue
        lines.append(
            f"| {row['dataset']} | {str(row['stratum']).replace('tanimoto_', '')} | {int(row['n'])} | "
            f"{fmt_ci(row['laign_mean'], row['laign_ci_low'], row['laign_ci_high'])} | "
            f"{fmt_ci(row['delta_fixed_mean'], row['delta_fixed_ci_low'], row['delta_fixed_ci_high'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The chemical-OOD result supports generalization only when the paired delta remains positive with a confidence interval excluding zero. Exact CCD novelty alone is weaker evidence than low ECFP4 similarity plus scaffold novelty. Small joint-OOD subsets are reported without extrapolating beyond their observed coverage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, low_memory=False)
    train = manifest[
        manifest["source"].eq("biolip2_nr")
        & manifest["supervision_split"].eq("train")
        & manifest["structure_complete"].eq(1)
    ].copy()
    train_codes = set(train["ligand_ccd_code"].dropna().astype(str).str.upper())
    evaluation = manifest[
        manifest["source"].eq("plinder") & manifest["supervision_split"].eq("external_test")
    ][["sample_id", "pdb_id", "ligand_ccd_code", "ligand_path"]].copy()
    evaluation["dataset"] = "PLINDER"

    temporal_manifest = pd.read_csv(args.temporal_manifest, low_memory=False)
    temporal = temporal_manifest[
        temporal_manifest["source_split"].astype(str).eq("postfreeze")
        & temporal_manifest["supervision_split"].eq("external_test")
    ][["sample_id", "pdb_id", "system_id", "ligand_ccd_code", "ligand_path"]].copy()
    temporal["dataset"] = "Temporal"

    eval_codes = set(evaluation["ligand_ccd_code"].dropna().astype(str).str.upper())
    eval_codes.update(temporal["ligand_ccd_code"].dropna().astype(str).str.upper())
    molecules, ccd_audit = load_ccd(Path(args.ccd_sdf), train_codes | eval_codes)
    train_molecules = {code: molecules[code] for code in train_codes if code in molecules}
    if not train_molecules:
        raise RuntimeError("No training ligand structures were resolved from the CCD")
    train_code_order = sorted(train_molecules)
    train_fps = [fingerprint(train_molecules[code]) for code in train_code_order]
    train_scaffolds = {scaffold_smiles(mol) for mol in train_molecules.values()}
    train_scaffolds.discard("")

    chemical_rows = []
    for record in pd.concat([evaluation, temporal], ignore_index=True).to_dict("records"):
        code = str(record["ligand_ccd_code"]).upper()
        mol = molecules.get(code)
        if mol is None:
            chemical_rows.append({**record, "structure_resolved": False})
            continue
        fp = fingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        best_index = int(np.argmax(similarities))
        scaffold = scaffold_smiles(mol)
        chemical_rows.append(
            {
                **record,
                "structure_resolved": True,
                "max_train_tanimoto": float(similarities[best_index]),
                "nearest_train_ligand_ccd": train_code_order[best_index],
                "bemis_murcko_scaffold": scaffold,
                "ccd_seen": code in train_codes,
                "ccd_unseen": code not in train_codes,
                "ringless": scaffold == "",
                "scaffold_seen": bool(scaffold and scaffold in train_scaffolds),
                "scaffold_unseen": bool(scaffold and scaffold not in train_scaffolds),
            }
        )
    chemistry = pd.DataFrame(chemical_rows)
    chemistry["similarity_bin"] = chemistry["max_train_tanimoto"].apply(
        lambda value: assign_similarity_bin(float(value)) if pd.notna(value) else "unresolved"
    )

    plinder_metrics = load_plinder_graph_metrics([Path(path) for path in args.archives])
    temporal_metrics = load_temporal_graph_metrics(Path(args.temporal_predictions))
    metrics = pd.concat([plinder_metrics, temporal_metrics], ignore_index=True)
    frame = chemistry.merge(metrics, on=["dataset", "sample_id"], how="inner", validate="one_to_one")

    identity = parse_mmseqs_max_identity(Path(args.mmseqs))
    frame["max_train_identity_percent"] = frame["sample_id"].map(identity)
    temporal_identity = pd.read_csv(args.temporal_identity)
    temporal_map = dict(
        zip(temporal_identity["system_id"].astype(str), temporal_identity["max_train_identity_percent"].astype(float))
    )
    temporal_mask = frame["dataset"].eq("Temporal")
    frame.loc[temporal_mask, "max_train_identity_percent"] = frame.loc[temporal_mask, "system_id"].map(temporal_map)
    frame["max_train_identity_percent"] = frame["max_train_identity_percent"].fillna(0.0)

    strata = summarize_strata(frame, args.bootstrap, args.seed)
    audit = {
        "rdkit_version": rdBase.rdkitVersion,
        "fingerprint": {"family": "Morgan/ECFP4", "radius": 2, "bits": 2048, "coefficient": "Tanimoto"},
        "scaffold": "Bemis-Murcko",
        "training_codes": len(train_codes),
        "training_structures_resolved": len(train_molecules),
        "training_structure_coverage": len(train_molecules) / len(train_codes),
        "evaluation_complexes": len(frame),
        "evaluation_structures_resolved": int(frame["structure_resolved"].sum()),
        "ccd_reader": ccd_audit,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "inputs": {
            "manifest": str(Path(args.manifest)),
            "ccd_sdf": str(Path(args.ccd_sdf)),
            "archives": [str(Path(path)) for path in args.archives],
            "temporal_predictions": str(Path(args.temporal_predictions)),
        },
    }
    frame.to_csv(output / "chemical_ood_per_complex.csv", index=False)
    strata.to_csv(output / "chemical_ood_strata.csv", index=False)
    (output / "chemical_ood_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(frame, strata, audit, output / "CHEMICAL_OOD_RESULTS.md")
    print(json.dumps({"output": str(output), **audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
