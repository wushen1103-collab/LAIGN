#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INTERACTION_TYPES = [
    "hydrophobic_contact",
    "hydrogen_bond",
    "water_bridge",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
]
AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
ELEMENTS = ["C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "MG", "MN", "FE", "ZN", "CA", "NA", "K"]
SOURCE_TO_ID = {"biolip2_nr": 0, "plinder": 1}
SPLIT_TO_ID = {"train": 0, "internal_val": 1, "external_val": 2, "external_test": 3}
RING_GEOMETRY_NAMES = ["both_aromatic", "ring_center_distance", "ring_normal_angle", "ring_offset"]
_WORKER_STATE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Build sampled residue-atom pair features for interaction-map training.")
    parser.add_argument(
        "--split-manifest",
        default=str(ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data/processed/structure_supervision/pair_features_smoke"))
    parser.add_argument("--table", default=str(ROOT / "outputs/tables/table_interaction_pair_features_smoke.md"))
    parser.add_argument("--max-train-complexes", type=int, default=6000)
    parser.add_argument("--max-internal-val-complexes", type=int, default=1000)
    parser.add_argument("--max-external-val-complexes", type=int, default=400)
    parser.add_argument("--max-external-test-complexes", type=int, default=400)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--negative-mode", choices=["random", "hard"], default="random")
    parser.add_argument("--negatives-without-positive", type=int, default=32)
    parser.add_argument("--max-negatives-per-complex", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=8)
    parser.add_argument("--include-ring-geometry", action="store_true")
    return parser.parse_args()


def clean_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_resnr(value):
    text = clean_str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def stable_seed(text, seed):
    digest = hashlib.sha1(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def element_from_atom_name(name):
    raw = "".join(ch for ch in str(name).strip().upper() if ch.isalpha())
    if not raw:
        return "UNK"
    if raw[:2] in {"CL", "BR", "MG", "MN", "FE", "ZN", "CA", "NA"}:
        return raw[:2]
    return raw[:1]


def parse_pdb_atoms(path):
    atoms = {}
    residue_atoms = defaultdict(list)
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resnr = line[22:26].strip() + line[26:27].strip()
                xyz = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            element = line[76:78].strip().upper() or element_from_atom_name(name)
            if element in {"*", ""}:
                element = element_from_atom_name(name)
            atom = {
                "serial": serial,
                "name": name,
                "resname": resname,
                "chain": chain,
                "resnr": resnr,
                "xyz": xyz,
                "element": element,
            }
            atoms[serial] = atom
            if line.startswith("ATOM  ") and resname in AA3:
                residue_atoms[(chain, resnr, resname)].append(atom)
                residue_atoms[(chain, resnr, "")].append(atom)
    return atoms, residue_atoms


def text(node, path, default=""):
    found = node.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def binding_site_serials(binding_site):
    raw = text(binding_site, "./mappings/smiles_to_pdb")
    serials = []
    for item in raw.split(","):
        if ":" not in item:
            continue
        try:
            serials.append(int(item.rsplit(":", 1)[1]))
        except ValueError:
            continue
    return serials


def choose_binding_site(row, positive_serials):
    try:
        binding_sites = ET.parse(row["plip_report"]).getroot().findall("./bindingsite")
    except Exception:
        return None, []
    if not binding_sites:
        return None, []
    ligand_code = clean_str(row.get("ligand_ccd_code", ""))
    mapped_target = int(row.get("mapped_ligand_atoms", 0) or 0)

    ranked = []
    for idx, site in enumerate(binding_sites):
        hetid = text(site, "./identifiers/hetid")
        serials = binding_site_serials(site)
        serial_set = set(serials)
        positive_overlap = len(serial_set & positive_serials)
        code_match = int(bool(ligand_code) and hetid == ligand_code)
        count_match = int(mapped_target > 0 and len(serials) == mapped_target)
        ranked.append((positive_overlap, code_match, count_match, len(serials), -idx, site, serials))
    ranked.sort(reverse=True, key=lambda item: item[:5])
    best = ranked[0]
    if best[0] == 0 and best[1] == 0 and positive_serials:
        return None, []
    return best[5], best[6]


def load_labels():
    biolip_i = pd.read_csv(ROOT / "data/processed/biolip2/biolip2_nr_plip_interaction_labels.csv.gz")
    biolip_s = pd.read_csv(ROOT / "data/processed/biolip2/biolip2_nr_plip_site_labels.csv.gz")
    plinder_i = pd.read_csv(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz")
    plinder_s = pd.read_csv(ROOT / "data/processed/plinder/plinder_plip_site_labels.csv.gz")
    return biolip_i, biolip_s, plinder_i, plinder_s


def group_interactions(frame, keys):
    grouped = defaultdict(lambda: np.zeros(len(INTERACTION_TYPES), dtype=np.uint8))
    serials = defaultdict(set)
    type_to_idx = {name: idx for idx, name in enumerate(INTERACTION_TYPES)}
    for row in frame.to_dict("records"):
        sample_key = tuple(row[key] for key in keys)
        residue_key = (
            clean_str(row["protein_chain"]),
            clean_resnr(row["protein_residue_number"]),
            clean_str(row["protein_residue_type"]),
        )
        try:
            ligand_serial = int(row["ligand_atom_serial"])
        except Exception:
            continue
        pair_key = sample_key + residue_key + (ligand_serial,)
        grouped[pair_key][type_to_idx[row["interaction_type"]]] = 1
        serials[sample_key].add(ligand_serial)
    return grouped, serials


def group_sites(frame, keys):
    grouped = defaultdict(list)
    for row in frame.to_dict("records"):
        sample_key = tuple(row[key] for key in keys)
        grouped[sample_key].append(
            (
                clean_str(row["protein_chain"]),
                clean_resnr(row["protein_residue_number"]),
                clean_str(row["protein_residue_type"]),
            )
        )
    return grouped


def select_manifest(frame, args):
    limits = {
        "train": args.max_train_complexes,
        "internal_val": args.max_internal_val_complexes,
        "external_val": args.max_external_val_complexes,
        "external_test": args.max_external_test_complexes,
    }
    selected = []
    for split, limit in limits.items():
        group = frame[frame["supervision_split"] == split].copy()
        if limit and len(group) > limit:
            group = group.sample(n=limit, random_state=args.seed + SPLIT_TO_ID[split])
        selected.append(group)
    return pd.concat(selected, ignore_index=True).sort_values(["supervision_split", "sample_id"])


def residue_coordinates(residue_atoms, residue_key):
    atoms = residue_atoms.get(residue_key)
    if not atoms and residue_key[2]:
        atoms = residue_atoms.get((residue_key[0], residue_key[1], ""))
    if not atoms:
        return None, None
    coords = np.stack([atom["xyz"] for atom in atoms], axis=0)
    return coords, coords.mean(axis=0)


def distances_to_residue(ligand_xyz, residue_atom_xyz, residue_centroid):
    deltas = residue_atom_xyz - ligand_xyz[None, :]
    min_atom_distance = float(np.sqrt(np.square(deltas).sum(axis=1).min()))
    centroid_distance = float(np.linalg.norm(residue_centroid - ligand_xyz))
    return min_atom_distance, centroid_distance


def ring_descriptor(ring):
    normal = np.asarray(ring.normal, dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        return None
    return {
        "center": np.asarray(ring.center, dtype=np.float32),
        "normal": normal / norm,
        "serials": {int(serial) for serial in ring.atoms_orig_idx},
    }


def load_ring_context(row, ligand_serials, pdb_atoms):
    try:
        from plip.basic import config
        from plip.structure.preparation import PDBComplex

        config.SILENT = True
        complex_obj = PDBComplex()
        complex_obj.load_pdb(row["plip_pdb"])
        for ligand in complex_obj.ligands:
            complex_obj.characterize_complex(ligand)

        selected_serials = {int(serial) for serial in ligand_serials}
        candidates = []
        for interaction_set in complex_obj.interaction_sets.values():
            ligand_rings = [
                descriptor
                for ring in interaction_set.ligand.rings
                if (descriptor := ring_descriptor(ring)) is not None
            ]
            ring_serials = set().union(*(ring["serials"] for ring in ligand_rings)) if ligand_rings else set()
            overlap = len(selected_serials & ring_serials)
            candidates.append((overlap, interaction_set, ligand_rings))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates or candidates[0][0] == 0:
            return {}, {}, True

        _, interaction_set, ligand_rings = candidates[0]
        protein_by_residue = defaultdict(list)
        for ring in interaction_set.bindingsite.rings:
            descriptor = ring_descriptor(ring)
            if descriptor is None:
                continue
            atoms = [pdb_atoms.get(serial) for serial in descriptor["serials"]]
            atoms = [atom for atom in atoms if atom is not None]
            if not atoms:
                continue
            residue_key = (atoms[0]["chain"], atoms[0]["resnr"], atoms[0]["resname"])
            protein_by_residue[residue_key].append(descriptor)

        ligand_by_serial = defaultdict(list)
        for descriptor in ligand_rings:
            for serial in descriptor["serials"] & selected_serials:
                ligand_by_serial[serial].append(descriptor)
        return protein_by_residue, ligand_by_serial, True
    except Exception:
        return {}, {}, False


def ring_pair_geometry(residue_key, ligand_serial, protein_by_residue, ligand_by_serial):
    best = None
    for protein_ring in protein_by_residue.get(residue_key, []):
        for ligand_ring in ligand_by_serial.get(int(ligand_serial), []):
            delta = protein_ring["center"] - ligand_ring["center"]
            center_distance = float(np.linalg.norm(delta))
            cosine = float(np.clip(abs(np.dot(protein_ring["normal"], ligand_ring["normal"])), 0.0, 1.0))
            normal_angle = float(np.degrees(np.arccos(cosine)))
            protein_offset = float(
                np.linalg.norm(delta - np.dot(delta, protein_ring["normal"]) * protein_ring["normal"])
            )
            ligand_offset = float(
                np.linalg.norm(delta - np.dot(delta, ligand_ring["normal"]) * ligand_ring["normal"])
            )
            candidate = (center_distance, normal_angle, min(protein_offset, ligand_offset))
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return np.zeros(len(RING_GEOMETRY_NAMES), dtype=np.float32)
    return np.asarray(
        [
            1.0,
            min(best[0] / 10.0, 2.0),
            min(best[1] / 90.0, 1.0),
            min(best[2] / 5.0, 2.0),
        ],
        dtype=np.float32,
    )


def process_sample(row, interaction_lookup, positive_serials_by_sample, site_lookup, args):
    source = row["source"]
    if source == "biolip2_nr":
        sample_key = (row["complex_id"],)
    else:
        sample_key = (row["system_id"], row["ligand_id"])
    positive_serials = positive_serials_by_sample.get(sample_key, set())
    site_residues = sorted(set(site_lookup.get(sample_key, [])))
    if not site_residues:
        return None

    site, ligand_serials = choose_binding_site(row, positive_serials)
    if not ligand_serials and positive_serials:
        ligand_serials = sorted(positive_serials)
    if not ligand_serials:
        return None

    try:
        pdb_atoms, residue_atoms = parse_pdb_atoms(row["plip_pdb"])
    except Exception:
        return None
    ligand_atoms = []
    for serial in ligand_serials:
        atom = pdb_atoms.get(int(serial))
        if atom is not None and atom["element"] != "H":
            ligand_atoms.append(atom)
    if not ligand_atoms:
        return None

    positive_pairs = {}
    for residue_key in site_residues:
        for ligand_atom in ligand_atoms:
            lookup_key = sample_key + residue_key + (ligand_atom["serial"],)
            labels = interaction_lookup.get(lookup_key)
            if labels is not None and labels.any():
                positive_pairs[(residue_key, ligand_atom["serial"])] = labels

    all_negative_pairs = []
    for residue_key in site_residues:
        atom_xyz, centroid = residue_coordinates(residue_atoms, residue_key)
        if atom_xyz is None:
            continue
        for ligand_atom in ligand_atoms:
            if (residue_key, ligand_atom["serial"]) in positive_pairs:
                continue
            all_negative_pairs.append((residue_key, ligand_atom["serial"]))

    rng = random.Random(stable_seed(row["sample_id"], args.seed))
    n_pos = len(positive_pairs)
    if n_pos:
        n_neg = min(len(all_negative_pairs), max(args.negative_ratio * n_pos, 1), args.max_negatives_per_complex)
    else:
        n_neg = min(len(all_negative_pairs), args.negatives_without_positive)
    if args.negative_mode == "hard" and n_neg < len(all_negative_pairs):
        scored = []
        for residue_key, serial in all_negative_pairs:
            atom_xyz, centroid = residue_coordinates(residue_atoms, residue_key)
            ligand_atom = pdb_atoms.get(int(serial))
            if atom_xyz is None or ligand_atom is None:
                continue
            min_distance, _ = distances_to_residue(ligand_atom["xyz"], atom_xyz, centroid)
            scored.append((min_distance, rng.random(), residue_key, serial))
        scored.sort(key=lambda item: (item[0], item[1]))
        negative_pairs = [(residue_key, serial) for _, _, residue_key, serial in scored[:n_neg]]
    else:
        negative_pairs = rng.sample(all_negative_pairs, n_neg) if n_neg < len(all_negative_pairs) else all_negative_pairs

    pair_items = [(residue_key, serial, labels) for (residue_key, serial), labels in positive_pairs.items()]
    zero = np.zeros(len(INTERACTION_TYPES), dtype=np.uint8)
    pair_items.extend((residue_key, serial, zero) for residue_key, serial in negative_pairs)
    if not pair_items:
        return None

    protein_rings = {}
    ligand_rings = {}
    ring_geometry_ok = True
    if args.include_ring_geometry:
        protein_rings, ligand_rings, ring_geometry_ok = load_ring_context(
            row,
            ligand_serials,
            pdb_atoms,
        )

    rows = []
    for residue_key, serial, labels in pair_items:
        atom_xyz, centroid = residue_coordinates(residue_atoms, residue_key)
        ligand_atom = pdb_atoms.get(int(serial))
        if atom_xyz is None or ligand_atom is None:
            continue
        min_distance, centroid_distance = distances_to_residue(ligand_atom["xyz"], atom_xyz, centroid)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["supervision_split"],
                "source": source,
                "residue_type": residue_key[2],
                "ligand_element": ligand_atom["element"],
                "min_atom_distance": min_distance,
                "centroid_distance": centroid_distance,
                "ring_geometry": ring_pair_geometry(
                    residue_key,
                    serial,
                    protein_rings,
                    ligand_rings,
                ),
                "ring_geometry_ok": ring_geometry_ok,
                "labels": labels.copy(),
                "loss_mask": np.asarray(
                    [1, 1, int(row["loss_mask_water_bridge"]), 1, 1, 1],
                    dtype=np.uint8,
                ),
            }
        )
    return rows


def configure_worker_state(biolip_lookup, biolip_serials, biolip_sites, plinder_lookup, plinder_serials, plinder_sites, args):
    global _WORKER_STATE
    _WORKER_STATE = {
        "biolip_lookup": biolip_lookup,
        "biolip_serials": biolip_serials,
        "biolip_sites": biolip_sites,
        "plinder_lookup": plinder_lookup,
        "plinder_serials": plinder_serials,
        "plinder_sites": plinder_sites,
        "args": args,
    }


def process_manifest_row(row):
    if row["source"] == "biolip2_nr":
        interaction_lookup = _WORKER_STATE["biolip_lookup"]
        positive_serials = _WORKER_STATE["biolip_serials"]
        site_lookup = _WORKER_STATE["biolip_sites"]
    else:
        interaction_lookup = _WORKER_STATE["plinder_lookup"]
        positive_serials = _WORKER_STATE["plinder_serials"]
        site_lookup = _WORKER_STATE["plinder_sites"]
    return process_sample(
        row,
        interaction_lookup,
        positive_serials,
        site_lookup,
        _WORKER_STATE["args"],
    )


def iter_sample_pairs(records, args):
    if args.num_workers <= 1:
        yield from map(process_manifest_row, records)
        return
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("--num-workers > 1 requires the multiprocessing fork start method")
    context = mp.get_context("fork")
    with context.Pool(processes=args.num_workers) as pool:
        yield from pool.imap(process_manifest_row, records, chunksize=args.worker_chunksize)


def encode_rows(rows):
    aa_to_id = {aa: idx + 1 for idx, aa in enumerate(AA3)}
    element_to_id = {element: idx + 1 for idx, element in enumerate(ELEMENTS)}
    n = len(rows)
    residue_type = np.zeros(n, dtype=np.uint8)
    ligand_element = np.zeros(n, dtype=np.uint8)
    source = np.zeros(n, dtype=np.uint8)
    split = np.zeros(n, dtype=np.uint8)
    distances = np.zeros((n, 2), dtype=np.float32)
    ring_geometry = np.zeros((n, len(RING_GEOMETRY_NAMES)), dtype=np.float32)
    labels = np.zeros((n, len(INTERACTION_TYPES)), dtype=np.uint8)
    loss_mask = np.zeros((n, len(INTERACTION_TYPES)), dtype=np.uint8)
    for i, row in enumerate(rows):
        residue_type[i] = aa_to_id.get(row["residue_type"], 0)
        ligand_element[i] = element_to_id.get(str(row["ligand_element"]).upper(), 0)
        source[i] = SOURCE_TO_ID.get(row["source"], 255)
        split[i] = SPLIT_TO_ID[row["split"]]
        distances[i, 0] = row["min_atom_distance"]
        distances[i, 1] = row["centroid_distance"]
        ring_geometry[i, :] = row["ring_geometry"]
        labels[i, :] = row["labels"]
        loss_mask[i, :] = row["loss_mask"]
    return residue_type, ligand_element, source, split, distances, ring_geometry, labels, loss_mask


def write_table(stats, path):
    lines = [
        "# Interaction Pair Feature Cache",
        "",
        f"- output: `{stats['output_npz']}`",
        f"- selected complexes: {stats['selected_complexes']}",
        f"- processed complexes: {stats['processed_complexes']}",
        f"- skipped complexes: {stats['skipped_complexes']}",
        f"- sampled residue-atom pairs: {stats['pairs']}",
        f"- ring geometry enabled: {str(stats['settings'].get('include_ring_geometry', False)).lower()}",
        f"- aromatic ring-pair features: {stats.get('ring_geometry_pairs', 0)}",
        f"- ring geometry parse failures: {stats.get('ring_geometry_failed_complexes', 0)}",
        "",
    ]
    skipped_samples = stats.get("skipped_samples", [])
    if skipped_samples:
        lines.extend(
            [
                "## Skipped Samples",
                "",
                "| sample_id | split | source |",
                "| --- | --- | --- |",
            ]
        )
        for row in skipped_samples:
            lines.append(f"| {row['sample_id']} | {row['split']} | {row['source']} |")
        lines.append("")
    lines.extend(
        [
            "| split | complexes | pairs | positive_pairs_any_type | water_loss_available_pairs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in stats["split_rows"].items():
        lines.append(
            f"| {split} | {row['complexes']} | {row['pairs']} | {row['positive_pairs_any_type']} | "
            f"{row['water_loss_available_pairs']} |"
        )
    lines.extend(["", "| interaction_type | positives | loss_available_pairs |", "| --- | ---: | ---: |"])
    for interaction_type in INTERACTION_TYPES:
        row = stats["interaction_type_rows"][interaction_type]
        lines.append(f"| {interaction_type} | {row['positives']} | {row['loss_available_pairs']} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    selected = select_manifest(manifest, args)
    biolip_i, biolip_s, plinder_i, plinder_s = load_labels()
    biolip_lookup, biolip_serials = group_interactions(biolip_i, ["complex_id"])
    plinder_lookup, plinder_serials = group_interactions(plinder_i, ["system_id", "ligand_id"])
    biolip_sites = group_sites(biolip_s, ["complex_id"])
    plinder_sites = group_sites(plinder_s, ["system_id", "ligand_id"])
    configure_worker_state(
        biolip_lookup,
        biolip_serials,
        biolip_sites,
        plinder_lookup,
        plinder_serials,
        plinder_sites,
        args,
    )

    rows = []
    processed = 0
    skipped = 0
    skipped_rows = []
    sample_rows = []
    selected_records = selected.to_dict("records")
    pair_iterator = iter_sample_pairs(selected_records, args)
    for done, (row, sample_pairs) in enumerate(zip(selected_records, pair_iterator), start=1):
        if sample_pairs:
            processed += 1
            rows.extend(sample_pairs)
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["supervision_split"],
                    "source": row["source"],
                    "pairs": len(sample_pairs),
                    "positives_any_type": sum(1 for item in sample_pairs if item["labels"].any()),
                    "ring_geometry_ok": int(sample_pairs[0]["ring_geometry_ok"]),
                    "ring_geometry_pairs": sum(item["ring_geometry"][0] > 0 for item in sample_pairs),
                }
            )
        else:
            skipped += 1
            skipped_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["supervision_split"],
                    "source": row["source"],
                }
            )
        if done % args.log_every == 0 or done == len(selected):
            print(f"{done}/{len(selected)} selected complexes processed={processed} skipped={skipped} pairs={len(rows)}", flush=True)

    residue_type, ligand_element, source, split, distances, ring_geometry, labels, loss_mask = encode_rows(rows)
    output_npz = out_dir / "interaction_pair_features.npz"
    arrays = {
        "residue_type": residue_type,
        "ligand_element": ligand_element,
        "source": source,
        "split": split,
        "distances": distances,
        "labels": labels,
        "loss_mask": loss_mask,
        "interaction_types": np.asarray(INTERACTION_TYPES),
        "aa3": np.asarray(["UNK"] + AA3),
        "elements": np.asarray(["UNK"] + ELEMENTS),
    }
    if args.include_ring_geometry:
        arrays["ring_geometry"] = ring_geometry
        arrays["ring_geometry_names"] = np.asarray(RING_GEOMETRY_NAMES)
    np.savez_compressed(output_npz, **arrays)
    sample_csv = out_dir / "sample_pair_counts.csv.gz"
    with gzip.open(sample_csv, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "source",
                "pairs",
                "positives_any_type",
                "ring_geometry_ok",
                "ring_geometry_pairs",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    stats = {
        "output_npz": str(output_npz),
        "sample_pair_counts": str(sample_csv),
        "selected_complexes": int(len(selected)),
        "processed_complexes": processed,
        "skipped_complexes": skipped,
        "skipped_samples": skipped_rows,
        "pairs": int(labels.shape[0]),
        "positive_pairs_any_type": int(labels.any(axis=1).sum()),
        "ring_geometry_pairs": int((ring_geometry[:, 0] > 0).sum()) if args.include_ring_geometry else 0,
        "ring_geometry_failed_complexes": (
            sum(1 for item in sample_rows if not item["ring_geometry_ok"])
            if args.include_ring_geometry
            else 0
        ),
        "split_rows": {},
        "interaction_type_rows": {},
        "settings": {
            "negative_ratio": args.negative_ratio,
            "negative_mode": args.negative_mode,
            "negatives_without_positive": args.negatives_without_positive,
            "max_negatives_per_complex": args.max_negatives_per_complex,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "worker_chunksize": args.worker_chunksize,
            "include_ring_geometry": args.include_ring_geometry,
        },
    }
    for split_name, split_id in SPLIT_TO_ID.items():
        mask = split == split_id
        sample_count = sum(1 for item in sample_rows if item["split"] == split_name)
        stats["split_rows"][split_name] = {
            "complexes": int(sample_count),
            "pairs": int(mask.sum()),
            "positive_pairs_any_type": int(labels[mask].any(axis=1).sum()) if mask.any() else 0,
            "water_loss_available_pairs": int(loss_mask[mask, INTERACTION_TYPES.index("water_bridge")].sum()) if mask.any() else 0,
        }
    for idx, interaction_type in enumerate(INTERACTION_TYPES):
        stats["interaction_type_rows"][interaction_type] = {
            "positives": int(labels[:, idx].sum()),
            "loss_available_pairs": int(loss_mask[:, idx].sum()),
        }
    stats_path = out_dir / "interaction_pair_feature_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_table(stats, args.table)
    print(json.dumps({"stats": str(stats_path), "table": args.table, "output_npz": str(output_npz)}))


if __name__ == "__main__":
    main()
