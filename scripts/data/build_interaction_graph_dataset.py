#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from scripts.data import build_interaction_pair_features as pair_features


ROOT = Path(__file__).resolve().parents[2]
PROTEIN_ELEMENT_OFFSET = 32
LIGAND_ELEMENT_OFFSET = 64
MAX_NODE_TYPE = 96
_WORKER_STATE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build raw-coordinate protein-ligand graphs with hard-negative interaction pairs."
    )
    parser.add_argument(
        "--split-manifest",
        default=str(
            ROOT
            / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT / "data/processed/structure_supervision/interaction_graphs_hardneg_smoke"
        ),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/table_interaction_graphs_hardneg_smoke.md"),
    )
    parser.add_argument("--max-train-complexes", type=int, default=6000)
    parser.add_argument("--max-internal-val-complexes", type=int, default=1000)
    parser.add_argument("--max-external-val-complexes", type=int, default=400)
    parser.add_argument("--max-external-test-complexes", type=int, default=400)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--negative-mode", choices=["random", "hard"], default="hard")
    parser.add_argument("--negatives-without-positive", type=int, default=32)
    parser.add_argument("--max-negatives-per-complex", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--worker-chunksize", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def element_id(element):
    value = str(element).upper()
    try:
        return pair_features.ELEMENTS.index(value) + 1
    except ValueError:
        return 0


def residue_id(residue_type):
    try:
        return pair_features.AA3.index(str(residue_type).upper()) + 1
    except ValueError:
        return 0


def get_residue_atoms(residue_atoms, residue_key):
    atoms = residue_atoms.get(residue_key)
    if not atoms and residue_key[2]:
        atoms = residue_atoms.get((residue_key[0], residue_key[1], ""))
    if not atoms:
        return []
    return [atom for atom in atoms if atom["element"] != "H"]


def select_pair_items(
    row,
    interaction_lookup,
    positive_serials_by_sample,
    site_lookup,
    pdb_atoms,
    residue_atoms,
    ligand_atoms,
    sample_key,
    args,
):
    site_residues = sorted(set(site_lookup.get(sample_key, [])))
    positive_pairs = {}
    for residue_key in site_residues:
        for ligand_atom in ligand_atoms:
            lookup_key = sample_key + residue_key + (ligand_atom["serial"],)
            labels = interaction_lookup.get(lookup_key)
            if labels is not None and labels.any():
                positive_pairs[(residue_key, ligand_atom["serial"])] = labels

    all_negative_pairs = []
    for residue_key in site_residues:
        atom_xyz, centroid = pair_features.residue_coordinates(residue_atoms, residue_key)
        if atom_xyz is None:
            continue
        for ligand_atom in ligand_atoms:
            key = (residue_key, ligand_atom["serial"])
            if key not in positive_pairs:
                all_negative_pairs.append(key)

    rng = random.Random(pair_features.stable_seed(row["sample_id"], args.seed))
    n_pos = len(positive_pairs)
    if n_pos:
        n_neg = min(
            len(all_negative_pairs),
            max(args.negative_ratio * n_pos, 1),
            args.max_negatives_per_complex,
        )
    else:
        n_neg = min(len(all_negative_pairs), args.negatives_without_positive)

    if args.negative_mode == "hard" and n_neg < len(all_negative_pairs):
        scored = []
        for residue_key, serial in all_negative_pairs:
            atom_xyz, centroid = pair_features.residue_coordinates(residue_atoms, residue_key)
            ligand_atom = pdb_atoms.get(int(serial))
            if atom_xyz is None or ligand_atom is None:
                continue
            min_distance, _ = pair_features.distances_to_residue(
                ligand_atom["xyz"],
                atom_xyz,
                centroid,
            )
            scored.append((min_distance, rng.random(), residue_key, serial))
        scored.sort(key=lambda item: (item[0], item[1]))
        negative_pairs = [
            (residue_key, serial)
            for _, _, residue_key, serial in scored[:n_neg]
        ]
    else:
        negative_pairs = (
            rng.sample(all_negative_pairs, n_neg)
            if n_neg < len(all_negative_pairs)
            else all_negative_pairs
        )

    items = [
        (residue_key, serial, labels)
        for (residue_key, serial), labels in positive_pairs.items()
    ]
    zero = np.zeros(len(pair_features.INTERACTION_TYPES), dtype=np.uint8)
    items.extend(
        (residue_key, serial, zero)
        for residue_key, serial in negative_pairs
    )
    return site_residues, items


def build_graph(
    row,
    interaction_lookup,
    positive_serials_by_sample,
    site_lookup,
    args,
):
    source = row["source"]
    if source == "biolip2_nr":
        sample_key = (row["complex_id"],)
    else:
        sample_key = (row["system_id"], row["ligand_id"])

    positive_serials = positive_serials_by_sample.get(sample_key, set())
    site_residues = sorted(set(site_lookup.get(sample_key, [])))
    if not site_residues:
        return None

    _, ligand_serials = pair_features.choose_binding_site(row, positive_serials)
    if not ligand_serials and positive_serials:
        ligand_serials = sorted(positive_serials)
    if not ligand_serials:
        return None

    try:
        pdb_atoms, residue_atoms = pair_features.parse_pdb_atoms(row["plip_pdb"])
    except Exception:
        return None

    ligand_atoms = []
    for serial in ligand_serials:
        atom = pdb_atoms.get(int(serial))
        if atom is not None and atom["element"] != "H":
            ligand_atoms.append(atom)
    if not ligand_atoms:
        return None

    site_residues, pair_items = select_pair_items(
        row,
        interaction_lookup,
        positive_serials_by_sample,
        site_lookup,
        pdb_atoms,
        residue_atoms,
        ligand_atoms,
        sample_key,
        args,
    )
    if not pair_items:
        return None

    node_type = []
    positions = []
    residue_node = {}
    ligand_node = {}
    residue_records = []
    occupied_positions = set()

    for residue_key in site_residues:
        atoms = get_residue_atoms(residue_atoms, residue_key)
        if not atoms:
            continue
        coords = np.stack([atom["xyz"] for atom in atoms], axis=0)
        centroid = coords.mean(axis=0).astype(np.float32)
        position_key = tuple(centroid.tolist())
        if position_key in occupied_positions:
            return None
        residue_node[residue_key] = len(node_type)
        node_type.append(1 + residue_id(residue_key[2]))
        positions.append(centroid)
        occupied_positions.add(position_key)
        residue_records.append((atoms, centroid))

    for atoms, centroid in residue_records:
        for atom in atoms:
            xyz = np.asarray(atom["xyz"], dtype=np.float32)
            position_key = tuple(xyz.tolist())
            if (
                np.linalg.norm(xyz - centroid) < 1e-4
                or position_key in occupied_positions
            ):
                continue
            node_type.append(PROTEIN_ELEMENT_OFFSET + element_id(atom["element"]))
            positions.append(xyz)
            occupied_positions.add(position_key)

    ligand_positions = set()
    for atom in ligand_atoms:
        xyz = np.asarray(atom["xyz"], dtype=np.float32)
        position_key = tuple(xyz.tolist())
        if (
            position_key in occupied_positions
            or position_key in ligand_positions
        ):
            return None
        ligand_node[atom["serial"]] = len(node_type)
        node_type.append(LIGAND_ELEMENT_OFFSET + element_id(atom["element"]))
        positions.append(xyz)
        ligand_positions.add(position_key)

    graph_pairs = []
    labels = []
    for residue_key, serial, pair_labels in pair_items:
        protein_index = residue_node.get(residue_key)
        atom_index = ligand_node.get(int(serial))
        if protein_index is None or atom_index is None:
            continue
        graph_pairs.append((protein_index, atom_index))
        labels.append(pair_labels.copy())
    if not graph_pairs:
        return None

    y = np.stack(labels, axis=0).astype(np.float32)
    loss_mask = np.tile(
        np.asarray(
            [1, 1, int(row["loss_mask_water_bridge"]), 1, 1, 1],
            dtype=np.float32,
        ),
        (len(graph_pairs), 1),
    )
    return {
        "z": np.asarray(node_type, dtype=np.int64),
        "pos": np.stack(positions).astype(np.float32),
        "pair_index": np.asarray(graph_pairs, dtype=np.int64).T,
        "y": y,
        "loss_mask": loss_mask,
        "split_id": pair_features.SPLIT_TO_ID[row["supervision_split"]],
        "sample_id": row["sample_id"],
        "source": source,
    }


def graph_record_to_data(record):
    return Data(
        z=torch.from_numpy(record["z"]),
        pos=torch.from_numpy(record["pos"]),
        pair_index=torch.from_numpy(record["pair_index"]).contiguous(),
        y=torch.from_numpy(record["y"]),
        loss_mask=torch.from_numpy(record["loss_mask"]),
        split_id=torch.tensor([record["split_id"]], dtype=torch.long),
        sample_id=record["sample_id"],
        source=record["source"],
        num_nodes=int(record["z"].shape[0]),
    )


def configure_worker_state(
    biolip_lookup,
    biolip_serials,
    biolip_sites,
    plinder_lookup,
    plinder_serials,
    plinder_sites,
    args,
):
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
        lookup = _WORKER_STATE["biolip_lookup"]
        serials = _WORKER_STATE["biolip_serials"]
        sites = _WORKER_STATE["biolip_sites"]
    else:
        lookup = _WORKER_STATE["plinder_lookup"]
        serials = _WORKER_STATE["plinder_serials"]
        sites = _WORKER_STATE["plinder_sites"]
    return build_graph(row, lookup, serials, sites, _WORKER_STATE["args"])


def iter_graphs(records, args):
    if args.num_workers <= 1:
        yield from map(process_manifest_row, records)
        return
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("--num-workers > 1 requires the multiprocessing fork start method")
    context = mp.get_context("fork")
    with context.Pool(processes=args.num_workers) as pool:
        yield from pool.imap(
            process_manifest_row,
            records,
            chunksize=args.worker_chunksize,
        )


def write_table(stats, path):
    lines = [
        "# Raw-Coordinate Interaction Graph Dataset",
        "",
        f"- output: `{stats['out_dir']}`",
        f"- selected complexes: {stats['selected_complexes']}",
        f"- processed graphs: {stats['processed_graphs']}",
        f"- skipped complexes: {stats['skipped_complexes']}",
        f"- graph nodes: {stats['nodes']}",
        f"- sampled interaction pairs: {stats['pairs']}",
        "- PLIP-derived ring features serialized: false",
        "",
        "| split | graphs | nodes | pairs | positive_pairs_any_type | shards |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in pair_features.SPLIT_TO_ID:
        row = stats["split_rows"][split]
        lines.append(
            f"| {split} | {row['graphs']} | {row['nodes']} | {row['pairs']} | "
            f"{row['positive_pairs_any_type']} | {row['shards']} |"
        )
    lines.extend(
        [
            "",
            "| interaction_type | positives | loss_available_pairs |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in pair_features.INTERACTION_TYPES:
        row = stats["interaction_type_rows"][name]
        lines.append(f"| {name} | {row['positives']} | {row['loss_available_pairs']} |")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    existing = list(out_dir.glob("*/graph_shard_*.pt"))
    if existing:
        raise FileExistsError(
            f"{out_dir} already contains graph shards; choose a new output directory"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    selected = pair_features.select_manifest(manifest, args)
    biolip_i, biolip_s, plinder_i, plinder_s = pair_features.load_labels()
    biolip_lookup, biolip_serials = pair_features.group_interactions(
        biolip_i,
        ["complex_id"],
    )
    plinder_lookup, plinder_serials = pair_features.group_interactions(
        plinder_i,
        ["system_id", "ligand_id"],
    )
    biolip_sites = pair_features.group_sites(biolip_s, ["complex_id"])
    plinder_sites = pair_features.group_sites(
        plinder_s,
        ["system_id", "ligand_id"],
    )
    configure_worker_state(
        biolip_lookup,
        biolip_serials,
        biolip_sites,
        plinder_lookup,
        plinder_serials,
        plinder_sites,
        args,
    )

    buffers = defaultdict(list)
    shard_counts = defaultdict(int)
    split_rows = {
        split: {
            "graphs": 0,
            "nodes": 0,
            "pairs": 0,
            "positive_pairs_any_type": 0,
            "shards": 0,
        }
        for split in pair_features.SPLIT_TO_ID
    }
    interaction_positives = np.zeros(
        len(pair_features.INTERACTION_TYPES),
        dtype=np.int64,
    )
    interaction_available = np.zeros_like(interaction_positives)
    skipped_samples = []

    def flush(split):
        if not buffers[split]:
            return
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"graph_shard_{shard_counts[split]:05d}.pt"
        torch.save(buffers[split], path)
        shard_counts[split] += 1
        split_rows[split]["shards"] += 1
        buffers[split] = []

    records = selected.to_dict("records")
    for done, (row, graph_record) in enumerate(
        zip(records, iter_graphs(records, args)),
        start=1,
    ):
        split = row["supervision_split"]
        if graph_record is None:
            skipped_samples.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "source": row["source"],
                }
            )
        else:
            graph = graph_record_to_data(graph_record)
            buffers[split].append(graph)
            split_rows[split]["graphs"] += 1
            split_rows[split]["nodes"] += int(graph.num_nodes)
            split_rows[split]["pairs"] += int(graph.y.shape[0])
            split_rows[split]["positive_pairs_any_type"] += int(
                graph.y.bool().any(dim=1).sum()
            )
            interaction_positives += graph.y.sum(dim=0).numpy().astype(np.int64)
            interaction_available += (
                graph.loss_mask.sum(dim=0).numpy().astype(np.int64)
            )
            if len(buffers[split]) >= args.shard_size:
                flush(split)
        if done % args.log_every == 0 or done == len(records):
            processed = sum(item["graphs"] for item in split_rows.values())
            pairs = sum(item["pairs"] for item in split_rows.values())
            print(
                f"{done}/{len(records)} selected complexes processed={processed} "
                f"skipped={len(skipped_samples)} pairs={pairs}",
                flush=True,
            )

    for split in pair_features.SPLIT_TO_ID:
        flush(split)

    stats = {
        "out_dir": str(out_dir),
        "selected_complexes": int(len(selected)),
        "processed_graphs": sum(item["graphs"] for item in split_rows.values()),
        "skipped_complexes": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "nodes": sum(item["nodes"] for item in split_rows.values()),
        "pairs": sum(item["pairs"] for item in split_rows.values()),
        "split_rows": split_rows,
        "interaction_type_rows": {
            name: {
                "positives": int(interaction_positives[index]),
                "loss_available_pairs": int(interaction_available[index]),
            }
            for index, name in enumerate(pair_features.INTERACTION_TYPES)
        },
        "node_type_scheme": {
            "residue_centroid": "1 + AA3 index",
            "protein_atom": f"{PROTEIN_ELEMENT_OFFSET} + element index",
            "ligand_atom": f"{LIGAND_ELEMENT_OFFSET} + element index",
            "max_node_type": MAX_NODE_TYPE,
        },
        "contains_plip_ring_features": False,
        "settings": {
            "negative_mode": args.negative_mode,
            "negative_ratio": args.negative_ratio,
            "negatives_without_positive": args.negatives_without_positive,
            "max_negatives_per_complex": args.max_negatives_per_complex,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "worker_chunksize": args.worker_chunksize,
            "shard_size": args.shard_size,
        },
    }
    stats_path = out_dir / "graph_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_table(stats, args.table)
    print(json.dumps({"stats": str(stats_path), "table": args.table}))


if __name__ == "__main__":
    main()
