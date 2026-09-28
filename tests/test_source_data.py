from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "derived"


def test_required_source_tables_exist() -> None:
    required = {
        "fig2_paired_graph_ap_source.csv",
        "fig3_temporal_scatter.csv",
        "fig4_auprc.csv",
        "figs1_components.csv",
        "figs2_pose.csv",
        "figs3_chemical_rows.csv",
    }
    assert required.issubset({path.name for path in SOURCE.glob("*.csv")})


def test_graph_ap_values_are_probabilities() -> None:
    path = SOURCE / "fig2_paired_graph_ap_source.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    numeric_columns = [
        name for name in rows[0] if name and ("graph_ap" in name or name == "delta")
    ]
    assert numeric_columns
    for row in rows:
        for column in numeric_columns:
            value = row[column]
            if not value:
                continue
            number = float(value)
            assert -1.0 <= number <= 1.0
