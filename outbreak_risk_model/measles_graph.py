"""Graph-state layer for measles modelling.

This module converts geography boundaries and measles input tables into graph
tables:

- nodes store local measles data such as population, protection, and cases;
- edges store either first-/second-order touching relationships or ODWP
  commuting flows.

It does not run the epidemic simulation yet. The purpose is to make the data
structure inspectable before adding graph-based transmission dynamics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DEFAULT_INPUT_DIR
from .geography_graph import _iter_lon_lat, haversine_km, load_geojson_nodes
from .odwp_mobility import default_odwp_csv, load_odwp_flow_edges


DEFAULT_BOUNDARY_DIR = Path(__file__).resolve().parent / "boundary_data"
DEFAULT_GRAPH_OUTPUT_DIR = Path(__file__).resolve().parent / "graph_outputs"
DEFAULT_CONTACT_MATRIX = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "measles_age"
    / "inputs"
    / "age_contact_matrix_reconnect_figure_normalized.csv"
)


@dataclass(frozen=True)
class MeaslesGraphTables:
    """Tables needed by the future graph-based measles simulator."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    contact_matrix: pd.DataFrame


def build_region_measles_graph(
    *,
    region_geojson: Path | str = DEFAULT_BOUNDARY_DIR / "uk_regions_2025.geojson",
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    contact_matrix_path: Path | str = DEFAULT_CONTACT_MATRIX,
    second_order_weight: float = 0.5,
    edge_source: str = "touching",
    odwp_csv: Path | str | None = None,
) -> MeaslesGraphTables:
    """Build a measles-ready graph for UK region boundaries.

    Current measles population/protection/case inputs only cover English
    regions. UK nations outside England are retained as boundary nodes with
    ``measles_data_available=False``.
    """

    nodes = load_geojson_nodes(region_geojson)
    if edge_source == "odwp":
        nodes = _england_nodes(nodes)
    nodes = attach_region_measles_data(nodes, input_dir=input_dir)
    if edge_source == "touching":
        edges = build_touching_order_edges(
            region_geojson,
            nodes=nodes,
            second_order_weight=second_order_weight,
        )
    elif edge_source == "odwp":
        edges = build_odwp_edges_for_nodes(
            nodes,
            odwp_csv=odwp_csv or default_odwp_csv("region"),
        )
    else:
        raise ValueError("edge_source must be either 'touching' or 'odwp'")
    contact_matrix = pd.read_csv(contact_matrix_path, index_col=0)
    nodes["contact_matrix_source"] = str(contact_matrix_path)
    nodes["contact_matrix_multiplier"] = 1.0
    return MeaslesGraphTables(nodes=nodes, edges=edges, contact_matrix=contact_matrix)


def build_local_authority_touch_graph(
    *,
    geojson_path: Path | str = DEFAULT_BOUNDARY_DIR / "uk_local_authority_upper_tier_2025.geojson",
    second_order_weight: float = 0.5,
    edge_source: str = "touching",
    odwp_csv: Path | str | None = None,
) -> MeaslesGraphTables:
    """Build upper-tier local authority topology with placeholder measles fields."""

    nodes = load_geojson_nodes(geojson_path)
    if edge_source == "odwp":
        nodes = _england_nodes(nodes)
    nodes["measles_data_available"] = False
    nodes["contact_matrix_source"] = str(DEFAULT_CONTACT_MATRIX)
    nodes["contact_matrix_multiplier"] = 1.0
    if edge_source == "touching":
        edges = build_touching_order_edges(
            geojson_path,
            nodes=nodes,
            second_order_weight=second_order_weight,
        )
    elif edge_source == "odwp":
        edges = build_odwp_edges_for_nodes(
            nodes,
            odwp_csv=odwp_csv or default_odwp_csv("utla"),
        )
    else:
        raise ValueError("edge_source must be either 'touching' or 'odwp'")
    contact_matrix = pd.read_csv(DEFAULT_CONTACT_MATRIX, index_col=0)
    return MeaslesGraphTables(nodes=nodes, edges=edges, contact_matrix=contact_matrix)


def build_odwp_edges_for_nodes(
    nodes: pd.DataFrame,
    *,
    odwp_csv: Path | str,
) -> pd.DataFrame:
    """Build England ODWP commuting edges aligned to graph nodes."""

    flows = load_odwp_flow_edges(odwp_csv, england_only=True, include_self_loops=True)
    node_codes = set(nodes["code"].astype(str))
    flows = flows.loc[
        flows["source_code"].astype(str).isin(node_codes)
        & flows["target_code"].astype(str).isin(node_codes)
    ].copy()
    source_total = flows.groupby("source_code")["commuter_count"].transform("sum")
    target_total = flows.groupby("target_code")["commuter_count"].transform("sum")
    flows["raw_mobility_weight"] = flows["commuter_count"]
    flows["mobility_weight"] = flows["commuter_count"] / source_total
    flows["destination_residence_share"] = flows["commuter_count"] / target_total
    node_lookup = nodes.set_index("code")

    distances = []
    for row in flows.itertuples(index=False):
        source = node_lookup.loc[row.source_code]
        target = node_lookup.loc[row.target_code]
        distances.append(
            haversine_km(
                source["latitude"],
                source["longitude"],
                target["latitude"],
                target["longitude"],
            )
        )

    flows["edge_source"] = "odwp_commuting"
    flows["touching_order"] = pd.NA
    flows["distance_km"] = distances
    return flows[
        [
            "source_code",
            "source_name",
            "target_code",
            "target_name",
            "edge_source",
            "touching_order",
            "distance_km",
            "commuter_count",
            "is_self_loop",
            "raw_mobility_weight",
            "mobility_weight",
            "destination_residence_share",
        ]
    ].reset_index(drop=True)


def attach_region_measles_data(
    nodes: pd.DataFrame,
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
) -> pd.DataFrame:
    """Attach age-structured measles inputs to geography nodes."""

    input_path = Path(input_dir)
    profile = pd.read_csv(input_path / "region_age_population_protection.csv")
    cases = pd.read_csv(input_path / "synthetic_region_age_weekly_cases.csv")
    cases["date"] = pd.to_datetime(cases["date"])
    latest_cases = cases[cases["date"].eq(cases["date"].max())]

    age_groups = list(profile["age_group"].drop_duplicates())
    out = nodes.copy()
    out["measles_data_available"] = out["name"].isin(profile["region"].unique())

    population = _wide_by_age(profile, "region", "age_group", "population", age_groups)
    protected = _wide_by_age(profile, "region", "age_group", "protected_fraction", age_groups)
    latest = _wide_by_age(
        latest_cases,
        "region",
        "age_group",
        "synthetic_cases",
        age_groups,
        fill_value=0.0,
    )
    risk = (
        profile.groupby("region")["region_risk_multiplier"]
        .first()
        .rename("region_risk_multiplier")
    )
    mmr_columns = [
        column
        for column in ("mmr1_24m", "mmr1_5y", "mmr2_5y")
        if column in profile.columns
    ]
    mmr = profile.groupby("region")[mmr_columns].first() if mmr_columns else pd.DataFrame()

    out = out.merge(population, left_on="name", right_index=True, how="left")
    out = out.merge(protected, left_on="name", right_index=True, how="left")
    out = out.merge(latest, left_on="name", right_index=True, how="left")
    out = out.merge(risk, left_on="name", right_index=True, how="left")
    if not mmr.empty:
        out = out.merge(mmr, left_on="name", right_index=True, how="left")

    population_cols = [f"population_{age}" for age in age_groups]
    case_cols = [f"current_cases_{age}" for age in age_groups]
    out["total_population"] = out[population_cols].sum(axis=1, min_count=1)
    out["total_current_cases"] = out[case_cols].sum(axis=1, min_count=1)
    return out


def build_touching_order_edges(
    geojson_path: Path | str,
    *,
    nodes: pd.DataFrame,
    second_order_weight: float = 0.5,
    coordinate_precision: int = 5,
    min_shared_points: int = 2,
) -> pd.DataFrame:
    """Build directed first- and second-order touching edges.

    First-order edges connect areas that share boundary coordinates.
    Second-order edges connect areas that are two first-order boundary steps
    away. Edge weights are normalized separately per source across all retained
    first- and second-order neighbours.
    """

    boundary_sets = boundary_coordinate_sets(
        geojson_path,
        coordinate_precision=coordinate_precision,
    )
    first_order = first_order_touch_pairs(
        boundary_sets,
        min_shared_points=min_shared_points,
    )
    neighbours = {name: set() for name in nodes["name"]}
    for source, target, _shared_points in first_order:
        neighbours[source].add(target)
        neighbours[target].add(source)

    rows = []
    node_lookup = nodes.set_index("name")
    for source in nodes["name"]:
        first = sorted(neighbours[source])
        second = sorted(
            {
                candidate
                for neighbour in first
                for candidate in neighbours.get(neighbour, set())
                if candidate != source and candidate not in first
            }
        )
        for target in first:
            rows.append(_edge_row(node_lookup, source, target, 1, 1.0))
        for target in second:
            rows.append(_edge_row(node_lookup, source, target, 2, second_order_weight))

    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    normalizer = edges.groupby("source_name")["raw_mobility_weight"].transform("sum")
    edges["mobility_weight"] = edges["raw_mobility_weight"] / normalizer
    return edges.sort_values(["source_name", "touching_order", "distance_km"]).reset_index(drop=True)


def boundary_coordinate_sets(
    geojson_path: Path | str,
    *,
    coordinate_precision: int = 5,
) -> dict[str, set[tuple[float, float]]]:
    """Return rounded boundary coordinate sets keyed by geography name."""

    payload = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    out = {}
    for idx, feature in enumerate(payload.get("features", []), start=1):
        properties = feature.get("properties") or {}
        name = _pick_property(properties, suffixes=("nm", "name"), fallback=f"feature_{idx}")
        points = {
            (round(lon, coordinate_precision), round(lat, coordinate_precision))
            for lon, lat in _iter_lon_lat(feature.get("geometry"))
        }
        out[name] = points
    return out


def first_order_touch_pairs(
    boundary_sets: dict[str, set[tuple[float, float]]],
    *,
    min_shared_points: int = 2,
) -> list[tuple[str, str, int]]:
    """Return undirected pairs that share at least ``min_shared_points``."""

    names = sorted(boundary_sets)
    pairs = []
    for i, source in enumerate(names):
        for target in names[i + 1 :]:
            shared_points = len(boundary_sets[source] & boundary_sets[target])
            if shared_points >= min_shared_points:
                pairs.append((source, target, shared_points))
    return pairs


def save_measles_graph(
    graph: MeaslesGraphTables,
    *,
    output_dir: Path | str = DEFAULT_GRAPH_OUTPUT_DIR,
    prefix: str = "measles_region_graph",
) -> dict[str, Path]:
    """Save graph tables to CSV files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "nodes": out / f"{prefix}_nodes.csv",
        "edges": out / f"{prefix}_edges.csv",
        "contact_matrix": out / f"{prefix}_contact_matrix.csv",
    }
    graph.nodes.to_csv(paths["nodes"], index=False)
    graph.edges.to_csv(paths["edges"], index=False)
    graph.contact_matrix.to_csv(paths["contact_matrix"])
    return paths


def _england_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Return only England-coded geography nodes."""

    return nodes.loc[nodes["code"].astype(str).str.startswith("E")].reset_index(drop=True)


def _wide_by_age(
    data: pd.DataFrame,
    region_column: str,
    age_column: str,
    value_column: str,
    age_groups: list[str],
    *,
    fill_value: float | None = None,
) -> pd.DataFrame:
    table = data.pivot(index=region_column, columns=age_column, values=value_column)
    table = table.reindex(columns=age_groups)
    if fill_value is not None:
        table = table.fillna(fill_value)
    table.columns = [f"{_value_prefix(value_column)}_{age}" for age in table.columns]
    return table


def _value_prefix(value_column: str) -> str:
    if value_column == "synthetic_cases":
        return "current_cases"
    return value_column


def _edge_row(
    node_lookup: pd.DataFrame,
    source: str,
    target: str,
    touching_order: int,
    raw_weight: float,
) -> dict:
    source_row = node_lookup.loc[source]
    target_row = node_lookup.loc[target]
    distance_km = haversine_km(
        source_row["latitude"],
        source_row["longitude"],
        target_row["latitude"],
        target_row["longitude"],
    )
    return {
        "source_code": source_row["code"],
        "source_name": source,
        "target_code": target_row["code"],
        "target_name": target,
        "touching_order": touching_order,
        "distance_km": distance_km,
        "raw_mobility_weight": raw_weight,
    }


def _pick_property(properties: dict, *, suffixes: tuple[str, ...], fallback: str) -> str:
    lowered = {str(key).lower(): key for key in properties}
    for suffix in suffixes:
        for lower_key, original_key in lowered.items():
            if lower_key.endswith(suffix):
                value = properties[original_key]
                if value not in (None, ""):
                    return str(value)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph-state tables for measles modelling.")
    parser.add_argument("--level", choices=["region", "utla"], default="region")
    parser.add_argument("--region-geojson", type=Path, default=DEFAULT_BOUNDARY_DIR / "uk_regions_2025.geojson")
    parser.add_argument("--utla-geojson", type=Path, default=DEFAULT_BOUNDARY_DIR / "uk_local_authority_upper_tier_2025.geojson")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_OUTPUT_DIR)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--edge-source", choices=["touching", "odwp"], default="touching")
    parser.add_argument("--odwp-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.level == "region":
        graph = build_region_measles_graph(
            region_geojson=args.region_geojson,
            input_dir=args.input_dir,
            edge_source=args.edge_source,
            odwp_csv=args.odwp_csv,
        )
        prefix = args.prefix or "measles_region_graph"
    else:
        graph = build_local_authority_touch_graph(
            geojson_path=args.utla_geojson,
            edge_source=args.edge_source,
            odwp_csv=args.odwp_csv,
        )
        prefix = args.prefix or "measles_utla_graph"

    paths = save_measles_graph(graph, output_dir=args.output_dir, prefix=prefix)
    print("Saved measles graph tables:")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")
    print(f"measles-ready nodes: {int(graph.nodes['measles_data_available'].sum())}")


if __name__ == "__main__":
    main()
