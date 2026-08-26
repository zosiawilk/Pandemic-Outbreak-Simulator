"""Build and visualize a simple English-region mixing graph.

This is the first mobility layer for the outbreak-risk model. It uses region
centroids and connects each region to its nearest neighbouring regions. Later,
the same interface can be fed with origin-destination mobility flows instead
of distance-derived weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DEFAULT_INPUT_DIR, load_model_inputs


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "graph_outputs"


REGION_CENTROIDS = {
    "North East": (54.9783, -1.6178),
    "North West": (53.4808, -2.2426),
    "Yorkshire and The Humber": (53.8008, -1.5491),
    "East Midlands": (52.9548, -1.1581),
    "West Midlands": (52.4862, -1.8904),
    "East of England": (52.2053, 0.1218),
    "London": (51.5072, -0.1276),
    "South East": (51.4545, -0.9781),
    "South West": (51.4545, -2.5879),
}


@dataclass(frozen=True)
class RegionGraph:
    """Region graph tables."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    adjacency: pd.DataFrame


def build_nearest_region_graph(
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    neighbours: int = 2,
) -> RegionGraph:
    """Build a directed graph from each region to its nearest neighbours."""

    model_inputs = load_model_inputs(input_dir=input_dir)
    nodes = _build_nodes(model_inputs.regions)
    edges = _build_nearest_edges(nodes, neighbours=neighbours)
    adjacency = _edges_to_adjacency(edges, nodes["region"].to_list())
    return RegionGraph(nodes=nodes, edges=edges, adjacency=adjacency)


def focus_neighbour_orders(
    nodes: pd.DataFrame,
    *,
    focus_region: str,
    first_order: int = 2,
    second_order: int = 3,
) -> pd.DataFrame:
    """Return first- and second-order nearest regions around a focus region.

    This is a distance-ring view, not a mobility-flow estimate:

    - first order: the closest ``first_order`` regions to the focus region;
    - second order: the next ``second_order`` closest regions.
    """

    focus = nodes.loc[nodes["region"].eq(focus_region)]
    if focus.empty:
        raise ValueError(f"Unknown focus_region {focus_region!r}")

    focus_row = focus.iloc[0]
    rows = []
    for row in nodes.itertuples(index=False):
        if row.region == focus_region:
            continue
        rows.append(
            {
                "focus_region": focus_region,
                "region": row.region,
                "distance_km": haversine_km(
                    focus_row["latitude"],
                    focus_row["longitude"],
                    row.latitude,
                    row.longitude,
                ),
            }
        )

    ordered = pd.DataFrame(rows).sort_values("distance_km").reset_index(drop=True)
    ordered["distance_rank"] = ordered.index + 1
    ordered["neighbour_order"] = np.where(
        ordered["distance_rank"].le(first_order),
        1,
        np.where(ordered["distance_rank"].le(first_order + second_order), 2, 0),
    )
    return ordered.loc[ordered["neighbour_order"].gt(0)].copy()


def save_region_graph(
    graph: RegionGraph,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save graph node, edge, and adjacency tables."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "nodes": out / "region_graph_nodes.csv",
        "edges": out / "region_graph_edges_top2_nearest.csv",
        "adjacency": out / "region_graph_adjacency_top2_nearest.csv",
    }
    graph.nodes.to_csv(paths["nodes"], index=False)
    graph.edges.to_csv(paths["edges"], index=False)
    graph.adjacency.to_csv(paths["adjacency"])
    return paths


def _build_nodes(regions: list[str]) -> pd.DataFrame:
    missing = [region for region in regions if region not in REGION_CENTROIDS]
    if missing:
        raise ValueError(f"Missing centroid coordinates for regions: {missing}")

    rows = []
    for region in regions:
        lat, lon = REGION_CENTROIDS[region]
        rows.append({"region": region, "latitude": lat, "longitude": lon})
    return pd.DataFrame(rows)


def _build_nearest_edges(nodes: pd.DataFrame, *, neighbours: int) -> pd.DataFrame:
    rows = []
    for _, source in nodes.iterrows():
        distances = []
        for _, target in nodes.iterrows():
            if source["region"] == target["region"]:
                continue
            distance_km = haversine_km(
                source["latitude"],
                source["longitude"],
                target["latitude"],
                target["longitude"],
            )
            distances.append((target["region"], distance_km))

        nearest = sorted(distances, key=lambda item: item[1])[:neighbours]
        inverse_distances = np.asarray([1.0 / distance for _, distance in nearest])
        weights = inverse_distances / inverse_distances.sum()
        for rank, ((target_region, distance_km), weight) in enumerate(
            zip(nearest, weights, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "source_region": source["region"],
                    "target_region": target_region,
                    "rank": rank,
                    "distance_km": distance_km,
                    "distance_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def _edges_to_adjacency(edges: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    adjacency = pd.DataFrame(0.0, index=regions, columns=regions)
    for row in edges.itertuples(index=False):
        adjacency.loc[row.source_region, row.target_region] = row.distance_weight
    return adjacency


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two latitude/longitude points."""

    radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(d_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    )
    return float(2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))
