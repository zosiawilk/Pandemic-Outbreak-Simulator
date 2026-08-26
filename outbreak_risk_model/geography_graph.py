"""Build whole-UK geography graphs from downloaded ONS GeoJSON boundaries.

Use this for Region and Local authority (upper tier) boundary files downloaded
from the ONS Digital UK TopoJSON/GeoJSON tool.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from .region_graph import DEFAULT_OUTPUT_DIR, haversine_km


@dataclass(frozen=True)
class GeographyGraph:
    """Generic geography graph tables."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    adjacency: pd.DataFrame


def build_geography_graph_from_geojson(
    geojson_path: Path | str,
    *,
    neighbours: int = 4,
) -> GeographyGraph:
    """Build a nearest-neighbour graph from GeoJSON feature centroids."""

    nodes = load_geojson_nodes(geojson_path)
    edges = build_nearest_edges(nodes, neighbours=neighbours)
    adjacency = edges_to_adjacency(edges, nodes["name"].to_list())
    return GeographyGraph(nodes=nodes, edges=edges, adjacency=adjacency)


def load_geojson_nodes(geojson_path: Path | str) -> pd.DataFrame:
    """Read GeoJSON features and return one centroid node per feature."""

    path = Path(geojson_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not features:
        raise ValueError(f"No GeoJSON features found in {path}")

    rows = []
    for idx, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        name = _pick_property(properties, suffixes=("nm", "name"), fallback=f"feature_{idx}")
        code = _pick_property(properties, suffixes=("cd", "code"), fallback=str(idx))
        points = list(_iter_lon_lat(feature.get("geometry")))
        if not points:
            continue
        lon, lat = _centroid_from_points(points)
        rows.append(
            {
                "code": code,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "n_boundary_points": len(points),
            }
        )

    if not rows:
        raise ValueError(f"No usable point coordinates found in {path}")
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def build_nearest_edges(nodes: pd.DataFrame, *, neighbours: int) -> pd.DataFrame:
    """Connect each node to its nearest neighbours by centroid distance."""

    rows = []
    for _, source in nodes.iterrows():
        distances = []
        for _, target in nodes.iterrows():
            if source["name"] == target["name"]:
                continue
            distance_km = haversine_km(
                source["latitude"],
                source["longitude"],
                target["latitude"],
                target["longitude"],
            )
            distances.append((target["code"], target["name"], distance_km))

        nearest = sorted(distances, key=lambda item: item[2])[:neighbours]
        inverse_distances = np.asarray([1.0 / max(distance, 1e-9) for _, _, distance in nearest])
        weights = inverse_distances / inverse_distances.sum()
        for rank, ((target_code, target_name, distance_km), weight) in enumerate(
            zip(nearest, weights, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "source_code": source["code"],
                    "source_name": source["name"],
                    "target_code": target_code,
                    "target_name": target_name,
                    "rank": rank,
                    "distance_km": distance_km,
                    "distance_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def edges_to_adjacency(edges: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    adjacency = pd.DataFrame(0.0, index=names, columns=names)
    for row in edges.itertuples(index=False):
        adjacency.loc[row.source_name, row.target_name] = row.distance_weight
    return adjacency


def save_geography_graph(
    graph: GeographyGraph,
    *,
    output_dir: Path | str,
    prefix: str,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "nodes": out / f"{prefix}_nodes.csv",
        "edges": out / f"{prefix}_edges.csv",
        "adjacency": out / f"{prefix}_adjacency.csv",
    }
    graph.nodes.to_csv(paths["nodes"], index=False)
    graph.edges.to_csv(paths["edges"], index=False)
    graph.adjacency.to_csv(paths["adjacency"])
    return paths


def write_geography_graph_svg(
    graph: GeographyGraph,
    *,
    output_path: Path | str,
    title: str,
    max_label_count: int = 80,
) -> None:
    """Write a lightweight SVG of the whole geography graph."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        geography_graph_svg(
            graph,
            title=title,
            max_label_count=max_label_count,
        ),
        encoding="utf-8",
    )


def geography_graph_svg(
    graph: GeographyGraph,
    *,
    title: str,
    max_label_count: int = 80,
) -> str:
    """Return a lightweight SVG of the whole geography graph."""

    nodes = graph.nodes
    edges = graph.edges
    width = 900
    height = 1050
    margin = 70
    lon_min = float(nodes["longitude"].min())
    lon_max = float(nodes["longitude"].max())
    lat_min = float(nodes["latitude"].min())
    lat_max = float(nodes["latitude"].max())
    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = margin + (lon - lon_min) / lon_span * (width - 2 * margin)
        y = height - margin - (lat - lat_min) / lat_span * (height - 2 * margin)
        return x, y

    positions = {
        row.name: project(row.longitude, row.latitude)
        for row in nodes.itertuples(index=False)
    }
    show_labels = len(nodes) <= max_label_count

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #202124; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #5f6368; }",
        ".label { font-size: 10px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        f'<text x="38" y="38" class="title">{escape(title)}</text>',
        f'<text x="38" y="60" class="subtitle">Nodes: {len(nodes)} | Edges: {len(edges)}</text>',
    ]

    for row in edges.itertuples(index=False):
        x1, y1 = positions[row.source_name]
        x2, y2 = positions[row.target_name]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#9aa4b2" stroke-width="1" stroke-opacity="0.32"/>'
        )

    radius = 4.6 if len(nodes) <= 50 else 2.4
    for row in nodes.itertuples(index=False):
        x, y = positions[row.name]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="#386cb0" '
            'stroke="white" stroke-width="0.8"/>'
        )
        if show_labels:
            parts.append(
                f'<text x="{x + 6:.1f}" y="{y - 5:.1f}" class="label">{escape(row.name)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _pick_property(properties: dict, *, suffixes: tuple[str, ...], fallback: str) -> str:
    lowered = {str(key).lower(): key for key in properties}
    for suffix in suffixes:
        for lower_key, original_key in lowered.items():
            if lower_key.endswith(suffix):
                value = properties[original_key]
                if value not in (None, ""):
                    return str(value)
    return fallback


def _iter_lon_lat(geometry: dict | None):
    if not geometry:
        return
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geom_type == "Point":
        yield float(coordinates[0]), float(coordinates[1])
    elif geom_type in {"LineString", "MultiPoint"}:
        for point in coordinates:
            yield float(point[0]), float(point[1])
    elif geom_type in {"Polygon", "MultiLineString"}:
        for line in coordinates:
            for point in line:
                yield float(point[0]), float(point[1])
    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                for point in ring:
                    yield float(point[0]), float(point[1])
    elif geom_type == "GeometryCollection":
        for sub_geometry in geometry.get("geometries", []):
            yield from _iter_lon_lat(sub_geometry)


def _centroid_from_points(points: list[tuple[float, float]]) -> tuple[float, float]:
    values = np.asarray(points, dtype=float)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def safe_prefix(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a UK geography graph from GeoJSON.")
    parser.add_argument("--geojson", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--neighbours", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    name = args.name or args.geojson.stem
    prefix = safe_prefix(name)
    graph = build_geography_graph_from_geojson(args.geojson, neighbours=args.neighbours)
    paths = save_geography_graph(graph, output_dir=args.output_dir, prefix=prefix)
    svg_path = args.output_dir / f"{prefix}_graph.svg"
    write_geography_graph_svg(
        graph,
        output_path=svg_path,
        title=args.title or f"{name}: nearest-neighbour graph",
    )

    print("Saved geography graph outputs:")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"plot: {svg_path}")


if __name__ == "__main__":
    main()
