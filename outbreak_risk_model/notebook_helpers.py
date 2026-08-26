"""Small helpers for using the outbreak-risk model in Jupyter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import DEFAULT_INPUT_DIR, load_model_inputs
from .geography_graph import (
    build_geography_graph_from_geojson,
    geography_graph_svg,
)
from .measles_graph import MeaslesGraphTables


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_OUTPUT_DIR = PACKAGE_DIR / "graph_outputs"


def available_region_age_groups(
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
) -> pd.DataFrame:
    """Return available region and age-group combinations."""

    model_inputs = load_model_inputs(input_dir=input_dir)
    return pd.DataFrame(
        [
            {"region": region, "age_group": age_group}
            for region in model_inputs.regions
            for age_group in model_inputs.age_groups
        ]
    )


def result_summary(result) -> pd.DataFrame:
    """Return one-row summary output that displays nicely in notebooks."""

    return pd.DataFrame(
        [
            {
                "region": result.region,
                "age_group": result.age_group,
                "current_cases": result.current_cases,
                "outbreak_threshold": result.outbreak_threshold,
                "target_population": result.target_population,
                "horizon_weeks": result.horizon_weeks,
                "n_simulations": result.n_simulations,
                "outbreak_probability": result.outbreak_probability,
                "risk_level": result.risk_level,
                "expected_future_cases": result.expected_future_cases,
                "model": result.model,
                "assumptions": " ".join(result.assumptions),
            }
        ]
    )


def build_and_show_geography_graph(
    geojson_path: Path | str,
    *,
    title: str | None = None,
    neighbours: int = 4,
    max_label_count: int = 80,
):
    """Build a geography graph and display it inline in Jupyter.

    Returns ``(graph, svg_display)``. In a notebook, put this as the last line
    of a cell or call ``display(svg_display)``.
    """

    from IPython.display import SVG

    resolved_geojson_path = resolve_project_path(geojson_path)
    graph = build_geography_graph_from_geojson(
        resolved_geojson_path,
        neighbours=neighbours,
    )
    plot_title = title or f"{resolved_geojson_path.stem}: nearest-neighbour graph"
    svg_text = geography_graph_svg(
        graph,
        title=plot_title,
        max_label_count=max_label_count,
    )
    return graph, SVG(data=svg_text)


def resolve_project_path(path: Path | str) -> Path:
    """Resolve notebook paths from either repo root or package directory."""

    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    package_dir = PACKAGE_DIR
    repo_root = package_dir.parent
    for base in (repo_root, package_dir):
        resolved = base / candidate
        if resolved.exists():
            return resolved

    package_relative = package_dir / candidate.name
    if package_relative.exists():
        return package_relative

    return candidate


def show_geography_graph(
    graph,
    *,
    title: str = "Nearest-neighbour graph",
    max_label_count: int = 80,
):
    """Display an already-built geography graph inline in Jupyter."""

    from IPython.display import SVG

    return SVG(
        data=geography_graph_svg(
            graph,
            title=title,
            max_label_count=max_label_count,
        )
    )


def load_measles_graph_tables(
    *,
    prefix: str = "measles_region_graph",
    graph_output_dir: Path | str = DEFAULT_GRAPH_OUTPUT_DIR,
) -> MeaslesGraphTables:
    """Load saved measles graph node, edge, and contact-matrix tables."""

    output_dir = resolve_project_path(graph_output_dir)
    nodes_path = output_dir / f"{prefix}_nodes.csv"
    edges_path = output_dir / f"{prefix}_edges.csv"
    contact_matrix_path = output_dir / f"{prefix}_contact_matrix.csv"
    missing = [
        path
        for path in (nodes_path, edges_path, contact_matrix_path)
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing saved measles graph table(s). Run "
            "`python -m outbreak_risk_model.measles_graph --level region` "
            "from the repo root, or pass graph_output_dir explicitly.\n"
            f"Looked in: {output_dir}\nMissing:\n{missing_text}"
        )
    return MeaslesGraphTables(
        nodes=pd.read_csv(nodes_path),
        edges=pd.read_csv(edges_path),
        contact_matrix=pd.read_csv(contact_matrix_path, index_col=0),
    )


def node_measles_summary(graph: MeaslesGraphTables, region_name: str) -> pd.DataFrame:
    """Return the stored node data for one region/local authority."""

    selected = graph.nodes.loc[graph.nodes["name"].eq(region_name)]
    if selected.empty:
        raise ValueError(f"No graph node named {region_name!r}")
    return selected.T.rename(columns={selected.index[0]: region_name})


def node_neighbours(graph: MeaslesGraphTables, region_name: str) -> pd.DataFrame:
    """Return first- and second-order touching neighbours for one node."""

    return (
        graph.edges.loc[graph.edges["source_name"].eq(region_name)]
        .sort_values(["touching_order", "distance_km"])
        .reset_index(drop=True)
    )
