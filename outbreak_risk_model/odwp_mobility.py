"""Build England origin-destination mobility graph tables from ODWP data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ODWP_DIR = REPO_ROOT / "data" / "raw" / "odwp" / "odwp01ew"
DEFAULT_GRAPH_OUTPUT_DIR = Path(__file__).resolve().parent / "graph_outputs"
NORMAL_WORKPLACE_INDICATOR = 3


@dataclass(frozen=True)
class ODWPMobilityGraph:
    """Origin-destination mobility graph tables."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    adjacency: pd.DataFrame


def build_odwp_mobility_graph(
    odwp_csv: Path | str,
    *,
    england_only: bool = True,
    include_self_loops: bool = True,
    min_count: int = 1,
) -> ODWPMobilityGraph:
    """Build directed mobility graph tables from an ODWP origin-destination CSV.

    ODWP rows are usual-residence area to workplace-area counts. This keeps
    ordinary UK workplace flows only, then optionally restricts both source and
    target to England geography codes.
    """

    flows = load_odwp_flow_edges(
        odwp_csv,
        england_only=england_only,
        include_self_loops=include_self_loops,
        min_count=min_count,
    )
    nodes = build_flow_nodes(flows)
    adjacency = flow_edges_to_adjacency(flows, nodes["name"].to_list())
    return ODWPMobilityGraph(nodes=nodes, edges=flows, adjacency=adjacency)


def load_odwp_flow_edges(
    odwp_csv: Path | str,
    *,
    england_only: bool = True,
    include_self_loops: bool = True,
    min_count: int = 1,
) -> pd.DataFrame:
    """Read ODWP rows and return normalized directed mobility edges."""

    path = Path(odwp_csv)
    data = pd.read_csv(path)
    source_code, source_name, target_code, target_name = _odwp_area_columns(data)

    flows = data.loc[
        data["Place of work indicator (4 categories) code"].eq(NORMAL_WORKPLACE_INDICATOR)
    ].copy()
    flows["commuter_count"] = pd.to_numeric(flows["Count"], errors="coerce").fillna(0)

    if england_only:
        flows = flows.loc[
            flows[source_code].astype(str).str.startswith("E")
            & flows[target_code].astype(str).str.startswith("E")
        ].copy()

    if not include_self_loops:
        flows = flows.loc[~flows[source_code].eq(flows[target_code])].copy()

    flows = flows.loc[flows["commuter_count"].ge(min_count)].copy()
    edges = (
        flows.groupby(
            [source_code, source_name, target_code, target_name],
            as_index=False,
            dropna=False,
        )["commuter_count"]
        .sum()
        .rename(
            columns={
                source_code: "source_code",
                source_name: "source_name",
                target_code: "target_code",
                target_name: "target_name",
            }
        )
    )
    edges["commuter_count"] = edges["commuter_count"].astype(int)
    edges["is_self_loop"] = edges["source_code"].eq(edges["target_code"])

    source_total = edges.groupby("source_code")["commuter_count"].transform("sum")
    target_total = edges.groupby("target_code")["commuter_count"].transform("sum")
    edges["raw_mobility_weight"] = edges["commuter_count"]
    edges["mobility_weight"] = edges["commuter_count"] / source_total
    edges["destination_residence_share"] = edges["commuter_count"] / target_total

    return edges.sort_values(
        ["source_name", "commuter_count", "target_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_flow_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize inflow/outflow totals for all nodes present in edge data."""

    source_nodes = edges[["source_code", "source_name"]].rename(
        columns={"source_code": "code", "source_name": "name"}
    )
    target_nodes = edges[["target_code", "target_name"]].rename(
        columns={"target_code": "code", "target_name": "name"}
    )
    nodes = (
        pd.concat([source_nodes, target_nodes], ignore_index=True)
        .drop_duplicates()
        .sort_values("name")
        .reset_index(drop=True)
    )

    outflow = edges.groupby("source_code")["commuter_count"].sum().rename("total_outflow")
    inflow = edges.groupby("target_code")["commuter_count"].sum().rename("total_inflow")
    internal = (
        edges.loc[edges["is_self_loop"]]
        .set_index("source_code")["commuter_count"]
        .rename("internal_flow")
    )

    nodes = nodes.merge(outflow, left_on="code", right_index=True, how="left")
    nodes = nodes.merge(inflow, left_on="code", right_index=True, how="left")
    nodes = nodes.merge(internal, left_on="code", right_index=True, how="left")
    for column in ("total_outflow", "total_inflow", "internal_flow"):
        nodes[column] = nodes[column].fillna(0).astype(int)
    nodes["external_outflow"] = nodes["total_outflow"] - nodes["internal_flow"]
    nodes["external_inflow"] = nodes["total_inflow"] - nodes["internal_flow"]
    nodes["net_external_inflow"] = nodes["external_inflow"] - nodes["external_outflow"]
    return nodes


def flow_edges_to_adjacency(edges: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return source-row-normalized mobility adjacency by geography name."""

    adjacency = pd.DataFrame(0.0, index=names, columns=names)
    for row in edges.itertuples(index=False):
        adjacency.loc[row.source_name, row.target_name] = row.mobility_weight
    return adjacency


def save_odwp_mobility_graph(
    graph: ODWPMobilityGraph,
    *,
    output_dir: Path | str = DEFAULT_GRAPH_OUTPUT_DIR,
    prefix: str,
) -> dict[str, Path]:
    """Save mobility graph node, edge, and adjacency tables."""

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


def _odwp_area_columns(data: pd.DataFrame) -> tuple[str, str, str, str]:
    columns = list(data.columns)
    required = {
        "Place of work indicator (4 categories) code",
        "Place of work indicator (4 categories) label",
        "Count",
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing expected ODWP columns: {sorted(missing)}")
    if len(columns) < 7:
        raise ValueError("ODWP CSV must contain source area, target area, indicator, and count columns")
    return columns[0], columns[1], columns[2], columns[3]


def default_odwp_csv(level: str) -> Path:
    """Return the standard ODWP01EW CSV path for a geography level."""

    file_by_level = {
        "region": "ODWP01EW_RGN.csv",
        "utla": "ODWP01EW_UTLA.csv",
        "ltla": "ODWP01EW_LTLA.csv",
        "msoa": "ODWP01EW_MSOA.csv",
        "oa": "ODWP01EW_OA.csv",
    }
    try:
        return DEFAULT_ODWP_DIR / file_by_level[level]
    except KeyError as exc:
        options = ", ".join(sorted(file_by_level))
        raise ValueError(f"Unknown ODWP level {level!r}. Choose one of: {options}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build England ODWP mobility graph tables.")
    parser.add_argument("--level", choices=["region", "utla", "ltla", "msoa", "oa"], default="region")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "ODWP CSV path. If omitted, use data/raw/odwp/odwp01ew/"
            "ODWP01EW_<LEVEL>.csv inside the repository."
        ),
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_OUTPUT_DIR)
    parser.add_argument("--include-wales", action="store_true", help="Keep non-England ODWP rows too.")
    parser.add_argument("--drop-self-loops", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    csv_path = args.csv or default_odwp_csv(args.level)
    prefix = args.prefix or f"odwp_england_{args.level}_mobility"
    graph = build_odwp_mobility_graph(
        csv_path,
        england_only=not args.include_wales,
        include_self_loops=not args.drop_self_loops,
        min_count=args.min_count,
    )
    paths = save_odwp_mobility_graph(graph, output_dir=args.output_dir, prefix=prefix)

    print("Saved ODWP mobility graph outputs:")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")


if __name__ == "__main__":
    main()
