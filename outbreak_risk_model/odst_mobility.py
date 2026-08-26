"""Build student-origin mobility graph tables from Census 2021 ODST data.

ODST is different from ODWP commuting. It describes usual residents aged 16+
who were living at a different address one year before the census, where that
previous address was a student term-time or boarding-school address in the UK.
It is therefore a student-origin movement layer, not daily school commuting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ODST_ZIP = REPO_ROOT / "data" / "raw" / "odst" / "odst01ew.zip"
DEFAULT_GRAPH_OUTPUT_DIR = Path(__file__).resolve().parent / "graph_outputs"


@dataclass(frozen=True)
class ODSTMobilityGraph:
    """Student-origin mobility graph tables."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    adjacency: pd.DataFrame


def build_odst_mobility_graph(
    odst_zip: Path | str = DEFAULT_ODST_ZIP,
    *,
    level: str = "region",
    england_only: bool = True,
    include_self_loops: bool = True,
    min_count: int = 1,
) -> ODSTMobilityGraph:
    """Build directed mobility graph tables from an ODST ZIP file."""

    edges = load_odst_flow_edges(
        odst_zip,
        level=level,
        england_only=england_only,
        include_self_loops=include_self_loops,
        min_count=min_count,
    )
    nodes = build_student_flow_nodes(edges)
    adjacency = student_flow_edges_to_adjacency(edges, nodes["name"].to_list())
    return ODSTMobilityGraph(nodes=nodes, edges=edges, adjacency=adjacency)


def load_odst_flow_edges(
    odst_zip: Path | str = DEFAULT_ODST_ZIP,
    *,
    level: str = "region",
    england_only: bool = True,
    include_self_loops: bool = True,
    min_count: int = 1,
) -> pd.DataFrame:
    """Read ODST rows and return normalized directed student-origin edges."""

    data = _read_odst_csv_from_zip(Path(odst_zip), level=level)
    source_code, source_name, target_code, target_name = _odst_area_columns(data)
    flows = data.copy()
    flows["student_origin_count"] = pd.to_numeric(flows["Count"], errors="coerce").fillna(0)

    if england_only:
        flows = flows.loc[
            flows[source_code].astype(str).str.startswith("E")
            & flows[target_code].astype(str).str.startswith("E")
        ].copy()

    if not include_self_loops:
        flows = flows.loc[~flows[source_code].eq(flows[target_code])].copy()

    flows = flows.loc[flows["student_origin_count"].ge(min_count)].copy()
    edges = (
        flows.groupby(
            [source_code, source_name, target_code, target_name],
            as_index=False,
            dropna=False,
        )["student_origin_count"]
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
    edges["student_origin_count"] = edges["student_origin_count"].astype(int)
    edges["is_self_loop"] = edges["source_code"].eq(edges["target_code"])

    source_total = edges.groupby("source_code")["student_origin_count"].transform("sum")
    target_total = edges.groupby("target_code")["student_origin_count"].transform("sum")
    edges["raw_mobility_weight"] = edges["student_origin_count"]
    edges["mobility_weight"] = edges["student_origin_count"] / source_total
    edges["destination_origin_share"] = edges["student_origin_count"] / target_total
    edges["edge_source"] = "odst_student_origin"

    return edges.sort_values(
        ["source_name", "student_origin_count", "target_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_student_flow_nodes(edges: pd.DataFrame) -> pd.DataFrame:
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

    outflow = edges.groupby("source_code")["student_origin_count"].sum().rename("total_outflow")
    inflow = edges.groupby("target_code")["student_origin_count"].sum().rename("total_inflow")
    internal = (
        edges.loc[edges["is_self_loop"]]
        .set_index("source_code")["student_origin_count"]
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


def student_flow_edges_to_adjacency(edges: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return source-row-normalized student-origin mobility adjacency."""

    adjacency = pd.DataFrame(0.0, index=names, columns=names)
    for row in edges.itertuples(index=False):
        adjacency.loc[row.source_name, row.target_name] = row.mobility_weight
    return adjacency


def save_odst_mobility_graph(
    graph: ODSTMobilityGraph,
    *,
    output_dir: Path | str = DEFAULT_GRAPH_OUTPUT_DIR,
    prefix: str,
) -> dict[str, Path]:
    """Save student-origin graph node, edge, and adjacency tables."""

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


def _read_odst_csv_from_zip(path: Path, *, level: str) -> pd.DataFrame:
    file_by_level = {
        "region": "ODST01EW_RGN.csv",
        "utla": "ODST01EW_UTLA.csv",
        "ltla": "ODST01EW_LTLA.csv",
        "msoa": "ODST01EW_MSOA.csv",
    }
    try:
        member = file_by_level[level]
    except KeyError as exc:
        options = ", ".join(sorted(file_by_level))
        raise ValueError(f"Unknown ODST level {level!r}. Choose one of: {options}") from exc

    with ZipFile(path) as archive:
        with archive.open(member) as csv_file:
            return pd.read_csv(csv_file)


def _odst_area_columns(data: pd.DataFrame) -> tuple[str, str, str, str]:
    columns = list(data.columns)
    required = {"Count"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing expected ODST columns: {sorted(missing)}")
    if len(columns) < 5:
        raise ValueError("ODST CSV must contain source area, target area, and count columns")
    return columns[0], columns[1], columns[2], columns[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build England ODST student-origin mobility graph tables.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ODST_ZIP,
        help="ODST ZIP path (default: data/raw/odst/odst01ew.zip).",
    )
    parser.add_argument("--level", choices=["region", "utla", "ltla", "msoa"], default="region")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_OUTPUT_DIR)
    parser.add_argument("--include-wales", action="store_true", help="Keep non-England ODST rows too.")
    parser.add_argument("--drop-self-loops", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    prefix = args.prefix or f"odst_england_{args.level}_student_mobility"
    graph = build_odst_mobility_graph(
        args.zip,
        level=args.level,
        england_only=not args.include_wales,
        include_self_loops=not args.drop_self_loops,
        min_count=args.min_count,
    )
    paths = save_odst_mobility_graph(graph, output_dir=args.output_dir, prefix=prefix)

    print("Saved ODST student-origin mobility graph outputs:")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")


if __name__ == "__main__":
    main()
