"""Create an SVG visualization of the English-region mixing graph."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from .data_loader import DEFAULT_INPUT_DIR
from .region_graph import (
    DEFAULT_OUTPUT_DIR,
    build_nearest_region_graph,
    focus_neighbour_orders,
    save_region_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and plot the region graph.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neighbours", type=int, default=2)
    parser.add_argument("--focus-region", default="London")
    parser.add_argument("--first-order", type=int, default=2)
    parser.add_argument("--second-order", type=int, default=3)
    args = parser.parse_args()

    graph = build_nearest_region_graph(
        input_dir=args.input_dir,
        neighbours=args.neighbours,
    )
    paths = save_region_graph(graph, output_dir=args.output_dir)
    focus_orders = focus_neighbour_orders(
        graph.nodes,
        focus_region=args.focus_region,
        first_order=args.first_order,
        second_order=args.second_order,
    )
    focus_orders_path = args.output_dir / f"region_graph_focus_{safe_name(args.focus_region)}_orders.csv"
    focus_orders.to_csv(focus_orders_path, index=False)
    plot_path = args.output_dir / "region_graph_top2_nearest.svg"
    focus_plot_path = args.output_dir / f"region_graph_focus_{safe_name(args.focus_region)}.svg"
    write_region_graph_svg(
        graph.nodes,
        graph.edges,
        output_path=plot_path,
        focus_region=args.focus_region,
        show_all_edges=True,
    )
    write_region_graph_svg(
        graph.nodes,
        graph.edges,
        output_path=focus_plot_path,
        focus_region=args.focus_region,
        show_all_edges=False,
        focus_orders=focus_orders,
    )

    print("Saved region graph outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"focus_orders: {focus_orders_path}")
    print(f"full_plot: {plot_path}")
    print(f"focus_plot: {focus_plot_path}")


def write_region_graph_svg(
    nodes,
    edges,
    *,
    output_path: Path,
    focus_region: str | None = None,
    show_all_edges: bool = True,
    focus_orders=None,
) -> None:
    """Write a dependency-free SVG using centroid longitude/latitude coordinates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 900
    margin = 75
    lon_min = float(nodes["longitude"].min())
    lon_max = float(nodes["longitude"].max())
    lat_min = float(nodes["latitude"].min())
    lat_max = float(nodes["latitude"].max())

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = margin + (lon - lon_min) / (lon_max - lon_min) * (width - 2 * margin)
        y = height - margin - (lat - lat_min) / (lat_max - lat_min) * (height - 2 * margin)
        return x, y

    positions = {
        row.region: project(row.longitude, row.latitude)
        for row in nodes.itertuples(index=False)
    }
    order_by_region = {}
    if focus_orders is not None:
        order_by_region = dict(
            zip(focus_orders["region"], focus_orders["neighbour_order"], strict=True)
        )
    focus_targets = {region for region, order in order_by_region.items() if order == 1}
    second_order_targets = {region for region, order in order_by_region.items() if order == 2}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #202124; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #5f6368; }",
        ".label { font-size: 13px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
    ]
    if show_all_edges:
        title = "England Region Graph: All Top-2 Nearest-Neighbour Links"
    else:
        title = "England Region Graph: First- and Second-Order Neighbourhood"
    parts.append(f'<text x="38" y="38" class="title">{escape(title)}</text>')
    if focus_region:
        parts.append(
            f'<text x="38" y="62" class="subtitle">Focus region: {escape(focus_region)}</text>'
        )

    if show_all_edges:
        for row in edges.itertuples(index=False):
            is_focus_edge = row.source_region == focus_region
            x1, y1 = positions[row.source_region]
            x2, y2 = positions[row.target_region]
            color = "#d95f02" if is_focus_edge else "#9aa4b2"
            width_px = 4 if is_focus_edge else 2
            opacity = 0.9 if is_focus_edge else 0.45
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{width_px}" stroke-opacity="{opacity}"/>'
            )
    elif focus_region:
        x1, y1 = positions[focus_region]
        for region in focus_targets:
            x2, y2 = positions[region]
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#d95f02" stroke-width="4" stroke-opacity="0.9"/>'
            )
        for region in second_order_targets:
            x2, y2 = positions[region]
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'stroke="#e6ab02" stroke-width="2.5" stroke-opacity="0.75" '
                'stroke-dasharray="7 6"/>'
            )

    for row in nodes.itertuples(index=False):
        x, y = positions[row.region]
        is_focus = row.region == focus_region
        is_focus_target = row.region in focus_targets
        is_second_order = row.region in second_order_targets
        radius = 14 if is_focus else 12 if is_focus_target else 11 if is_second_order else 8
        color = (
            "#1b9e77"
            if is_focus
            else "#7570b3"
            if is_focus_target
            else "#e6ab02"
            if is_second_order
            else "#386cb0"
        )
        label_weight = "700" if is_focus else "400"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            'stroke="white" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + 14:.1f}" y="{y - 12:.1f}" class="label" '
            f'font-weight="{label_weight}">{escape(row.region)}</text>'
        )

    if not show_all_edges:
        legend_y = height - 82
        legend = [
            ("#1b9e77", "Focus region"),
            ("#7570b3", "First order"),
            ("#e6ab02", "Second order"),
            ("#386cb0", "Other region"),
        ]
        for idx, (color, label) in enumerate(legend):
            x = 42 + idx * 170
            parts.append(f'<circle cx="{x}" cy="{legend_y}" r="8" fill="{color}"/>')
            parts.append(f'<text x="{x + 14}" y="{legend_y + 5}" class="subtitle">{label}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("&", "and")


if __name__ == "__main__":
    main()
