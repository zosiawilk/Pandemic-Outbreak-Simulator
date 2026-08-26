"""Generate transparent network visualisations from the model inputs."""

from html import escape
from pathlib import Path
from math import sqrt
import shutil
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outbreak_probability_model.model import load_default_inputs


OUTPUT = Path(__file__).resolve().parent / "figures"
GRAPH_OUTPUTS = ROOT / "outbreak_risk_model" / "graph_outputs"

REGION_POSITIONS = {
    "North East": (-1.62, 54.98),
    "North West": (-2.65, 53.72),
    "Yorkshire and The Humber": (-1.35, 53.78),
    "East Midlands": (-0.95, 52.90),
    "West Midlands": (-2.15, 52.48),
    "East of England": (0.45, 52.20),
    "London": (-0.10, 51.48),
    "South East": (0.10, 51.05),
    "South West": (-3.25, 50.85),
}


def projection(points, width, height, margin):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    def project(x, y):
        px = margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)
        py = height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)
        return px, py

    return project


def svg_header(width, height, *, arrows=False):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.label{font-size:14px}.note{font-size:12px;fill:#526175}</style>',
    ]
    if arrows:
        parts.append('<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto"><path d="M0,0 L0,5 L5,2.5 z" fill="#64748b"/></marker></defs>')
    return parts


def regional_mobility_svg():
    inputs = load_default_inputs()
    populations = inputs.population.sum(axis=1)
    mobility = inputs.mobility_matrix
    width, height, margin = 820, 760, 85
    project = projection(list(REGION_POSITIONS.values()), width, height, margin)
    positions = {name: project(*REGION_POSITIONS[name]) for name in inputs.regions}
    parts = svg_header(width, height, arrows=True)

    threshold = 0.10
    for q, source in enumerate(inputs.regions):
        for r, target in enumerate(inputs.regions):
            weight = float(mobility[q, r])
            if source == target or weight < threshold:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            dx, dy = x2 - x1, y2 - y1
            distance = max(sqrt(dx * dx + dy * dy), 1.0)
            ux, uy = dx / distance, dy / distance
            start = (x1 + ux * 20, y1 + uy * 20)
            end = (x2 - ux * 22, y2 - uy * 22)
            bend = 0.07 * distance
            mx = (start[0] + end[0]) / 2 - uy * bend
            my = (start[1] + end[1]) / 2 + ux * bend
            line_width = 0.7 + 5.0 * weight
            opacity = 0.22 + 0.60 * min(weight / 0.55, 1.0)
            parts.append(
                f'<path d="M{start[0]:.1f},{start[1]:.1f} Q{mx:.1f},{my:.1f} {end[0]:.1f},{end[1]:.1f}" '
                f'fill="none" stroke="#64748b" stroke-width="{line_width:.2f}" stroke-opacity="{opacity:.2f}" marker-end="url(#arrow)"/>'
            )

    offsets = {
        "North East": (16, -14), "North West": (-92, -18),
        "Yorkshire and The Humber": (18, -10), "East Midlands": (18, -8),
        "West Midlands": (-112, 28), "East of England": (-128, -18),
        "London": (24, 5), "South East": (22, 27), "South West": (-54, 34),
    }
    for index, name in enumerate(inputs.regions):
        x, y = positions[name]
        population = float(populations[index])
        radius = 13.0 * sqrt(population / 3_000_000.0)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#3b82c4" stroke="white" stroke-width="2"/>')
        dx, dy = offsets[name]
        parts.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" class="label">{escape(name)}</text>')

    legend_x, legend_y = 55, 105
    parts.append(f'<text x="{legend_x}" y="{legend_y - 46}" class="note">Node area: regional population</text>')
    for offset, population, label in [(0, 3_000_000, "3m"), (78, 6_000_000, "6m"), (170, 9_000_000, "9m")]:
        radius = 13.0 * sqrt(population / 3_000_000.0)
        cx = legend_x + offset + radius
        parts.append(f'<circle cx="{cx:.1f}" cy="{legend_y:.1f}" r="{radius:.1f}" fill="#3b82c4" stroke="white" stroke-width="1.5"/>')
        parts.append(f'<text x="{cx + radius + 7:.1f}" y="{legend_y + 4:.1f}" class="note">{label}</text>')
    parts.append(f'<text x="{width - 65}" y="{height - 28}" text-anchor="end" class="note">Displayed arrows: W_q,r ≥ 0.10</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def utla_topology_svg():
    nodes = pd.read_csv(GRAPH_OUTPUTS / "uk_utla_2025_nodes.csv")
    edges = pd.read_csv(GRAPH_OUTPUTS / "uk_utla_2025_edges.csv")
    width, height, margin = 760, 900, 55
    raw = list(zip(nodes["longitude"], nodes["latitude"]))
    project = projection(raw, width, height, margin)
    positions = {row.name: project(row.longitude, row.latitude) for row in nodes.itertuples(index=False)}
    parts = svg_header(width, height)
    for edge in edges.itertuples(index=False):
        x1, y1 = positions[edge.source_name]
        x2, y2 = positions[edge.target_name]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="0.65" stroke-opacity="0.24"/>')
    for row in nodes.itertuples(index=False):
        x, y = positions[row.name]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="#3478b9" stroke="white" stroke-width="0.65"/>')
    parts.append('</svg>')
    return "\n".join(parts)


def generate_all_formats():
    """Write SVG figures and PDF/PNG copies for display or reports."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "england_population_mobility_network": regional_mobility_svg(),
        "uk_utla_exploratory_topology": utla_topology_svg(),
    }
    paths = {}
    converter = shutil.which("rsvg-convert")
    for stem, svg_text in figures.items():
        svg_path = OUTPUT / f"{stem}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        paths[f"{stem}_svg"] = svg_path
        if converter is not None:
            pdf_path = OUTPUT / f"{stem}.pdf"
            png_path = OUTPUT / f"{stem}.png"
            subprocess.run(
                [converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
                check=True,
            )
            subprocess.run(
                [converter, "-f", "png", "-w", "1400", "-o", str(png_path), str(svg_path)],
                check=True,
            )
            paths[f"{stem}_pdf"] = pdf_path
            paths[f"{stem}_png"] = png_path
    return paths


if __name__ == "__main__":
    generate_all_formats()
