"""Optional network-input preprocessing helpers."""

__all__ = [
    "available_region_age_groups",
    "build_and_show_geography_graph",
    "build_nearest_region_graph",
    "build_odwp_mobility_graph",
    "default_odwp_csv",
    "result_summary",
    "show_geography_graph",
    "save_odwp_mobility_graph",
    "save_region_graph",
]


def __getattr__(name):
    if name in {
        "build_odwp_mobility_graph",
        "default_odwp_csv",
        "save_odwp_mobility_graph",
    }:
        from . import odwp_mobility

        return getattr(odwp_mobility, name)
    if name in {"build_nearest_region_graph", "save_region_graph"}:
        from . import region_graph

        return getattr(region_graph, name)
    if name in {
        "available_region_age_groups",
        "build_and_show_geography_graph",
        "load_measles_graph_tables",
        "node_measles_summary",
        "node_neighbours",
        "result_summary",
        "show_geography_graph",
    }:
        from . import notebook_helpers

        return getattr(notebook_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
