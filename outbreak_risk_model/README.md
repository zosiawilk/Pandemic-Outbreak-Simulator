# Network input preprocessing

This directory contains the optional network-preprocessing step for the final
six-week measles outbreak-probability model. The forecast itself lives in
`outbreak_probability_model/`; the code here constructs or inspects the
geographic and mobility matrices it consumes.

## Notebook

`Measles_Graph_Model.ipynb` is optional step 0 in the documented run order. Run
it only when the population, protection, geography, commuting, or student-flow
inputs change. The checked-in processed matrices allow steps 1–8 to run without
rebuilding the network.

## Scripts

| File | Purpose |
|---|---|
| `data_loader.py` | Load region-age population, protection, and case inputs. |
| `region_graph.py` | Build a simple regional nearest-neighbour graph. |
| `geography_graph.py` | Build graph tables from ONS boundary GeoJSON. |
| `odwp_mobility.py` | Process Census 2021 residence-to-workplace flows. |
| `odst_mobility.py` | Process student residence-to-study flows. |
| `measles_graph.py` | Combine node attributes with geographic or mobility edges. |
| `visualize_region_graph.py` | Produce dependency-free SVG graph diagnostics. |
| `notebook_helpers.py` | Small display and inspection helpers for the notebook. |
| `network_figures.py` | Produce retained network diagrams from graph tables. |

## Retained processed inputs

The final model directly reads:

```text
graph_outputs/odwp_england_region_mobility_adjacency.csv
graph_outputs/odst_england_region_student_mobility_adjacency.csv
```

The retained UTLA node and edge tables support reproducible topology figures.
Other graph CSV/SVG files are generated diagnostics and are ignored by Git.

Mobility weights describe observed movement patterns; they are model inputs
rather than independently fitted transmission parameters.
