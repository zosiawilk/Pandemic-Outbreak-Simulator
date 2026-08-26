"""Build a model contact matrix from the Reconnect paper's Figure 4A.

The PDF figure labelled "Total" gives a 16x16 age contact matrix with values
printed to one decimal place.  This script encodes those visible values and maps
them to the seven age groups used by the measles model.

Important limitation:
    These are rounded figure values, not the full-precision S2 Table values.
    Use this when the paper figure is the available source; replace with S2
    Table values if/when the exact supplementary table is available.

Figure orientation:
    In the paper figure, columns are participant age group and rows are contact
    age group.  The simulator expects rows = participant age and columns =
    contact age, so the matrix is transposed before aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_AGE_GROUPS = [
    "under_1",
    "1_to_4",
    "5_to_10",
    "11_to_14",
    "15_to_24",
    "25_to_34",
    "35_and_over",
]

MODEL_AGE_RANGES = {
    "under_1": (0, 0),
    "1_to_4": (1, 4),
    "5_to_10": (5, 10),
    "11_to_14": (11, 14),
    "15_to_24": (15, 24),
    "25_to_34": (25, 34),
    "35_and_over": (35, 100),
}

SOURCE_AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75+",
]


# Values read from the paper figure. Rows are contact age groups from bottom to
# top in the plotted matrix; columns are participant age groups left to right.
FIGURE_VALUES_CONTACT_ROWS = {
    "0-4": [2.5, 1.0, 0.5, 0.3, 0.5, 0.7, 0.8, 0.7, 0.4, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.1],
    "5-9": [1.1, 5.5, 1.2, 0.5, 0.6, 0.7, 0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.3, 0.3, 0.2, 0.2],
    "10-14": [0.6, 1.2, 6.8, 1.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.7, 0.4, 0.2, 0.2, 0.2, 0.2, 0.1],
    "15-19": [0.3, 0.5, 1.1, 4.3, 1.0, 0.5, 0.4, 0.4, 0.6, 0.7, 0.5, 0.2, 0.2, 0.4, 0.2, 0.1],
    "20-24": [0.6, 0.6, 0.3, 1.1, 2.1, 1.1, 0.6, 0.5, 0.5, 0.6, 0.5, 0.3, 0.3, 0.4, 0.3, 0.2],
    "25-29": [0.9, 0.7, 0.4, 0.5, 1.1, 1.5, 1.0, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.3, 0.2],
    "30-34": [1.1, 1.0, 0.6, 0.4, 0.7, 1.0, 1.5, 1.0, 0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.4, 0.3],
    "35-39": [0.9, 0.9, 0.8, 0.5, 0.5, 0.6, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.5, 0.3, 0.3],
    "40-44": [0.5, 0.8, 0.9, 0.6, 0.5, 0.5, 0.6, 0.8, 1.0, 0.8, 0.5, 0.4, 0.4, 0.5, 0.5, 0.3],
    "45-49": [0.3, 0.4, 0.7, 0.8, 0.6, 0.5, 0.4, 0.5, 0.7, 1.0, 0.6, 0.4, 0.4, 0.3, 0.4, 0.4],
    "50-54": [0.4, 0.3, 0.4, 0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.6, 0.7, 0.6, 0.4, 0.3, 0.4, 0.4],
    "55-59": [0.3, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.6, 0.8, 0.6, 0.4, 0.3, 0.4],
    "60-64": [0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.5, 0.3, 0.4],
    "65-69": [0.2, 0.3, 0.2, 0.3, 0.3, 0.4, 0.5, 0.4, 0.4, 0.2, 0.2, 0.3, 0.5, 0.9, 0.5, 0.2],
    "70-74": [0.3, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.2, 0.3, 0.5, 0.6, 0.4],
    "75+": [0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.4, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.7, 1.6],
}


def parse_age_band(label: str, max_age: int = 100) -> tuple[int, int]:
    if label.endswith("+"):
        return int(label[:-1]), max_age
    low, high = label.split("-")
    return int(low), int(high)


def overlap_count(a: tuple[int, int], b: tuple[int, int]) -> int:
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    return max(0, high - low + 1)


def aggregate_to_model_groups(source_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_ranges = {band: parse_age_band(band) for band in SOURCE_AGE_GROUPS}
    rows = []
    for target, target_range in MODEL_AGE_RANGES.items():
        target_size = target_range[1] - target_range[0] + 1
        for source, source_range in source_ranges.items():
            overlap = overlap_count(target_range, source_range)
            rows.append(
                {
                    "target_age_group": target,
                    "source_age_band": source,
                    "overlap_years": overlap,
                    "target_weight": overlap / target_size,
                    "source_share": overlap / (source_range[1] - source_range[0] + 1),
                }
            )
    weights = pd.DataFrame(rows)

    out = pd.DataFrame(0.0, index=MODEL_AGE_GROUPS, columns=MODEL_AGE_GROUPS)
    for participant_age in MODEL_AGE_GROUPS:
        p_weights = weights[
            weights["target_age_group"].eq(participant_age)
            & weights["target_weight"].gt(0)
        ]
        for contact_age in MODEL_AGE_GROUPS:
            c_weights = weights[
                weights["target_age_group"].eq(contact_age)
                & weights["source_share"].gt(0)
            ]
            value = 0.0
            for _, prow in p_weights.iterrows():
                for _, crow in c_weights.iterrows():
                    value += (
                        prow["target_weight"]
                        * crow["source_share"]
                        * source_matrix.loc[prow["source_age_band"], crow["source_age_band"]]
                    )
            out.loc[participant_age, contact_age] = value
    return out, weights


def main():
    parser = argparse.ArgumentParser(description="Build a 7x7 contact matrix from Reconnect Figure 4A.")
    parser.add_argument("--outdir", default="experiments/measles_age/inputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure rows are contact age; columns are participant age. Transpose so
    # rows are participant age and columns are contact age for the simulator.
    figure_matrix_contact_by_participant = pd.DataFrame(
        FIGURE_VALUES_CONTACT_ROWS,
        index=SOURCE_AGE_GROUPS,
    ).T.reindex(index=SOURCE_AGE_GROUPS, columns=SOURCE_AGE_GROUPS)
    source_matrix = figure_matrix_contact_by_participant.T
    source_matrix.to_csv(outdir / "age_contact_matrix_reconnect_figure_16x16.csv")

    model_matrix, weights = aggregate_to_model_groups(source_matrix)
    model_matrix.to_csv(outdir / "age_contact_matrix_reconnect_figure.csv")
    (model_matrix / model_matrix.to_numpy().mean()).to_csv(
        outdir / "age_contact_matrix_reconnect_figure_normalized.csv"
    )
    weights.to_csv(outdir / "age_contact_matrix_reconnect_figure_overlap_weights.csv", index=False)

    pd.DataFrame(
        [
            {
                "source": "Goodfellow et al. 2026 PLOS Medicine Figure 4A, Total panel",
                "paper_doi": "10.1371/journal.pmed.1005038",
                "method": "Visible rounded 16x16 figure values manually encoded; transposed to participant-by-contact orientation; age-overlap weighted to seven model groups.",
                "note": "Approximate because figure values are rounded to one decimal place.",
            }
        ]
    ).to_csv(outdir / "age_contact_matrix_reconnect_figure_source_note.csv", index=False)

    print(f"Saved Reconnect figure-derived contact matrix in {outdir}")
    print(model_matrix.round(3).to_string())


if __name__ == "__main__":
    main()
