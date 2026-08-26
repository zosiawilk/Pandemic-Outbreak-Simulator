# Age-contact matrix input

The final outbreak-probability model uses
`inputs/age_contact_matrix_reconnect_figure_normalized.csv` as its seven-group
age-contact matrix.

`build_reconnect_figure_matrix.py` reconstructs this matrix from
the visible values in Goodfellow et al. (2026), Figure 4A, “Total” panel. The
script takes the 16-by-16 values, transposes them into the simulator's
participant-by-contact orientation, aggregates them to the seven model age
groups using age-overlap weights, and normalises the result to mean one.

Run from the repository root with:

```bash
python experiments/measles_age/build_reconnect_figure_matrix.py
```

The displayed paper values are rounded to one decimal
place, so this is an approximation rather than a full-precision
supplementary-table extraction (BETTER CONTACT MATRIX AS A FUTURE WORK).

Source: Goodfellow L, Quilty BJ, van Zandvoort K, and Edmunds WJ (2026),
*PLOS Medicine* 23(5), e1005038,
https://doi.org/10.1371/journal.pmed.1005038.
