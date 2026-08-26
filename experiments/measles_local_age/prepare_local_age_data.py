"""Prepare region-age measles inputs for a structured simulator.

This script combines:

- England weekly measles timing from ``data/ukhsa-measles_weekly_cases.csv``;
- annual/period region-age case totals from
  ``data/england_measles_cases_by_age_and_region_2023_2026.csv``;
- ONS 2024 mid-year population estimates by region and single year of age;
- COVER MMR coverage at 24 months and 5 years.

The output is a set of CSVs that can be used by ``fit_local_age_blocks.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT / "experiments"))

from validate_measles import load_england_weekly_cases


REGION_COLUMNS = [
    "east_midlands",
    "east_of_england",
    "london",
    "north_east",
    "north_west",
    "south_east",
    "south_west",
    "west_midlands",
    "yorkshire_and_humber",
]

REGION_NAME_MAP = {
    "east_midlands": "East Midlands",
    "east_of_england": "East of England",
    "london": "London",
    "north_east": "North East",
    "north_west": "North West",
    "south_east": "South East",
    "south_west": "South West",
    "west_midlands": "West Midlands",
    "yorkshire_and_humber": "Yorkshire and The Humber",
}

AGE_GROUPS = [
    "under_1",
    "1_to_4",
    "5_to_10",
    "11_to_14",
    "15_to_24",
    "25_to_34",
    "35_and_over",
]

AGE_BINS = {
    "under_1": [0],
    "1_to_4": list(range(1, 5)),
    "5_to_10": list(range(5, 11)),
    "11_to_14": list(range(11, 15)),
    "15_to_24": list(range(15, 25)),
    "25_to_34": list(range(25, 35)),
    "35_and_over": list(range(35, 91)),
}


def safe_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    value = str(value).strip().replace(",", "")
    if value in {"", "[z]", "[x]", "[c]", "No data"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def load_xlsx_shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for si in root.findall("main:si", XLSX_NS):
        text = "".join(t.text or "" for t in si.findall(".//main:t", XLSX_NS))
        values.append(text)
    return values


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read raw XLSX sheet values without openpyxl."""
    with ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", XLSX_NS)
        }
        sheet_target = None
        for sheet in workbook.findall(".//main:sheet", XLSX_NS):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib.get(f"{{{XLSX_NS['rel']}}}id")
                sheet_target = rel_map[rid]
                break
        if sheet_target is None:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}")
        if not sheet_target.startswith("xl/"):
            sheet_target = "xl/" + sheet_target.lstrip("/")

        shared = load_xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read(sheet_target))
        rows = []
        for row in root.findall(".//main:row", XLSX_NS):
            values = {}
            for cell in row.findall("main:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                idx = column_index(ref)
                ctype = cell.attrib.get("t")
                v = cell.find("main:v", XLSX_NS)
                if ctype == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//main:t", XLSX_NS))
                    values[idx] = text
                elif v is None:
                    values[idx] = ""
                elif ctype == "s":
                    values[idx] = shared[int(v.text)]
                else:
                    values[idx] = safe_float(v.text)
            if values:
                width = max(values) + 1
                rows.append([values.get(i, "") for i in range(width)])
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(rows)


def load_region_age_population(path: Path) -> pd.DataFrame:
    """Load ONS MYE2 region rows and aggregate single-year ages."""
    raw0 = read_xlsx_sheet(path, "MYE2 - Persons")
    header = raw0.iloc[7].tolist()
    raw = raw0.iloc[8:].copy()
    raw.columns = header
    raw = raw[raw["Geography"].astype(str).str.lower().eq("region")].copy()
    raw["region"] = raw["Name"].astype(str).str.title()
    # Keep the ONS spelling aligned with the case table spelling.
    raw["region"] = raw["region"].replace(
        {
            "East": "East of England",
            "Yorkshire And The Humber": "Yorkshire and The Humber",
        }
    )

    rows = []
    for _, row in raw.iterrows():
        for age_group, ages in AGE_BINS.items():
            total = 0.0
            for age in ages:
                col = str(age)
                if col in row.index:
                    total += safe_float(row[col])
            rows.append(
                {
                    "region": row["region"],
                    "age_group": age_group,
                    "population": total,
                    "source": "ONS MYE2 - Persons, mid-2024",
                }
            )
    return pd.DataFrame(rows)


ODS_NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}


def ods_cell_text(cell) -> str:
    values = []
    for elem in cell.iter():
        if elem.tag.endswith("}p") or elem.tag.endswith("}span"):
            if elem.text:
                values.append(elem.text)
    return " ".join(values).strip()


def read_ods_table(path: Path, table_name: str) -> pd.DataFrame:
    """Read a simple ODS table without requiring odfpy."""
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("content.xml"))
    wanted = None
    for table in root.findall(".//table:table", ODS_NS):
        if table.attrib.get(f"{{{ODS_NS['table']}}}name") == table_name:
            wanted = table
            break
    if wanted is None:
        raise ValueError(f"Table {table_name!r} not found in {path}")

    rows = []
    for row in wanted.findall("table:table-row", ODS_NS):
        values = []
        for cell in row.findall("table:table-cell", ODS_NS):
            repeat = int(cell.attrib.get(f"{{{ODS_NS['table']}}}number-columns-repeated", "1"))
            text = ods_cell_text(cell)
            values.extend([text] * min(repeat, 20))
        if any(v != "" for v in values):
            rows.append(values)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows)


def table_from_header(raw: pd.DataFrame, first_column: str = "Code") -> pd.DataFrame:
    header_index = None
    for idx, row in raw.iterrows():
        if str(row.iloc[0]).strip() == first_column:
            header_index = idx
            break
    if header_index is None:
        raise ValueError(f"Could not find header row starting with {first_column!r}")
    header = raw.iloc[header_index].tolist()
    data = raw.iloc[header_index + 1 :].copy()
    data.columns = header
    data = data.loc[:, [c for c in data.columns if str(c).strip() != ""]]
    return data.reset_index(drop=True)


def load_region_mmr_coverage(cover_path: Path) -> pd.DataFrame:
    """Load regional MMR1/MMR2 coverage from the COVER ODS."""
    t5 = table_from_header(read_ods_table(cover_path, "T5a_UTLA24m"))
    t6 = table_from_header(read_ods_table(cover_path, "T6a_UTLA5y"))

    region_or_england_code_re = re.compile(r"^(E120\d+|E92000001)$")

    rows = []
    for _, row in t5.iterrows():
        code = str(row.get("Code", "")).strip()
        if not region_or_england_code_re.match(code):
            continue
        rows.append(
            {
                "region": str(row.get("Region name", "")).strip(),
                "mmr1_24m": safe_float(row.get("Coverage at 24 months MMR1 (%)")) / 100.0,
            }
        )
    cover = pd.DataFrame(rows).dropna()

    rows = []
    for _, row in t6.iterrows():
        code = str(row.get("Code", "")).strip()
        if not region_or_england_code_re.match(code):
            continue
        rows.append(
            {
                "region": str(row.get("Region name", "")).strip(),
                "mmr1_5y": safe_float(row.get("Coverage at 5 years MMR1 (%)")) / 100.0,
                "mmr2_5y": safe_float(row.get("Coverage at 5 years MMR2 (%)")) / 100.0,
            }
        )
    cover5 = pd.DataFrame(rows).dropna(subset=["region"])
    cover = cover.merge(cover5, on="region", how="outer")
    cover["region"] = cover["region"].replace(
        {
            "East": "East of England",
            "Yorkshire and Humber": "Yorkshire and The Humber",
        }
    )
    return cover


def protection_by_region_age(population: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Create age-specific protected fractions from COVER + fallbacks."""
    df = population.merge(coverage, on="region", how="left")
    england = coverage[coverage["region"].eq("England")]
    defaults = {
        "mmr1_24m": float(england["mmr1_24m"].iloc[0]) if not england.empty else 0.889,
        "mmr1_5y": float(england["mmr1_5y"].iloc[0]) if not england.empty else 0.918,
        "mmr2_5y": float(england["mmr2_5y"].iloc[0]) if not england.empty else 0.837,
    }
    for col, value in defaults.items():
        df[col] = df[col].fillna(value)

    def protection(row):
        age = row["age_group"]
        if age == "under_1":
            return 0.0
        if age == "1_to_4":
            return row["mmr1_24m"]
        if age == "5_to_10":
            return row["mmr2_5y"]
        # COVER does not measure protection in these older cohorts. Use the
        # England MMR1-at-five value as a neutral national proxy rather than
        # projecting current regional childhood differences onto adults.
        return defaults["mmr1_5y"]

    df["protected_fraction"] = df.apply(protection, axis=1).clip(0, 0.99)
    return df[
        [
            "region",
            "age_group",
            "population",
            "protected_fraction",
            "mmr1_24m",
            "mmr1_5y",
            "mmr2_5y",
            "source",
        ]
    ]


def add_region_risk_multiplier(pop_protection: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Estimate a fixed regional risk multiplier from annual burden per capita.

    This is deliberately simple: regions with higher annual measles burden per
    person get a multiplier above 1, and lower-burden regions get a multiplier
    below 1. It gives the simulator some regional heterogeneity while keeping
    temporal targets transparent.
    """
    burden = annual.groupby("region", as_index=False)["annual_cases"].sum()
    pop = pop_protection.groupby("region", as_index=False)["population"].sum()
    risk = burden.merge(pop, on="region", how="inner")
    risk["cases_per_person"] = risk["annual_cases"] / risk["population"].clip(lower=1.0)
    national_rate = risk["annual_cases"].sum() / risk["population"].sum()
    risk["region_risk_multiplier"] = (
        risk["cases_per_person"] / max(national_rate, 1e-12)
    ).clip(0.25, 4.0)
    return pop_protection.drop(columns=["region_risk_multiplier"], errors="ignore").merge(
        risk[["region", "region_risk_multiplier"]],
        on="region",
        how="left",
    )


def load_annual_region_age_cases(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path, parse_dates=["period_start", "period_end"])
    rows = []
    for _, row in wide.iterrows():
        for region_col in REGION_COLUMNS:
            rows.append(
                {
                    "year": int(row["year"]),
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                    "complete_year": bool(row["complete_year"]),
                    "region": REGION_NAME_MAP[region_col],
                    "age_group": row["age_group"],
                    "annual_cases": float(row[region_col]),
                    "england_age_cases": float(row["england"]),
                }
            )
    return pd.DataFrame(rows)


def build_synthetic_weekly_targets(weekly_cases: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_cases.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["year"] = weekly["date"].dt.year

    outputs = []
    for (year, region, age_group), group in annual.groupby(["year", "region", "age_group"]):
        start = group["period_start"].iloc[0]
        end = group["period_end"].iloc[0]
        annual_cases = float(group["annual_cases"].iloc[0])
        weeks = weekly[(weekly["date"] >= start) & (weekly["date"] <= end)].copy()
        total_england_weekly = float(weeks["observed_cases"].sum())
        if total_england_weekly <= 0:
            weeks["synthetic_cases"] = 0.0
        else:
            weeks["synthetic_cases"] = weeks["observed_cases"] * annual_cases / total_england_weekly
        weeks["region"] = region
        weeks["age_group"] = age_group
        weeks["annual_region_age_cases"] = annual_cases
        outputs.append(weeks[["date", "year", "region", "age_group", "synthetic_cases", "annual_region_age_cases"]])
    return pd.concat(outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare region-age measles inputs.")
    parser.add_argument("--weekly-cases", default="data/ukhsa-measles_weekly_cases.csv")
    parser.add_argument("--annual-cases", default="data/england_measles_cases_by_age_and_region_2023_2026.csv")
    parser.add_argument("--population", default="/Users/zosiawilk/Downloads/mye24tablesuk.xlsx")
    parser.add_argument("--coverage", default="/Users/zosiawilk/Downloads/cover-anual-data-tables-2024-to-2025.ods")
    parser.add_argument("--outdir", default="experiments/measles_local_age/inputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    weekly = load_england_weekly_cases(args.weekly_cases)
    annual = load_annual_region_age_cases(Path(args.annual_cases))
    population = load_region_age_population(Path(args.population))
    coverage = load_region_mmr_coverage(Path(args.coverage))
    pop_protection = protection_by_region_age(population, coverage)
    pop_protection = add_region_risk_multiplier(pop_protection, annual)
    targets = build_synthetic_weekly_targets(weekly, annual)

    annual.to_csv(outdir / "annual_region_age_cases.csv", index=False)
    population.to_csv(outdir / "region_age_population.csv", index=False)
    coverage.to_csv(outdir / "region_mmr_coverage.csv", index=False)
    pop_protection.to_csv(outdir / "region_age_population_protection.csv", index=False)
    targets.to_csv(outdir / "synthetic_region_age_weekly_cases.csv", index=False)

    print(f"Saved prepared inputs in {outdir}")
    print(f"  regions: {pop_protection['region'].nunique()}")
    print(f"  age groups: {pop_protection['age_group'].nunique()}")
    print(f"  synthetic target rows: {len(targets):,}")


if __name__ == "__main__":
    main()
