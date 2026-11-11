"""Clean the raw Spanish wine dataset: data/raw/wines_SPA.csv -> data/wines.csv.

Run by hand, once: ``python data/clean_wines.py``. The output is committed. The steps
follow DEVELOPMENT_PLAN.md task 1.1 and the result is byte-for-byte reproducible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_PATH = DATA_DIR / "raw" / "wines_SPA.csv"
OUT_PATH = DATA_DIR / "wines.csv"

RAW_COLUMNS = [
    "winery",
    "wine",
    "year",
    "rating",
    "num_reviews",
    "country",
    "region",
    "price",
    "type",
    "body",
    "acidity",
]

OUT_COLUMNS = [
    "wine_id",
    "winery",
    "name",
    "year",
    "rating",
    "num_reviews",
    "region",
    "retail_price",
    "wine_type",
    "style",
    "body",
    "acidity",
    "imputed",
]

TEXT_COLUMNS = ["winery", "wine", "year", "country", "region", "type"]

STYLE_TYPES: dict[str, tuple[str, ...]] = {
    "red": (
        "Ribera Del Duero Red",
        "Rioja Red",
        "Priorat Red",
        "Red",
        "Toro Red",
        "Tempranillo",
        "Grenache",
        "Monastrell",
        "Mencia",
        "Montsant Red",
        "Syrah",
        "Cabernet Sauvignon",
    ),
    "white": ("Rioja White", "Albarino", "Verdejo", "Chardonnay", "Sauvignon Blanc"),
    "sparkling": ("Cava", "Sparkling"),
    "fortified": ("Sherry", "Pedro Ximenez"),
    "unknown": ("",),
}
TYPE_TO_STYLE: dict[str, str] = {t: style for style, types in STYLE_TYPES.items() for t in types}

EXPECTED_ROWS_OUT = 2024


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, halves up (no banker's rounding)."""
    return int(math.floor(value + 0.5))


def _impute(df: pd.DataFrame, column: str) -> pd.Series:
    """Fill missing ``column`` with the median of its type, else the global median."""
    global_median = df[column].median()
    type_median = df.groupby("type")[column].transform("median")
    filled = df[column].fillna(type_median).fillna(global_median)
    return filled.map(_round_half_up).astype(int)


def clean(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the cleaned wine table and a summary dict (rows in/out, per style, imputed)."""
    df = pd.read_csv(raw_path, dtype={"year": str})
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {raw_path}: {list(df.columns)}; expected {RAW_COLUMNS}"
        )
    rows_in = len(df)

    # 2. strip whitespace from text columns
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    # 3. year: "N.V." or blank -> missing, everything else an integer
    year = df["year"].where(~df["year"].isin(["N.V.", ""]))
    df["year"] = pd.to_numeric(year, errors="raise").astype("Int64")

    # 4. drop rows with a missing or non-positive price
    bad_price = df["price"].isna() | (df["price"] <= 0)
    dropped_price = int(bad_price.sum())
    df = df[~bad_price].copy()

    # 5. de-duplicate on (winery, wine, year): most reviews first, then lowest price
    df["_row"] = range(len(df))
    df = df.sort_values(["num_reviews", "price", "_row"], ascending=[False, True, True])
    before = len(df)
    df = df.drop_duplicates(subset=["winery", "wine", "year"], keep="first")
    dropped_duplicates = before - len(df)
    df = df.drop(columns="_row")

    # 6./7. type and style
    df["type"] = df["type"].fillna("").astype(str)
    unknown_types = sorted(set(df["type"]) - set(TYPE_TO_STYLE))
    if unknown_types:
        raise ValueError(f"Wine types with no style mapping: {unknown_types}")
    df["style"] = df["type"].map(TYPE_TO_STYLE)

    # 8. impute body and acidity
    df["imputed"] = df["body"].isna() | df["acidity"].isna()
    df["body"] = _impute(df, "body")
    df["acidity"] = _impute(df, "acidity")

    # 9. rename and tidy
    df["price"] = df["price"].round(2)
    df = df.rename(columns={"price": "retail_price", "wine": "name", "type": "wine_type"})
    df = df.drop(columns="country")
    for col in ("winery", "name", "region"):
        df[col] = df[col].fillna("").astype(str)
    df["num_reviews"] = df["num_reviews"].astype(int)

    # 10. stable order and ids
    df = df.sort_values(["winery", "name", "year"], na_position="last", kind="mergesort")
    df = df.reset_index(drop=True)
    df["wine_id"] = [f"w{i:04d}" for i in range(1, len(df) + 1)]
    df = df[OUT_COLUMNS]

    summary: dict[str, object] = {
        "rows_in": rows_in,
        "dropped_price": dropped_price,
        "dropped_duplicates": dropped_duplicates,
        "rows_out": len(df),
        "per_style": {s: int((df["style"] == s).sum()) for s in STYLE_TYPES},
        "imputed": int(df["imputed"].sum()),
    }
    return df, summary


def write(df: pd.DataFrame, out_path: Path = OUT_PATH) -> None:
    """Write the cleaned table as UTF-8 CSV with Unix newlines."""
    df.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")


def main(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> dict[str, object]:
    """Clean the raw file, write the output and print the summary."""
    df, summary = clean(raw_path)
    write(df, out_path)
    print(f"rows in:             {summary['rows_in']}")
    print(f"dropped (bad price): {summary['dropped_price']}")
    print(f"dropped (duplicate): {summary['dropped_duplicates']}")
    print(f"rows out:            {summary['rows_out']}")
    for style, count in summary["per_style"].items():  # type: ignore[union-attr]
        print(f"  style {style:<10} {count}")
    print(f"rows imputed:        {summary['imputed']}")
    print(f"wrote {out_path}")
    if summary["rows_out"] != EXPECTED_ROWS_OUT:
        print(
            f"WARNING: expected {EXPECTED_ROWS_OUT} rows, got {summary['rows_out']}",
            file=sys.stderr,
        )
    return summary


if __name__ == "__main__":
    main()
