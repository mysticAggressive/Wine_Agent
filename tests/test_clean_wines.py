"""Tests for data/clean_wines.py and the committed data/wines.csv (task 1.1)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from agent.config import DATA
from agent.models import Style

WINES_CSV = DATA / "wines.csv"
RAW_CSV = DATA / "raw" / "wines_SPA.csv"
EXPECTED_COLUMNS = [
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


def _load_script():
    spec = importlib.util.spec_from_file_location("clean_wines", DATA / "clean_wines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with open(WINES_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_COLUMNS
        return list(reader)


def test_row_count(rows):
    assert len(rows) == 2024


def test_no_nulls_except_year(rows):
    # wine_type is "" when the raw type is missing; that goes with style "unknown".
    for row in rows:
        for column, value in row.items():
            if column == "year":
                continue
            if column == "wine_type":
                assert (value == "") == (row["style"] == "unknown")
                continue
            assert value != "", f"{row['wine_id']} has empty {column}"


def test_wine_ids_unique_and_sequential(rows):
    ids = [row["wine_id"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert ids == [f"w{i:04d}" for i in range(1, len(rows) + 1)]


def test_year_is_integer_or_empty(rows):
    years = [row["year"] for row in rows]
    assert all(y == "" or y.isdigit() for y in years)
    assert any(y == "" for y in years)


def test_styles_valid_and_no_rose(rows):
    styles = {row["style"] for row in rows}
    assert styles <= {s.value for s in Style}
    assert {"red", "white", "sparkling", "fortified", "unknown"} == styles


def test_value_ranges(rows):
    for row in rows:
        assert 2 <= int(row["body"]) <= 5
        assert 1 <= int(row["acidity"]) <= 3
        assert float(row["retail_price"]) > 0
        assert row["imputed"] in {"True", "False"}


def test_unique_winery_name_year(rows):
    keys = [(row["winery"], row["name"], row["year"]) for row in rows]
    assert len(set(keys)) == len(keys)


def test_script_is_reproducible(tmp_path: Path, capsys):
    script = _load_script()
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    summary = script.main(RAW_CSV, first)
    script.main(RAW_CSV, second)
    out = capsys.readouterr().out
    assert "rows in" in out and "rows out" in out
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == WINES_CSV.read_bytes()
    assert summary["rows_in"] == 7500
    assert summary["rows_out"] == 2024
    assert sum(summary["per_style"].values()) == 2024


def test_unmapped_type_raises(tmp_path: Path):
    script = _load_script()
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "winery,wine,year,rating,num_reviews,country,region,price,type,body,acidity\n"
        "A,B,2020,4.3,30,Espana,Rioja,10,Rosado,3,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Rosado"):
        script.clean(raw)


def test_wrong_columns_raise(tmp_path: Path):
    script = _load_script()
    raw = tmp_path / "raw.csv"
    raw.write_text("winery,wine\nA,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected columns"):
        script.clean(raw)


def test_dedup_and_imputation(tmp_path: Path):
    script = _load_script()
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "winery,wine,year,rating,num_reviews,country,region,price,type,body,acidity\n"
        " A ,B,2020,4.3,30,Espana,Rioja,10,Rioja Red,4,3\n"
        "A,B,2020,4.3,50,Espana,Rioja,12,Rioja Red,4,3\n"
        "A,B,2020,4.3,50,Espana,Rioja,11,Rioja Red,4,3\n"
        "A,C,N.V.,4.4,10,Espana,Rioja,20,Rioja Red,,\n"
        "A,D,,4.4,10,Espana,Rioja,20,,,\n"
        "A,E,2019,4.4,10,Espana,Rioja,0,Rioja Red,4,3\n",
        encoding="utf-8",
    )
    df, summary = script.clean(raw)
    assert summary["rows_in"] == 6
    assert summary["dropped_price"] == 1
    assert summary["dropped_duplicates"] == 2
    assert list(df["name"]) == ["B", "C", "D"]
    kept = df.iloc[0]
    assert (kept["num_reviews"], kept["retail_price"]) == (50, 11.0)
    assert list(df["imputed"]) == [False, True, True]
    assert list(df["body"]) == [4, 4, 4]
    assert list(df["style"]) == ["red", "red", "unknown"]
    assert df["year"].isna().tolist() == [False, True, True]
