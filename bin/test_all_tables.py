#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from uuid import UUID

import pandas as pd


REQUIRED_COLUMNS = {
    "Id",
    "Country",
    "StartDate",
    "EndDate",
    "Type",
    "RegionalScope",
    "Name",
}

# Columns that must exist and must not be empty per row.
REQUIRED_NON_EMPTY = {
    "Id",
    "Country",
    "StartDate",
    "Type",
    "RegionalScope",
    "Name",
}


def _csv_line(row_index: int) -> int:
    # +1 for header, +1 for 0-based index -> line number
    return row_index + 2


def _read_csv(path: Path) -> pd.DataFrame:
    # Use UTF-8 with BOM and force strings to avoid accidental type coercion.
    # Treat only empty fields as missing.
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def _check_required_columns(df: pd.DataFrame, filename: Path) -> list[str]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if not missing:
        return []
    return [f"{filename}: missing required columns: {sorted(missing)}"]


def _check_required_values(df: pd.DataFrame, filename: Path) -> list[str]:
    errors: list[str] = []
    for col in sorted(REQUIRED_NON_EMPTY):
        if col not in df.columns:
            continue
        missing = df[col].isna()
        if not missing.any():
            continue
        for idx in missing[missing].index[:10]:
            errors.append(f"{filename} (line {_csv_line(int(idx))}): missing value in column '{col}'")
        more = int(missing.sum()) - min(10, int(missing.sum()))
        if more > 0:
            errors.append(f"{filename}: {more} more missing '{col}' values not shown")
    return errors


def _check_country_column(df: pd.DataFrame, filename: Path, expected_country: str) -> list[str]:
    if "Country" not in df:
        return []
    wrong = df[~df["Country"].isna() & (df["Country"] != expected_country)]
    if wrong.empty:
        return []
    lines = ", ".join(str(_csv_line(int(i))) for i in wrong.index[:10])
    return [
        f"{filename}: Country column mismatch (expected '{expected_country}') at lines {lines}"
    ]


def _parse_dates(df: pd.DataFrame, filename: Path) -> tuple[pd.Series, pd.Series, list[str]]:
    errors: list[str] = []

    start_raw = df["StartDate"]
    end_raw = df["EndDate"]

    start_parsed = pd.to_datetime(start_raw, format="%Y-%m-%d", errors="coerce")
    end_parsed = pd.to_datetime(end_raw, format="%Y-%m-%d", errors="coerce")

    invalid_start = start_raw.notna() & start_parsed.isna()
    invalid_end = end_raw.notna() & end_parsed.isna()

    if invalid_start.any():
        bad = df[invalid_start].head(10)
        for idx, val in bad["StartDate"].items():
            errors.append(f"{filename} (line {_csv_line(int(idx))}): invalid StartDate '{val}'")

    if invalid_end.any():
        bad = df[invalid_end].head(10)
        for idx, val in bad["EndDate"].items():
            errors.append(f"{filename} (line {_csv_line(int(idx))}): invalid EndDate '{val}'")

    return start_parsed, end_parsed, errors


def _check_duration(
    start_dates: pd.Series, end_dates: pd.Series, filename: Path
) -> list[str]:
    # EndDate may be empty -> allowed. If present, must be >= StartDate.
    mask = end_dates.notna() & start_dates.notna() & (end_dates < start_dates)
    if not mask.any():
        return []

    errors: list[str] = []
    for idx in mask[mask].index[:10]:
        errors.append(
            f"{filename} (line {_csv_line(int(idx))}): EndDate < StartDate ({end_dates.loc[idx].date()} < {start_dates.loc[idx].date()})"
        )
    more = int(mask.sum()) - len(errors)
    if more > 0:
        errors.append(f"{filename}: {more} more negative durations not shown")
    return errors


def _check_sorting(start_dates: pd.Series, filename: Path) -> list[str]:
    # Only compare rows with valid StartDate values.
    errors: list[str] = []
    for i in range(len(start_dates) - 1):
        a = start_dates.iloc[i]
        b = start_dates.iloc[i + 1]
        if pd.isna(a) or pd.isna(b):
            continue
        if a > b:
            errors.append(
                f"{filename}: not sorted by StartDate: line {_csv_line(i)} ({a.date()}) > line {_csv_line(i + 1)} ({b.date()})"
            )
            if len(errors) >= 5:
                break
    return errors


def _split_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _check_subdivisions(df: pd.DataFrame, subdivisions: set[str], filename: Path) -> list[str]:
    if not subdivisions:
        return []
    if "Subdivisions" not in df.columns:
        return []

    used: set[str] = set()
    for val in df["Subdivisions"].dropna():
        used.update(_split_csv_list(val))

    unknown = used - subdivisions
    if not unknown:
        return []
    return [f"{filename}: unknown Subdivisions values: {sorted(unknown)}"]


def _check_uuids_and_global_uniqueness(
    df: pd.DataFrame, filename: Path, seen: dict[str, tuple[Path, int]]
) -> list[str]:
    if "Id" not in df.columns:
        return [f"{filename}: missing Id column"]

    errors: list[str] = []
    for idx, raw in df["Id"].items():
        line = _csv_line(int(idx))
        if pd.isna(raw) or str(raw).strip() == "":
            errors.append(f"{filename} (line {line}): missing UUID")
            continue

        try:
            normalized = str(UUID(str(raw))).lower()
        except (ValueError, AttributeError, TypeError):
            errors.append(f"{filename} (line {line}): invalid UUID '{raw}'")
            continue

        if normalized in seen:
            prev_file, prev_line = seen[normalized]
            errors.append(
                "Duplicate UUID "
                + normalized
                + ":\n"
                + f"  - {prev_file} (line {prev_line})\n"
                + f"  - {filename} (line {line})"
            )
        else:
            seen[normalized] = (filename, line)

    return errors


def _load_subdivisions(country_dir: Path) -> set[str]:
    subdivisions_csv = country_dir / "subdivisions.csv"
    if not subdivisions_csv.exists():
        return set()
    try:
        df = _read_csv(subdivisions_csv)
    except pd.errors.ParserError:
        return set()
    if "ShortName" not in df.columns:
        return set()
    return set(df["ShortName"].dropna().astype(str))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-folder", type=Path, default=Path("src"), help="Location of the data"
    )
    args = parser.parse_args()

    errors: list[str] = []
    seen_uuids: dict[str, tuple[Path, int]] = {}

    for country_dir in sorted([p for p in args.data_folder.iterdir() if p.is_dir()]):
        expected_country = country_dir.name.upper()
        subdivisions = _load_subdivisions(country_dir)

        holidays_dir = country_dir / "holidays"
        if not holidays_dir.exists():
            continue

        for holidays_file in sorted(holidays_dir.glob("*.csv")):
            try:
                df = _read_csv(holidays_file)
            except pd.errors.ParserError as error:
                errors.append(f"{holidays_file}: could not parse CSV - {error}")
                continue

            errors.extend(_check_required_columns(df, holidays_file))
            if REQUIRED_COLUMNS - set(df.columns):
                # Don’t cascade on missing columns.
                continue

            errors.extend(_check_required_values(df, holidays_file))

            errors.extend(_check_country_column(df, holidays_file, expected_country))
            errors.extend(_check_uuids_and_global_uniqueness(df, holidays_file, seen_uuids))

            start_dates, end_dates, date_errors = _parse_dates(df, holidays_file)
            errors.extend(date_errors)
            errors.extend(_check_duration(start_dates, end_dates, holidays_file))
            errors.extend(_check_sorting(start_dates, holidays_file))
            errors.extend(_check_subdivisions(df, subdivisions, holidays_file))

    if errors:
        print(f"Validation failed with {len(errors)} error(s):\n", file=sys.stderr)
        for message in errors:
            print(f"- {message}", file=sys.stderr)
        sys.exit(1)

    print("✓ All validations passed")


if __name__ == "__main__":
    main()
