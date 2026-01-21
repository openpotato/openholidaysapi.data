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


def _check_name_format(df: pd.DataFrame, filename: Path) -> list[str]:
    """Check that each part of the Name field starts with a language code (two uppercase letters) followed by a space."""
    if "Name" not in df.columns:
        return []

    errors: list[str] = []
    for idx, name in df["Name"].items():
        if pd.isna(name):
            continue

        parts = _split_csv_list(str(name))
        for part in parts:
            # Each part must start with a language code (2 uppercase letters) followed by a space
            if len(part) < 3:
                errors.append(
                    f"{filename} (line {_csv_line(int(idx))}): Name part '{part}' is too short (must be language code + space + text). Hint: Use %2C instead of comma within text."
                )
                continue

            # Check first 2 characters are uppercase letters (language code)
            if not (part[0].isupper() and part[0].isalpha() and
                    part[1].isupper() and part[1].isalpha()):
                errors.append(
                    f"{filename} (line {_csv_line(int(idx))}): Name part '{part}' must start with a language code (two uppercase letters). Hint: If this text contains a comma, use %2C instead."
                )
                continue

            # Check that the 3rd character is a space
            if part[2] != ' ':
                errors.append(
                    f"{filename} (line {_csv_line(int(idx))}): Name part '{part}' must have a space after the language code"
                )

    return errors


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
    total_files = 0
    files_with_errors = 0
    errors_by_file: dict[Path, list[str]] = {}

    for country_dir in sorted([p for p in args.data_folder.iterdir() if p.is_dir()]):
        expected_country = country_dir.name.upper()
        subdivisions = _load_subdivisions(country_dir)

        holidays_dir = country_dir / "holidays"
        if not holidays_dir.exists():
            continue

        for holidays_file in sorted(holidays_dir.glob("*.csv")):
            total_files += 1
            file_errors: list[str] = []

            try:
                df = _read_csv(holidays_file)
            except pd.errors.ParserError as error:
                file_errors.append(f"{holidays_file}: could not parse CSV - {error}")
                errors.extend(file_errors)
                files_with_errors += 1
                continue

            file_errors.extend(_check_required_columns(df, holidays_file))
            if REQUIRED_COLUMNS - set(df.columns):
                # Don't cascade on missing columns.
                errors.extend(file_errors)
                files_with_errors += 1
                continue

            file_errors.extend(_check_required_values(df, holidays_file))

            file_errors.extend(_check_country_column(df, holidays_file, expected_country))
            file_errors.extend(_check_uuids_and_global_uniqueness(df, holidays_file, seen_uuids))

            start_dates, end_dates, date_errors = _parse_dates(df, holidays_file)
            file_errors.extend(date_errors)
            file_errors.extend(_check_duration(start_dates, end_dates, holidays_file))
            file_errors.extend(_check_sorting(start_dates, holidays_file))
            file_errors.extend(_check_subdivisions(df, subdivisions, holidays_file))
            file_errors.extend(_check_name_format(df, holidays_file))

            if file_errors:
                files_with_errors += 1
                errors.extend(file_errors)
                errors_by_file[holidays_file] = file_errors

    if errors:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"VALIDATION FAILED", file=sys.stderr)
        print(f"{'=' * 70}\n", file=sys.stderr)

        # Group errors by file for better readability
        for file_path, file_errors in errors_by_file.items():
            print(f"\n{file_path}: {len(file_errors)} error(s)", file=sys.stderr)
            for message in file_errors[:10]:  # Show first 10 errors per file
                # Remove redundant file path from message
                clean_message = message.replace(f"{file_path} ", "")
                print(f"  • {clean_message}", file=sys.stderr)
            if len(file_errors) > 10:
                print(f"  ... and {len(file_errors) - 10} more errors", file=sys.stderr)

        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"Summary: {files_with_errors}/{total_files} files with errors ({len(errors)} total errors)", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"✓ All validations passed!")
    print(f"{'=' * 70}")
    print(f"Checked {total_files} files successfully.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
