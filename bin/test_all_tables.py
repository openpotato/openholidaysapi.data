#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def _check_duration(df: pd.DataFrame, filename: Path) -> None:
    """Parses StartDate and EndDate as YYYY-MM-DD and checks that EndDate if after the
    StartDate"""

    df.StartDate = pd.to_datetime(df.StartDate, format="%Y-%m-%d")
    df.EndDate = pd.to_datetime(df.EndDate, format="%Y-%m-%d")
    positive_duration_mask = df.EndDate.isna() | ((df.EndDate >= df.StartDate))
    if not positive_duration_mask.all():
        raise ValueError(
            f"Holidays with negative duration in '{filename}':\n"
            f"{df[~positive_duration_mask]}"
        )


def _check_subdivisions(
    df: pd.DataFrame, subdivisions: set[str], filename: Path
) -> None:
    """Checks that the subdivisions in df are also present in subdivisions.csv"""
    if "Subdivisions" in df:
        unknown_subdivisions = set(
            df.Subdivisions.dropna().map(lambda x: x.split(",")).explode()
        ) - set(subdivisions)
        if unknown_subdivisions:
            raise ValueError(
                f"Unknown subdivisions in {filename}: {unknown_subdivisions}. "
                f"Known are {subdivisions}"
            )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-folder", type=Path, default=Path("src"), help="Location of the data"
    )
    args = parser.parse_args()

    for country_dir in sorted(args.data_folder.iterdir()):
        if not country_dir.is_dir():
            continue

        try:
            df_subdivisions = pd.read_csv(country_dir / "subdivisions.csv", sep=";")
            subdivisions = set(df_subdivisions.ShortName)
        except FileNotFoundError:
            subdivisions = {}

        for holidays_file in (country_dir / "holidays").iterdir():
            try:
                df = pd.read_csv(holidays_file, sep=";")
            except pd.errors.ParserError as error:
                raise ValueError(f"Could not parse '{holidays_file}'") from error
            _check_subdivisions(df, subdivisions, holidays_file)
            _check_duration(df, holidays_file)


main()
