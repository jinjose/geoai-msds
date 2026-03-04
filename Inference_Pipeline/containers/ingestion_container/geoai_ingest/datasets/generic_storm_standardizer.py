#!/usr/bin/env python3
"""
storm_standardizer.py

Standardize storm parquet files into a canonical schema used by feature builder.

Canonical output columns (minimum):
- county (lowercase)
- date (timestamp, day-level)
- year (int)
- wind_mph (float)
- event_count (int)
- severe_gust_58 (int 0/1)

Optional:
- county_fips (string) if mapping provided

Usage examples:
  # Standardize a single file (auto output name):
  python storm_standardizer.py --in storm_daily_2025.parquet --out storm_daily_2025_standardized.parquet

  # Standardize and split into one file per year:
  python storm_standardizer.py --in storm_all_years.parquet --out-dir out/ --split-by-year

  # Standardize using county->fips mapping:
  python storm_standardizer.py --in storm_daily_2025.parquet --out storm_2025_std.parquet \
      --county-map counties_iowa.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


CANON_COLS = ["county", "date", "year", "wind_mph", "event_count", "severe_gust_58"]
DEFAULT_GUST_CUTOFF = 58.0


@dataclass
class StandardizeOptions:
    gust_cutoff_mph: float = DEFAULT_GUST_CUTOFF
    county_map_path: Optional[Path] = None
    keep_extra_cols: bool = False
    # If True, require that after mapping we have county_fips present
    require_county_fips: bool = False


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path} (use .parquet or .csv)")


def _normalize_county(s: pd.Series) -> pd.Series:
    # Normalize county text (handles "Black Hawk" -> "black hawk")
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )


def _coerce_date(df: pd.DataFrame) -> pd.Series:
    """
    Determine date column:
      - prefer 'date' if present
      - else use 'datetime' if present
    Normalize to day-level timestamp.
    """
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError("Input must contain either 'date' or 'datetime' column.")
    return dt.dt.floor("D")


def _coerce_wind(df: pd.DataFrame) -> pd.Series:
    if "wind_mph" not in df.columns:
        raise ValueError("Input must contain 'wind_mph' column.")
    return pd.to_numeric(df["wind_mph"], errors="coerce")


def _ensure_event_count(df: pd.DataFrame) -> pd.Series:
    # If already present, keep it; else set 1 per record
    if "event_count" in df.columns:
        return pd.to_numeric(df["event_count"], errors="coerce").fillna(1).astype(int)
    return pd.Series([1] * len(df), index=df.index, dtype="int64")


def _ensure_year(df: pd.DataFrame, date_col: pd.Series) -> pd.Series:
    if "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        # If year missing for some rows, derive from date
        y = y.fillna(date_col.dt.year)
        return y.astype(int)
    return date_col.dt.year.astype(int)


def _ensure_severe_flag(wind: pd.Series, cutoff: float, df: pd.DataFrame) -> pd.Series:
    if "severe_gust_58" in df.columns:
        # keep existing but coerce to 0/1
        s = pd.to_numeric(df["severe_gust_58"], errors="coerce").fillna(0)
        return (s > 0).astype(int)
    # compute using cutoff (default 58)
    return (wind >= cutoff).fillna(False).astype(int)


def _load_county_map(path: Path) -> pd.DataFrame:
    """
    County mapping file must include:
      - county (string)
      - county_fips (string)
    Optional:
      - state_fips (string/int)
    """
    m = _read_any(path)
    # normalize required columns
    if "county" not in m.columns or "county_fips" not in m.columns:
        raise ValueError("County map must contain columns: 'county' and 'county_fips'.")
    m = m.copy()
    m["county"] = _normalize_county(m["county"])
    m["county_fips"] = m["county_fips"].astype(str).str.zfill(3)  # Iowa counties are 3-digit
    # keep first unique mapping
    m = m.drop_duplicates(subset=["county"], keep="first")
    return m[["county", "county_fips"]]


def standardize_storm_df(df_in: pd.DataFrame, opts: StandardizeOptions) -> pd.DataFrame:
    df = df_in.copy()

    if "county" not in df.columns:
        raise ValueError("Input must contain 'county' column.")

    df["county"] = _normalize_county(df["county"])
    df["date"] = _coerce_date(df)

    wind = _coerce_wind(df)
    df["wind_mph"] = wind

    df["event_count"] = _ensure_event_count(df)
    df["year"] = _ensure_year(df, df["date"])
    df["severe_gust_58"] = _ensure_severe_flag(wind, opts.gust_cutoff_mph, df)

    # Optional mapping to county_fips
    if opts.county_map_path:
        m = _load_county_map(opts.county_map_path)
        df = df.merge(m, on="county", how="left")
        if opts.require_county_fips and df["county_fips"].isna().any():
            missing = df.loc[df["county_fips"].isna(), "county"].dropna().unique()[:20]
            raise ValueError(
                f"Missing county_fips mapping for {df['county_fips'].isna().sum()} rows. "
                f"Examples: {missing}"
            )

    # Keep only canonical columns unless asked to keep extras
    if opts.keep_extra_cols:
        # Ensure canonical columns are first
        front = [c for c in CANON_COLS if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]
    else:
        cols = CANON_COLS.copy()
        if "county_fips" in df.columns:
            cols.insert(1, "county_fips")
        df = df[cols]

    # Drop rows with invalid dates or wind if you want strictness; here we keep but you can tighten.
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df


def write_output(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def split_and_write_by_year(df: pd.DataFrame, out_dir: Path, prefix: str = "storm_daily") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, g in df.groupby("year"):
        out_path = out_dir / f"{prefix}_{int(year)}.parquet"
        g = g.sort_values(["county", "date"])
        g.to_parquet(out_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize storm dataset schema.")
    p.add_argument("--in", dest="in_path", required=True, help="Input parquet/csv path")
    p.add_argument("--out", dest="out_path", default=None, help="Output parquet path")
    p.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory (used with --split-by-year)")
    p.add_argument("--split-by-year", action="store_true", help="Write one parquet per year")
    p.add_argument("--gust-cutoff", type=float, default=DEFAULT_GUST_CUTOFF, help="Severe gust cutoff mph (default 58)")
    p.add_argument("--county-map", type=str, default=None, help="Parquet/CSV with columns: county, county_fips")
    p.add_argument("--require-county-fips", action="store_true", help="Fail if any county_fips missing after mapping")
    p.add_argument("--keep-extra-cols", action="store_true", help="Keep extra columns beyond canonical schema")
    p.add_argument("--prefix", default="storm_daily", help="Filename prefix for --split-by-year outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)

    opts = StandardizeOptions(
        gust_cutoff_mph=args.gust_cutoff,
        county_map_path=Path(args.county_map) if args.county_map else None,
        keep_extra_cols=args.keep_extra_cols,
        require_county_fips=args.require_county_fips,
    )

    df_in = _read_any(in_path)
    df_std = standardize_storm_df(df_in, opts)

    if args.split_by_year:
        if not args.out_dir:
            raise ValueError("--out-dir is required when using --split-by-year")
        split_and_write_by_year(df_std, Path(args.out_dir), prefix=args.prefix)
        print(f"Wrote yearly files to: {args.out_dir}")
        return

    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_standardized.parquet")
    write_output(df_std, out_path)
    print(f"Wrote standardized file: {out_path}")


if __name__ == "__main__":
    main()