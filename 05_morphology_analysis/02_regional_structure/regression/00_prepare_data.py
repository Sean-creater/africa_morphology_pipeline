"""
00_prepare_data.py
------------------
Merge city-level and country-level covariates with Leiden community labels
from the community detection pipeline.

No category merging or recoding is performed beyond what is documented
below.  All original Köppen climate classes and colonial legacy categories
are retained as-is.  Dummy encoding is deferred to the modelling scripts
so that reference-category choices are explicit and auditable there.

Inputs
------
input/city.xlsx
    City-level: country, city, latitude, longitude, grid_cell_count,
    climate_zone(dominant), population_2020 (Africapolis 2020; retained
    only as a sensitivity comparison, see below)
input/Country.xlsx
    Country-level: Country, Geographic_Region,
    GDP_per_Capita(2020 dollar, source: World Bank),
    Colonial History (source: Our World in Data Colonizers Dataset)
../community_detection/outputs/city_communities.parquet
    City-level Leiden community labels (1–10).
../../04_center_periphery/core/outputs/tables/city_centres.parquet
    City-level total_pop2022: per-city sum of the area-weighted WorldPop
    2022 population grid (see 04_center_periphery/core/01_define_centres.py).
    This is the primary population variable (see below).

Outputs
-------
outputs/tables/analysis_input.csv
    One row per city.  Contains:
      city, country, community (1–10),
      latitude, longitude, n_cells,
      climate_zone  (original Köppen code, 13 classes),
      colonial      (original 6-class encoding; see encode_colonial()),
      log_gdp       (log GDP per capita, 2020 USD, World Bank),
      log_pop_worldpop_2022, log_pop_worldpop_2022_std
                    (primary population variable — see below),
      log_pop_africapolis_2020
                    (sensitivity-only; Africapolis 2020, retained for
                    comparison, NOT used in the primary MNL model),
      n_population_cells, n_valid_population_cells, population_coverage
                    (grid-cell coverage of the Pop2022 field this city's
                    population_worldpop_2022 was summed from — see the
                    exclusion rule below),
      lat_c, lon_c  (mean-centred, SD-scaled latitude/longitude),
      lat2, lon2, lat_lon  (second-order spatial polynomial terms).
outputs/tables/covariate_summary.csv
    Counts and country coverage for climate_zone and colonial,
    and community × covariate cross-tabulations.  Used to assess
    separation risk before modelling.
outputs/tables/population_definition_comparison.csv
    Per-city comparison of population_worldpop_2022 vs
    population_africapolis_2020 (both raw counts), for the sensitivity
    analysis described below.

Source data notes
-----------------
Country.xlsx uses the same country-name spellings as city.xlsx and
../../../02_morphology_metrics/'s country folder structure.

Population definition
----------------------
City population is calculated by aggregating WorldPop 2022 population
counts within each Africapolis urban agglomeration (population_worldpop_2022,
from 04_center_periphery/core/01_define_centres.py's per-city sum of the
area-weighted WorldPop 2022 grid). This is the primary population variable
used throughout the analysis (MNL regression, urban-hierarchy scaling, and
the center-periphery radial analyses all now share this definition).

Exception: Ethiopia's WorldPop 2022 layer does not exist; its cities use
WorldPop 2020 instead (see 04_center_periphery/core/01_define_centres.py
docstring). This is a data-availability constraint, not an inconsistency
in method.

Cities with zero, missing, or implausibly low-coverage WorldPop 2022
population are flagged and EXCLUDED from the model sample rather than
imputed (no clip(lower=1)-style silent fix) — see
population_worldpop_2022_valid in the missing-value report below.

Africapolis 2020 population (population_2020 in city.xlsx) is retained
only as a sensitivity check: population_definition_comparison.csv reports
the Pearson/Spearman correlation between the two definitions, the
log-ratio distribution, and the cities with the largest divergence.

Colonial history encoding
-------------------------
Source: Our World in Data Colonizers Dataset (as recorded in Country.xlsx).
Multi-coloniser entries are semicolon-separated; the LAST entry is used as
the primary administrative legacy (reflects the final colonial period before
independence).  Countries with no formal colonisation are coded 'Independent'.
Unmatched entries are coded 'Other European' (Italy, Germany, Spain, etc.).

  British        – United Kingdom administration
  French         – France administration
  Belgian        – Belgium administration (DRC, Rwanda, Burundi)
  Portuguese     – Portugal administration
  Independent    – No formal colonisation recorded
                   NOTE: in this dataset this category consists entirely of
                   Ethiopian cities; coefficients reflect Ethiopian urban
                   characteristics, not a generalised "never-colonised" effect.
  Other European – Italy, Germany, Spain, and other minor colonial powers

The Independent category is retained without merging.  Its interpretive
limits are documented in covariate_summary.csv (country coverage) and
must be acknowledged in any publication.

Usage
-----
    python 00_prepare_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
INPUT_DIR   = ROOT / "input"
OUT_TABLES  = ROOT / "outputs" / "tables"
CD_OUT      = ROOT.parent / "community_detection" / "outputs"
CENTRES_PQ  = (ROOT.parent.parent / "04_center_periphery" / "core"
               / "outputs" / "tables" / "city_centres.parquet")

# Minimum fraction of a city's grid cells that must have a valid (finite,
# positive) Pop2022 value for that city's total_pop2022 to be trusted as a
# population estimate. 7900 cities have population_coverage == 1.0 in the
# vast majority of cases (median and 75th percentile are both exactly
# 1.0); only 67 cities (0.85%) fall below 0.95, so this threshold excludes
# a small, genuinely low-coverage tail rather than a systematic chunk of
# the sample.
MIN_POPULATION_COVERAGE = 0.95

COLONIAL_MAP = {
    "United Kingdom": "British",
    "France":         "French",
    "Portugal":       "Portuguese",
    "Belgium":        "Belgian",
}

# Country name harmonisation: city.xlsx spelling → Country.xlsx spelling.
# Populated only if the two source files' country-name spellings diverge;
# no recoding of country identity is implied.
COUNTRY_NAME_MAP: dict[str, str] = {}


# ---------------------------------------------------------------------------
def encode_colonial(raw: str) -> str:
    """
    Assign a primary colonial administrative legacy category.

    Rule: for semicolon-separated multi-coloniser strings, use the LAST
    entry (rightmost), which typically reflects the final administrative
    period before independence.  Countries with no formal colonisation are
    coded 'Independent'.  All remaining cases (Italy, Germany, Spain, etc.)
    are coded 'Other European'.
    """
    if pd.isna(raw):
        return "Unknown"
    raw = str(raw).strip()
    if "independent" in raw.lower() and "american" not in raw.lower():
        return "Independent"
    parts = [p.strip() for p in raw.split(";")]
    last  = parts[-1]
    for keyword, label in COLONIAL_MAP.items():
        if keyword in last:
            return label
    return "Other European"


def standardise_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add centred/scaled spatial terms (computed on the analysis sample):
      lat_c, lon_c  : mean-centred, unit-SD scaled
      lat2, lon2    : squared terms
      lat_lon       : interaction
    Centering and scaling parameters are logged for reproducibility.
    """
    df = df.copy()
    lat_mean, lat_sd = df["latitude"].mean(),  df["latitude"].std()
    lon_mean, lon_sd = df["longitude"].mean(), df["longitude"].std()
    logging.info("Spatial standardisation: lat mean=%.4f sd=%.4f; "
                 "lon mean=%.4f sd=%.4f", lat_mean, lat_sd, lon_mean, lon_sd)
    df["lat_c"]   = (df["latitude"]  - lat_mean) / lat_sd
    df["lon_c"]   = (df["longitude"] - lon_mean) / lon_sd
    df["lat2"]    = df["lat_c"] ** 2
    df["lon2"]    = df["lon_c"] ** 2
    df["lat_lon"] = df["lat_c"] * df["lon_c"]
    return df


def write_population_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Sensitivity comparison of the two population definitions, on the
    common sample where both are valid: Pearson/Spearman correlation
    (log scale), the log-ratio distribution, and the cities with the
    largest divergence. Written for the model sample (WorldPop 2022
    valid); Africapolis-invalid cities are NaN in the ratio columns.
    """
    from scipy.stats import pearsonr, spearmanr

    comp = df[["city", "country", "population_worldpop_2022",
               "population_africapolis_2020"]].copy()
    ap = comp["population_africapolis_2020"]
    ap_valid = np.isfinite(ap) & (ap > 0)
    comp["log_ratio_worldpop_to_africapolis"] = np.where(
        ap_valid,
        np.log(comp["population_worldpop_2022"]) - np.log(ap.where(ap_valid)),
        np.nan,
    )
    comp.to_csv(out_dir / "population_definition_comparison.csv", index=False)

    common = comp[ap_valid]
    if len(common) >= 10:
        pearson_r, pearson_p = pearsonr(
            np.log(common["population_worldpop_2022"]),
            np.log(common["population_africapolis_2020"]),
        )
        spearman_r, spearman_p = spearmanr(
            common["population_worldpop_2022"], common["population_africapolis_2020"]
        )
        lr = common["log_ratio_worldpop_to_africapolis"]
        logging.info(
            "Population definition comparison (n=%d common cities):\n"
            "  Pearson r (log-log)  = %.4f (p=%.2e)\n"
            "  Spearman rho         = %.4f (p=%.2e)\n"
            "  log-ratio (WP22/AP20): mean=%.3f  median=%.3f  sd=%.3f\n"
            "  median ratio (WP22/AP20) = %.3f",
            len(common), pearson_r, pearson_p, spearman_r, spearman_p,
            lr.mean(), lr.median(), lr.std(), float(np.exp(lr.median())),
        )
        top_div = common.reindex(lr.abs().sort_values(ascending=False).index).head(10)
        logging.info("Largest population-definition divergences:\n%s",
                     top_div[["city", "country", "population_worldpop_2022",
                              "population_africapolis_2020",
                              "log_ratio_worldpop_to_africapolis"]].to_string())
    logging.info("Population definition comparison saved.")


def write_covariate_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write covariate_summary.csv: for each categorical covariate, report
    N cities, N countries, and country coverage.
    Used to document interpretive limits (e.g. Independent=Ethiopia,
    Belgian≈DRC) and to assess separation risk before modelling.
    """
    rows = []

    for var in ["climate_zone", "colonial"]:
        for cat in sorted(df[var].dropna().unique()):
            sub = df[df[var] == cat]
            rows.append({
                "variable":   var,
                "category":   cat,
                "n_cities":   len(sub),
                "n_countries": sub["country"].nunique(),
                "pct_cities": round(len(sub) / len(df) * 100, 1),
                "top_countries": "; ".join(
                    sub["country"].value_counts().head(3).index.tolist()
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "covariate_summary.csv", index=False)
    logging.info("Covariate summary saved.")


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # --- Load raw inputs ---
    city_xlsx    = INPUT_DIR / "city.xlsx"
    country_xlsx = INPUT_DIR / "Country.xlsx"
    city_df    = pd.read_excel(city_xlsx)
    country_df = pd.read_excel(country_xlsx)
    logging.info("Loaded city.xlsx (%d rows) and Country.xlsx (%d rows).",
                 len(city_df), len(country_df))

    # --- City table ---
    city_df = city_df.rename(columns={"climate_zone(dominant)": "climate_zone",
                                       "population_2020": "population_africapolis_2020"})
    # Harmonise country names to match Country.xlsx spelling
    n_remapped = city_df["country"].isin(COUNTRY_NAME_MAP).sum()
    city_df["country"] = city_df["country"].replace(COUNTRY_NAME_MAP)
    logging.info("Country name harmonisation: %d cities remapped (%s).",
                 n_remapped,
                 ", ".join(f"{k}→{v}" for k, v in COUNTRY_NAME_MAP.items()))

    # --- WorldPop 2022 population (primary) ---
    # Per-city sum of the area-weighted WorldPop 2022 grid, from
    # 04_center_periphery/core/01_define_centres.py. Ethiopia uses WorldPop
    # 2020 there (2022 coverage does not exist for Ethiopia) — a documented
    # data-availability exception, not an inconsistency.
    if not CENTRES_PQ.exists():
        raise FileNotFoundError(
            f"city_centres.parquet not found: {CENTRES_PQ}\n"
            "Run 04_center_periphery/core/01_define_centres.py first."
        )
    centres_df = pd.read_parquet(
        CENTRES_PQ,
        columns=["city", "country", "total_pop2022",
                 "n_cells", "n_valid_population_cells", "population_coverage"],
    )
    centres_df = centres_df.rename(columns={
        "total_pop2022": "population_worldpop_2022",
        "n_cells": "n_population_cells",
    })
    n_before = len(city_df)
    city_df = city_df.merge(centres_df, on=["city", "country"], how="left")
    n_unmatched = city_df["population_worldpop_2022"].isna().sum()
    logging.info("WorldPop 2022 merge: %d/%d cities matched (%d unmatched).",
                 n_before - n_unmatched, n_before, n_unmatched)

    # --- Country table ---
    country_df = country_df.rename(columns={
        "Country": "country",
        " Geographic_Region": "geographic_region",
        "GDP_per_Capita(2020 dollar, source: World Bank)": "gdp_pc_2020",
        "Colonial History (source: Our World in Data Colonizers Dataset)":
            "colonial_raw",
    })
    country_df = country_df[
        ["country", "geographic_region", "gdp_pc_2020", "colonial_raw"]
    ].copy()
    country_df["log_gdp"] = np.log(country_df["gdp_pc_2020"].clip(lower=1))
    country_df["colonial"] = country_df["colonial_raw"].apply(encode_colonial)
    country_df = country_df.drop(columns=["colonial_raw", "gdp_pc_2020"])

    logging.info("Colonial legacy distribution (country level):\n%s",
                 country_df["colonial"].value_counts().to_string())

    unknown = (country_df["colonial"] == "Unknown").sum()
    if unknown > 0:
        logging.warning("%d countries have unknown colonial history.", unknown)

    # --- Community labels ---
    # Source: city_communities.parquet from 02_leiden_communities.py
    # community column is 0-based (0–8); convert to 1-based for analysis.
    part_path = CD_OUT / "city_communities.parquet"
    if not part_path.exists():
        raise FileNotFoundError(
            f"Community partition not found: {part_path}\n"
            "Run community_detection/02_leiden_communities.py first."
        )
    part_df = pd.read_parquet(part_path)[["city", "country", "community"]]
    part_df["community"] = part_df["community"] + 1   # 0-based → 1-based
    part_df["country"] = part_df["country"].replace(COUNTRY_NAME_MAP)
    logging.info("Loaded community partition: %d cities, %d communities.",
                 len(part_df), part_df["community"].nunique())

    # --- Merge ---
    df = (
        part_df
        .merge(city_df,    on=["city", "country"], how="inner")
        .merge(country_df, on="country",           how="left")
    )
    logging.info("After merge: %d cities, %d countries.",
                 len(df), df["country"].nunique())

    # --- Climate zone completeness check ---
    if df["climate_zone"].isna().sum() > 0:
        logging.warning("Unexpected missing climate_zone values: %d cities",
                        df["climate_zone"].isna().sum())
        logging.warning("%s", df.loc[df["climate_zone"].isna(),
                                     ["city", "country"]].to_string())

    # --- Population validity (WorldPop 2022, primary) ---
    # Zero, missing, or non-finite population is flagged and EXCLUDED from
    # the model sample rather than imputed (no clip(lower=1)-style fix),
    # per the population-definition policy documented above. A city is
    # also excluded if fewer than MIN_POPULATION_COVERAGE of its grid
    # cells have a valid (finite, positive) Pop2022 value — a low-coverage
    # city's total_pop2022 sum is not a reliable population estimate even
    # when it happens to be positive.
    pop_wp   = df["population_worldpop_2022"]
    coverage = df["population_coverage"]
    valid_pop      = np.isfinite(pop_wp) & (pop_wp > 0)
    valid_coverage = coverage.fillna(0) >= MIN_POPULATION_COVERAGE
    df["population_worldpop_2022_valid"] = valid_pop & valid_coverage
    n_invalid = (~df["population_worldpop_2022_valid"]).sum()
    if n_invalid > 0:
        excluded = df.loc[~df["population_worldpop_2022_valid"],
                          ["city", "country", "population_coverage"]]
        logging.warning(
            "%d cities have zero/missing/unmatched WorldPop 2022 population, "
            "or population_coverage below %.0f%%, and are EXCLUDED from the "
            "model sample (not imputed):\n%s",
            n_invalid, MIN_POPULATION_COVERAGE * 100, excluded.to_string(),
        )
    df = df[df["population_worldpop_2022_valid"]].copy()
    df["log_pop_worldpop_2022"] = np.log(df["population_worldpop_2022"])

    # --- Africapolis 2020 population (sensitivity-only) ---
    pop_ap = df["population_africapolis_2020"]
    ap_valid = np.isfinite(pop_ap) & (pop_ap > 0)
    df["log_pop_africapolis_2020"] = np.where(ap_valid, np.log(pop_ap.where(ap_valid)), np.nan)

    # --- Spatial terms ---
    df = standardise_spatial(df)

    df["log_pop_worldpop_2022_std"] = (
        (df["log_pop_worldpop_2022"] - df["log_pop_worldpop_2022"].mean())
        / df["log_pop_worldpop_2022"].std()
    )

    # --- Missing value report ---
    key_cols = ["community", "climate_zone", "colonial", "log_gdp",
                "log_pop_worldpop_2022", "lat_c", "lon_c"]
    missing = df[key_cols].isna().sum()
    logging.info("Missing values (post population-validity filter):\n%s",
                 missing.to_string())
    if missing.any():
        logging.warning("Rows with any missing key covariate: %d",
                        df[key_cols].isna().any(axis=1).sum())

    # --- Output columns (no dummy encoding here) ---
    out_cols = [
        "city", "country", "community",
        "latitude", "longitude",
        "climate_zone", "colonial",
        "log_gdp",
        "log_pop_worldpop_2022", "log_pop_worldpop_2022_std",
        "log_pop_africapolis_2020",
        "n_population_cells", "n_valid_population_cells", "population_coverage",
        "lat_c", "lon_c", "lat2", "lon2", "lat_lon",
    ]
    out = df[out_cols].copy()
    out.to_csv(OUT_TABLES / "analysis_input.csv", index=False)
    logging.info("Analysis input saved: %d rows × %d cols.",
                 len(out), out.shape[1])

    # --- Population definition sensitivity comparison ---
    write_population_comparison(df, OUT_TABLES)

    # --- Covariate summary and cross-tabulations ---
    write_covariate_summary(out, OUT_TABLES)

    # --- Console summary ---
    print(f"\n  Cities            : {len(out)}  ({n_invalid} excluded for "
          f"invalid WorldPop 2022 population)")
    print(f"  Countries         : {out['country'].nunique()}")
    print(f"  Communities       : {out['community'].nunique()}  "
          f"(sizes: {out['community'].value_counts().sort_index().to_dict()})")
    print(f"\n  Climate zones     : {sorted(out['climate_zone'].dropna().unique())}")
    print(f"  Colonial categories: {sorted(out['colonial'].dropna().unique())}")
    print(f"\n  Missing values (key columns):\n"
          + "\n".join(f"    {c}: {missing[c]}" for c in key_cols))


if __name__ == "__main__":
    main()
