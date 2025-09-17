from dotenv import load_dotenv
import os
import pandas as pd
from censusdis import data as ced

# ---- Config ----
#YEAR = 2023
START_YEAR = 2010
END_YEAR = 2023

DATASET = "acs/acs5/subject"
#STATE_FIPS = "22"  # Louisiana
#OUT_CSV = "us_county_unemployment_by_sex_acs5_2010_2023.csv"

load_dotenv()

VARS = [
    "NAME",
    #"S2301_C04_001E",  # total unemployment rate (16+)
    "S2301_C01_021E",
    "S2301_C02_021E",
    "S2301_C03_021E",
    "S2301_C04_021E",
    "S2301_C01_022E",
    "S2301_C02_022E",
    "S2301_C03_022E",
    "S2301_C04_022E",
    "S2301_C01_023E",
    "S2301_C02_023E",
    "S2301_C03_023E",
    "S2301_C04_023E",

    #"S2301_C04_002E",  # male unemployment rate (16+)
    #"S2301_C04_003E",  # female unemployment rate (16+)
]



rename_map = {
    "NAME": "name",
    "S2301_C01_021E": "pop_total",
    "S2301_C02_021E": "labor_force_total",
    "S2301_C03_021E": "emp_to_total_ratio_total",
    "S2301_C04_021E": "unemp_rate_total",
    "S2301_C01_022E": "pop_male",
    "S2301_C02_022E": "labor_force_male",
    "S2301_C03_022E": "emp_to_total_ratio_male",
    "S2301_C04_022E": "unemp_rate_male",
    "S2301_C01_023E": "pop_female",
    "S2301_C02_023E": "labor_force_female",
    "S2301_C03_023E": "emp_to_total_ratio_female",
    "S2301_C04_023E": "unemp_rate_female",
}
"""
df = ced.download(
    DATASET,
    YEAR,
    VARS,
    state="*",
    county="*",
)
"""
all_years = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Downloading {year}…")
    try:
        df = ced.download(DATASET, year, VARS, state="*", county="*")
    except Exception as e:
        print(f"⚠️ Skipping {year} due to error: {e}")
        continue

    out = df.rename(columns=rename_map)
# ---- Rename variables ----
#out = df.rename(columns=rename_map)

    # ---- Build FIPS (use df, not out, because geo fields aren't renamed) ----
    if {"state", "county"}.issubset(df.columns):
        out["state_fips"] = df["state"].astype(str).str.zfill(2)
        out["county_fips"] = df["county"].astype(str).str.zfill(3)
    elif {"STATE", "COUNTY"}.issubset(df.columns):
        out["state_fips"] = df["STATE"].astype(str).str.zfill(2)
        out["county_fips"] = df["COUNTY"].astype(str).str.zfill(3)
    elif "GEO_ID" in df.columns:
        out["state_fips"] = df["GEO_ID"].str.extract(r"US(\d{2})\d{3}$")[0]
        out["county_fips"] = df["GEO_ID"].str.extract(r"US\d{2}(\d{3})$")[0]
    else:
        raise RuntimeError(f"Can't find geography columns for {year}. Columns: {list(df.columns)}")

    out["fips"] = out["state_fips"] + out["county_fips"]
    out["acs_year"] = year
    out["acs_dataset"] = "acs5_subject"
    all_years.append(out)

# append all years
panel = pd.concat(all_years, ignore_index=True)

# sort neatly
panel = panel.sort_values(["acs_year", "state_fips", "county_fips"]).reset_index(drop=True)

# save
panel.to_csv("us_county_unemployment_by_sex_2010_2023.csv", index=False)
print("Saved us_county_unemployment_by_sex_2010_2023.csv")
# ---- Reorder ----
#ordered = ["fips", "state_fips", "county_fips", "name"] + [v for v in rename_map.values() if v != "name"]
#out = out[ordered + ["acs_year", "acs_dataset"]]



#out["acs_year"] = YEAR
#out["acs_dataset"] = "acs5_subject"

#out = out.sort_values(["acs_year", "state_fips", "county_fips"]).reset_index(drop=True)

#out.to_csv("county_unemployment_by_sex.csv", index=False)
#print("Saved national_county_unemployment_by_sex.csv")


