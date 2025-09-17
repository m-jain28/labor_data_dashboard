from dotenv import load_dotenv
import os
import pandas as pd
from censusdis import data as ced

# ---- Config ----
YEAR = 2023
DATASET = "acs/acs5/subject"
STATE_FIPS = "22"  # Louisiana

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

df = ced.download(
    DATASET,
    YEAR,
    VARS,
    state=STATE_FIPS,
    county="*",
)

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

out = df.rename(columns=rename_map)

# --- Build FIPS on *out* (handles upper/lower case or GEO_ID) ---
cols = set(out.columns)

if {"state", "county"}.issubset(cols):
    out["fips"] = out["state"].astype(str).str.zfill(2) + out["county"].astype(str).str.zfill(3)
elif {"STATE", "COUNTY"}.issubset(cols):
    out["fips"] = out["STATE"].astype(str).str.zfill(2) + out["COUNTY"].astype(str).str.zfill(3)
elif "GEO_ID" in cols:
    # e.g., '0500000US22001' -> '22001'
    out["fips"] = out["GEO_ID"].str.extract(r"US(\d{5})$")[0]
else:
    raise RuntimeError(f"Can't find geography columns in out. Available: {list(out.columns)}")

print("Has fips?", "fips" in out.columns)  # should be True now

# reorder so fips and name come first, then all renamed variables
ordered = ["fips", "name"] + [v for v in rename_map.values() if v != "name"]
missing = [c for c in ordered if c not in out.columns]
if missing:
    raise RuntimeError(f"Missing before reorder: {missing}\nAvailable: {list(out.columns)}")

out = out[ordered]


out["acs_year"] = YEAR
out["acs_dataset"] = "acs5_subject"

out.sort_values("name").to_csv("la_county_unemployment_by_sex.csv", index=False)
print("Saved national_county_unemployment_by_sex.csv")
