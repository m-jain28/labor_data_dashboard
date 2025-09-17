# app.py
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from branca.colormap import linear
import streamlit as st
from streamlit_folium import st_folium
import altair as alt

# ---------- CONFIG ----------
CSV_PATH = "us_county_unemployment_by_sex_2010_2023.csv"  # <- your panel output
LA_STATE_FIPS = "22"
#TIGER_COUNTY_ZIP = "https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip"
# ----------------------------

st.set_page_config(
    page_title="Louisiana Labor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

@st.cache_data(show_spinner=False)
def load_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"state_fips":"string","county_fips":"string","fips":"string"})
    return df



@st.cache_data(show_spinner=True)
def load_counties() -> gpd.GeoDataFrame:
    # Path to the local shapefile you unzipped
    local_shp = "tl_2024_us_county/tl_2024_us_county.shp"   

    shp = gpd.read_file(local_shp)
    shp["fips"] = (shp["STATEFP"] + shp["COUNTYFP"]).astype(str)

    # Keep only Louisiana
    shp = shp[shp["STATEFP"] == "22"].to_crs(epsg=4326)

    return shp[["fips", "NAME", "geometry"]]


from branca.colormap import linear

def make_choropleth(gdf, value_col: str, title: str, cmap_name="YlGnBu"):
    # Only use palettes that exist across branca versions
    PALETTES = {
        "YlGnBu": linear.YlGnBu_09,
        "YlOrRd": linear.YlOrRd_09,
        "Blues":  linear.Blues_09,
        "Reds":   linear.Reds_09,
        "Greens": linear.Greens_09,
        "Purples": linear.Purples_09,
        "Greys":  linear.Greys_09,
    }
    base_map = PALETTES.get(cmap_name, linear.Blues_09)

    # min/max for color scale
    vmin = float(gdf[value_col].min()) if value_col in gdf else 0.0
    vmax = float(gdf[value_col].max()) if value_col in gdf else 1.0
    if not (vmin < vmax):  # handle all-NaN or constant
        vmin, vmax = 0.0, 1.0

    cmap = base_map.scale(vmin, vmax)

    m = folium.Map(location=[31.0, -92.0], zoom_start=6, control_scale=True)

    def style_fn(feature):
        val = feature["properties"].get(value_col, None)
        color = "#cccccc" if val is None or pd.isna(val) else cmap(val)
        return {"fillColor": color, "color": "#555", "weight": 0.8, "fillOpacity": 0.85}

    folium.GeoJson(
        data=gdf,
        style_function=style_fn,
        highlight_function=lambda x: {"weight": 2, "color": "#000"},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["NAME", value_col],
            aliases=["Parish:", title + ": "],
            localize=True,
        ),
        name=title,
    ).add_to(m)

    cmap.caption = title
    cmap.add_to(m)
    return m

# --------- APP UI ----------
st.title("Louisiana County Labor Dashboard (ACS 5-year)")
st.caption("Pick a year and metric; view Total, Male, Female side-by-side for Louisiana parishes.")

panel = load_panel(CSV_PATH)
import os
st.write("CSV exists?", os.path.exists(CSV_PATH))
st.write("CSV path:", os.path.abspath(CSV_PATH))

if not os.path.exists(CSV_PATH):
    st.error("CSV not found. Fix CSV_PATH in app.py.")
    st.stop()

st.write("Panel shape:", panel.shape)
st.write("Panel columns:", list(panel.columns))
st.write("Example rows:", panel.head(3))
st.write("Years available:", sorted(panel["acs_year"].dropna().unique().tolist()))

shp_la = load_counties()

# Make a tidy list of candidate metrics from your naming scheme
# (we’ll only show metrics that actually exist in the CSV)
METRIC_BASES = [
    ("Unemployment rate (%)", "unemp_rate"),
    ("Employment-to-population ratio (%)", "emp_to_total_ratio"),
    ("Labor force (count)", "labor_force"),
    ("Population (count)", "pop"),
]

years = sorted(panel["acs_year"].dropna().unique().tolist())
year = st.selectbox("Year", years, index=0)

# Build metric options present in your CSV (robust to varying columns across years)
available_cols = set(panel.columns)
metric_options = []
for label, base in METRIC_BASES:
    trio = [f"{base}_total", f"{base}_male", f"{base}_female"]
    if all(col in available_cols for col in trio):
        metric_options.append((label, base))
if not metric_options:
    st.error("No compatible metrics found in your CSV.")
    st.stop()

metric_label, metric_base = st.selectbox(
    "Metric",
    metric_options,
    format_func=lambda x: x[0],
    index=0
)

# Choose the three columns we’ll map
col_total  = f"{metric_base}_total"
col_male   = f"{metric_base}_male"
col_female = f"{metric_base}_female"

# Filter panel data to Louisiana + chosen year
df_la = panel[(panel["state_fips"] == LA_STATE_FIPS) & (panel["acs_year"] == year)].copy()

st.write("Rows for LA/year:", len(df_la))
st.write("Unique FIPS in LA/year:", df_la["fips"].nunique() if "fips" in df_la else "no fips column")
st.write("Chosen metric columns present?:", 
         col_total in df_la.columns, col_male in df_la.columns, col_female in df_la.columns)





# Join to geometry
g = shp_la.merge(df_la, on="fips", how="left")

st.write("LA geoms:", len(shp_la), "Merged rows:", len(g))
st.write("Null metric counts:", 
         g[col_total].isna().sum() if col_total in g else "missing col")


# Build 3 synchronized maps
left, middle, right = st.columns(3)
with left:
    st.subheader(f"Total · {metric_label}")
    m_total = make_choropleth(g, col_total, f"{metric_label} — Total")
    st_folium(m_total, width=None, height=520)
with middle:
    st.subheader(f"Male · {metric_label}")
    m_male = make_choropleth(g, col_male, f"{metric_label} — Male")
    st_folium(m_male, width=None, height=520)
with right:
    st.subheader(f"Female · {metric_label}")
    m_female = make_choropleth(g, col_female, f"{metric_label} — Female")
    st_folium(m_female, width=None, height=520)

st.markdown(
    "<small>Source: ACS 5-year Subject Tables (S2301 et al.). "
    "Your CSV must include columns: `acs_year`, `state_fips`, `county_fips`, `fips`, "
    "`name`, and metric columns like `unemp_rate_total/male/female`.</small>",
    unsafe_allow_html=True
)
