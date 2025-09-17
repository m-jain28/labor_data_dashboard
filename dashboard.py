# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from branca.colormap import linear
from folium.plugins import Fullscreen, MiniMap, MeasureControl
import streamlit as st
from streamlit_folium import st_folium
import altair as alt

# ------------ CONFIG ------------
CSV_PATH = "us_county_unemployment_by_sex_2010_2023.csv"  # your panel csv
#LOCAL_SHP = "tl_2024_us_county/tl_2024_us_county.shp"
LOCAL_SHP = "cb_2024_us_county_500k/cb_2024_us_county_500k.shp"
# --------------------------------

st.set_page_config(
    page_title="US County Labor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")

# ---------- DATA LOADERS ----------
@st.cache_data(show_spinner=False)
def load_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={"state_fips": "string", "county_fips": "string", "fips": "string"},
        low_memory=False,
    )
    return df

@st.cache_data(show_spinner=False)
def load_counties(local_shp: str) -> gpd.GeoDataFrame:
    shp = gpd.read_file(local_shp)
    shp["fips"] = (shp["STATEFP"] + shp["COUNTYFP"]).astype(str)
    return shp[["fips", "STATEFP", "NAME", "geometry"]].to_crs(epsg=4326)

# ---------- COLOR / LEGEND HELPERS ----------
PALETTES = {
    "YlGnBu": linear.YlGnBu_09,
    "YlOrRd": linear.YlOrRd_09,
    "Blues":  linear.Blues_09,
    "Reds":   linear.Reds_09,
    "Greens": linear.Greens_09,
    "Purples":linear.Purples_09,
    "Greys":  linear.Greys_09,
}

def make_stepped_cmap(values: pd.Series, palette_name="YlOrRd", n=7, mode="Quantiles (percentiles)"):
    base = PALETTES.get(palette_name, linear.YlOrRd_09)
    vals = np.array(values.dropna(), dtype=float)

    if vals.size == 0 or np.all(vals == vals[0]):
        vmin, vmax = 0.0, 1.0
        bins = np.linspace(vmin, vmax, n + 1)
        return base.scale(vmin, vmax).to_step(n), bins

    if mode.startswith("Quantiles"):
        qs = np.linspace(0, 1, n + 1)
        bins = np.unique(np.quantile(vals, qs))
        if bins.size < 3:
            bins = np.linspace(vals.min(), vals.max(), n + 1)
    else:
        bins = np.linspace(vals.min(), vals.max(), n + 1)

    vmin, vmax = float(bins[0]), float(bins[-1])
    cmap = base.scale(vmin, vmax).to_step(index=bins.tolist())
    return cmap, bins

def format_value_for_tooltip(val):
    if val is None or pd.isna(val):
        return "—"
    try:
        f = float(val)
        if abs(f) >= 1000 and f.is_integer():
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except Exception:
        return str(val)

# ---------- MAP BUILDER ----------
def make_oa_style_map(gdf: gpd.GeoDataFrame, value_col: str, title: str,
                      palette_name="YlOrRd", n_bins=7, classify="Quantiles (percentiles)"):
    cmap, bins = make_stepped_cmap(gdf[value_col], palette_name, n_bins, classify)

    # Center map on US
    m = folium.Map(
        location=[37.8, -96],  # continental US center
        zoom_start=4,
        control_scale=True,
        tiles=None,
    )
    folium.TileLayer(
        tiles="CartoDB Positron",
        name="Positron",
        control=False,
        attr="© OpenStreetMap © CARTO",
    ).add_to(m)

    tooltip = GeoJsonTooltip(
        fields=["NAME", value_col],
        aliases=["County:", f"{title}:"],
        localize=True,
        labels=True,
        sticky=False,
    )

    def style_fn(feat):
        val = feat["properties"].get(value_col, None)
        if val is None or pd.isna(val):
            fill = "#d9d9d9"
        else:
            fill = cmap(val)
        return {"fillColor": fill, "color": "#666", "weight": 0.3, "fillOpacity": 0.85}

    gj = folium.GeoJson(
        data=gdf,
        name=title,
        style_function=style_fn,
        highlight_function=lambda x: {"weight": 2, "color": "#000"},
        tooltip=tooltip,
    ).add_to(m)

    folium.GeoJsonPopup(
        fields=["NAME", value_col],
        aliases=["County", title],
        localize=True,
        labels=True,
        parse_html=True,
        max_width=320,
    ).add_to(gj)

    cmap.caption = title
    cmap.add_to(m)

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl().add_to(m)

    return m

# ---------- APP ----------
st.title("U.S. County Labor Dashboard (ACS 5-year)")
st.caption("Opportunity-Atlas-style: choose year, metric, group, and state; hover or click counties for details.")

# Load data
if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found at {os.path.abspath(CSV_PATH)}")
    st.stop()

panel = load_panel(CSV_PATH)
shp_us = load_counties(LOCAL_SHP)

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    years = sorted(panel["acs_year"].dropna().unique().tolist())
    year = st.selectbox("Year", years, index=len(years)-1)

    METRIC_BASES = [
        ("Unemployment rate (%)", "unemp_rate"),
        ("Employment-to-population ratio (%)", "emp_to_total_ratio"),
        ("Labor force (count)", "labor_force"),
        ("Population (count)", "pop"),
    ]
    available_cols = set(panel.columns)
    metric_options = [(label, base) for label, base in METRIC_BASES
                      if all(f"{base}_{g}" in available_cols for g in ["total","male","female"])]
    metric_label, metric_base = st.selectbox(
        "Metric", metric_options, format_func=lambda x: x[0], index=0
    )

    group = st.radio("Group", ["Total", "Male", "Female"], horizontal=True)
    palette = st.selectbox("Color palette", list(PALETTES.keys()), index=1)
    classify = st.selectbox("Legend type", ["Quantiles (percentiles)", "Equal intervals"], index=0)
    n_bins = st.slider("Number of bins", 5, 9, 7)

    # State filter (optional)
    states = sorted(panel["state_fips"].dropna().unique().tolist())
    state_choice = st.selectbox("Filter by State (optional)", ["All states"] + states, index=0)

# Active column
col_map = {"Total": "total", "Male": "male", "Female": "female"}
active_col = f"{metric_base}_{col_map[group]}"

# Filter panel data
df_year = panel[panel["acs_year"] == year].copy()
if state_choice != "All states":
    df_year = df_year[df_year["state_fips"] == state_choice]

g = shp_us.merge(df_year, on="fips", how="left")

# Map
st.subheader(f"{metric_label} — {group} · {year}")
m = make_oa_style_map(
    g,
    value_col=active_col,
    title=f"{metric_label} — {group}",
    palette_name=palette,
    n_bins=n_bins,
    classify=classify,
)
m_state = st_folium(m, width=None, height=650)

# Details + Distribution
left, right = st.columns([1, 1])
selected_name, selected_val = None, None
if m_state and m_state.get("last_active_drawing"):
    props = m_state["last_active_drawing"].get("properties", {})
    selected_name = props.get("NAME")
    selected_val = props.get(active_col)

with left:
    st.markdown("### Details")
    if selected_name is not None:
        st.markdown(f"**County:** {selected_name}")
        st.markdown(f"**{metric_label} ({group})**: {format_value_for_tooltip(selected_val)}")
    else:
        st.caption("Click a county to see details.")

with right:
    st.markdown("### Distribution (current selection)")
    dist = g[[active_col]].dropna().rename(columns={active_col: "value"})
    if not dist.empty:
        base = alt.Chart(dist).mark_bar().encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=metric_label),
            y=alt.Y("count()", title="Counties"),
        ).properties(height=200)

        layers = [base]
        state_avg = float(dist["value"].mean())
        layers.append(
            alt.Chart(pd.DataFrame({"x": [state_avg]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
        )
        if selected_val is not None and pd.notna(selected_val):
            layers.append(
                alt.Chart(pd.DataFrame({"x": [float(selected_val)]})).mark_rule().encode(x="x:Q")
            )
        st.altair_chart(alt.layer(*layers), use_container_width=True)
    else:
        st.caption("No data to plot.")

# Download current table
st.download_button(
    "Download current county table (CSV)",
    df_year.to_csv(index=False).encode(),
    file_name=f"us_{metric_base}_{group.lower()}_{year}.csv",
    mime="text/csv",
)
