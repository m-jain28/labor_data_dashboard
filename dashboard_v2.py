# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from folium.plugins import Fullscreen, MiniMap, MeasureControl
from branca.colormap import linear
import streamlit as st
from streamlit_folium import st_folium
import altair as alt

# ---------------- CONFIG ----------------
CSV_PATH = "us_county_unemployment_by_sex_2010_2023.csv"  # <- your panel CSV
# Use Cartographic Boundary counties (5m) from Census:
# e.g. https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip
LOCAL_SHP = "cb_2023_us_county_5m/cb_2023_us_county_5m.shp"
# ----------------------------------------

st.set_page_config(
    page_title="U.S. County Labor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")

# ---------- DATA LOADERS ----------
@st.cache_data(show_spinner=False)
def load_panel(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        dtype={"state_fips": "string", "county_fips": "string", "fips": "string"},
        low_memory=False,
    )

@st.cache_data(show_spinner=False)
def load_counties(local_shp: str) -> gpd.GeoDataFrame:
    shp = gpd.read_file(local_shp)
    # Cartographic boundary files usually have a GEOID column
    if "GEOID" in shp.columns:
        shp["fips"] = shp["GEOID"].astype(str)
    else:
        shp["fips"] = (shp["STATEFP"].astype(str) + shp["COUNTYFP"].astype(str))
    # Normalize name column
    if "NAME" not in shp.columns:
        if "NAMELSAD" in shp.columns:
            shp = shp.rename(columns={"NAMELSAD": "NAME"})
        else:
            shp["NAME"] = shp.index.astype(str)
    cols = [c for c in ["fips", "STATEFP", "NAME", "geometry"] if c in shp.columns]
    return shp[cols].to_crs(epsg=4326)

# ---------- GEOMETRY & COLOR HELPERS ----------
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
    return base.scale(vmin, vmax).to_step(index=bins.tolist()), bins

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

# FIX: underscore param so Streamlit doesn’t try to hash GeoDataFrame
@st.cache_data(show_spinner=False)
def simplify_counties(_gdf: gpd.GeoDataFrame, tol_deg: float) -> gpd.GeoDataFrame:
    gdf = _gdf.copy()
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = (
        gdf.geometry
           .simplify(tol_deg, preserve_topology=True)
           .buffer(0)
    )
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]
    return gdf

def make_oa_style_map(gdf: gpd.GeoDataFrame, value_col: str, title: str,
                      palette_name="YlOrRd", n_bins=7, classify="Quantiles (percentiles)"):
    cmap, _ = make_stepped_cmap(gdf[value_col], palette_name, n_bins, classify)

    m = folium.Map(location=[37.8, -96], zoom_start=4, control_scale=True, tiles=None)
    folium.TileLayer(
        tiles="CartoDB Positron",
        name="Positron",
        control=False,
        attr="© OpenStreetMap © CARTO",
    ).add_to(m)

    tooltip = GeoJsonTooltip(
        fields=["NAME", value_col],              # <-- correct key here
        aliases=["County:", f"{title}:"],
        localize=True,
        labels=True,
        sticky=False,
    )

    def style_fn(feat):
        val = feat["properties"].get(value_col, None)
        fill = "#d9d9d9" if (val is None or pd.isna(val)) else cmap(val)
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
