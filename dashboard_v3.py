# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import altair as alt
import pydeck as pdk

# ============== CONFIG ==============
CSV_PATH = "us_county_unemployment_by_sex_2010_2023.csv"   # your panel CSV
LOCAL_SHP = "cb_2024_us_county_500k/cb_2024_us_county_500k.shp"  # cartographic boundary counties

# U.S. view + bounds (deck.gl expects [[west, south], [east, north]])
US_CENTER = (37.8, -96.0)
US_BOUNDS_XY = [[-124.848974, 24.396308], [-66.885444, 49.384358]]

st.set_page_config(page_title="US County Labor Dashboard", layout="wide", initial_sidebar_state="expanded")
alt.themes.enable("dark")

# ============== DATA LOADERS ==============
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
    shp = gpd.read_file(local_shp)[["STATEFP", "COUNTYFP", "NAME", "geometry"]]
    shp["fips"] = (shp["STATEFP"] + shp["COUNTYFP"]).astype(str)
    shp = shp.to_crs(epsg=4326)
    # Optional: small simplification to keep payload light; tweak tolerance if needed
    # shp["geometry"] = shp.geometry.simplify(tolerance=0.002, preserve_topology=True)
    return shp[["fips", "STATEFP", "NAME", "geometry"]]

# ============== COLOR BINS / LEGEND HELPERS ==============
def compute_quantile_bins(values: pd.Series, n=7):
    vals = values.dropna().astype(float).to_numpy()
    if vals.size == 0 or np.all(vals == vals[0]):
        bins = np.linspace(0, 1, n + 1)
    else:
        bins = np.unique(np.quantile(vals, np.linspace(0, 1, n + 1)))
        if bins.size < 3:  # fallback if too few uniques
            bins = np.linspace(vals.min(), vals.max(), n + 1)
    return bins

# YlOrRd-ish ramp (light -> dark)
COLOR_RAMP = [
    [255, 255, 204], [255, 237, 160], [254, 217, 118], [254, 178, 76],
    [253, 141, 60],  [252, 78, 42],   [227, 26, 28],   [177, 0, 38]
]

def pick_rgba(v, bins, ramp=COLOR_RAMP):
    if v is None or pd.isna(v):
        return [217, 217, 217, 200]  # gray for missing
    v = float(v)
    for i in range(len(bins) - 1):
        if v <= bins[i + 1]:
            return ramp[min(i, len(ramp) - 1)] + [220]
    return ramp[-1] + [220]

def legend_html(bins, ramp, title):
    # Build a small HTML legend (low→high)
    rows = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        c = ramp[min(i, len(ramp) - 1)]
        color = f"rgb({c[0]}, {c[1]}, {c[2]})"
        rows.append(
            f'<div style="display:flex;align-items:center;margin:2px 0">'
            f'<div style="width:18px;height:12px;background:{color};border:1px solid #444;margin-right:6px"></div>'
            f'<div style="font-size:12px">{lo:,.2f} – {hi:,.2f}</div>'
            f'</div>'
        )
    block = (
        f'<div style="background:#fff;border:1px solid #999;border-radius:6px;padding:8px 10px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.2);max-width:220px">'
        f'<div style="font-weight:600;margin-bottom:6px">{title}</div>'
        f'{"".join(rows)}'
        f'</div>'
    )
    return block

# ============== MAP (pydeck + Carto) ==============
# ============== MAP (pydeck + Carto) ==============
def view_for_gdf(gdf: gpd.GeoDataFrame) -> pdk.ViewState:
    """Return a reasonable ViewState centered on the gdf with a zoom that fits its bbox."""
    if gdf.empty or gdf.geometry.is_empty.all():
        return pdk.ViewState(latitude=US_CENTER[0], longitude=US_CENTER[1], zoom=4, min_zoom=3, max_zoom=12)

    minx, miny, maxx, maxy = gdf.total_bounds
    lat = (miny + maxy) / 2.0
    lon = (minx + maxx) / 2.0

    # Heuristic zoom from the larger angular span
    span_deg = max(maxx - minx, maxy - miny)
    if span_deg > 20:
        zoom = 4
    elif span_deg > 10:
        zoom = 5
    elif span_deg > 5:
        zoom = 6
    else:
        zoom = 7

    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, min_zoom=3, max_zoom=12, bearing=0, pitch=0)

def make_deck_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    n_bins: int = 7,
    view_state: pdk.ViewState | None = None,
):
    """Build a pydeck Deck with Carto basemap, quantile coloring, and
    an auto-fit view to gdf (unless a view_state is provided)."""

    # ----- bins + colors
    bins = compute_quantile_bins(gdf[value_col], n=n_bins)
    data = gdf[["NAME", value_col, "geometry"]].copy()
    data["fillColor"] = data[value_col].apply(lambda v: pick_rgba(v, bins, COLOR_RAMP))

    # ----- FeatureCollection (plain Python types only)
    fc = {"type": "FeatureCollection", "features": []}
    for _, row in data.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue
        val = None if pd.isna(row[value_col]) else float(row[value_col])
        val_str = "—" if val is None else (
            f"{val:,.2f}" if (abs(val) < 1000 or not float(val).is_integer()) else f"{int(val):,}"
        )
        fc["features"].append({
            "type": "Feature",
            "properties": {
                "NAME": str(row["NAME"]),
                value_col: val,
                "VALUE_STR": val_str,
                "fillColor": [int(c) for c in row["fillColor"]],
            },
            "geometry": geom.__geo_interface__,
        })

    # ----- auto-fit view (unless provided)
    if view_state is None:
        if gdf.empty or gdf.geometry.is_empty.all():
            view_state = pdk.ViewState(
                latitude=US_CENTER[0], longitude=US_CENTER[1],
                zoom=4, min_zoom=3, max_zoom=12, bearing=0, pitch=0
            )
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
            lat = (miny + maxy) / 2.0
            lon = (minx + maxx) / 2.0
            span_deg = max(maxx - minx, maxy - miny)
            # simple heuristic: smaller bbox -> closer zoom
            if span_deg > 20:
                zoom = 4
            elif span_deg > 10:
                zoom = 5
            elif span_deg > 5:
                zoom = 6
            else:
                zoom = 7
            view_state = pdk.ViewState(
                latitude=lat, longitude=lon,
                zoom=zoom, min_zoom=3, max_zoom=12, bearing=0, pitch=0
            )

    # ----- layer + deck
    layer = pdk.Layer(
        "GeoJsonLayer",
        fc,
        stroked=True,
        filled=True,
        get_fill_color="properties.fillColor",
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": f"<b>County:</b> {{NAME}}<br><b>{title}:</b> {{VALUE_STR}}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        views=[pdk.View(type="MapView", controller=True)],
        map_provider=None,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip=tooltip,
    )

    return deck, bins


# ============== APP ==============
st.title("U.S. County Labor Dashboard (ACS 5-year)")
st.caption("Opportunity-Atlas-style: choose year, metric, group, and state; hover counties for details. Pan/zoom is clamped to the U.S.")

# Load data
if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found at {os.path.abspath(CSV_PATH)}")
    st.stop()

panel = load_panel(CSV_PATH)
shp_us = load_counties(LOCAL_SHP)

# Sidebar
with st.sidebar:
    st.header("Controls")

    years = sorted(panel["acs_year"].dropna().unique().tolist())
    year = st.selectbox("Year", years, index=len(years) - 1)

    METRIC_BASES = [
        ("Unemployment rate (%)", "unemp_rate"),
        ("Employment-to-population ratio (%)", "emp_to_total_ratio"),
        ("Labor force (count)", "labor_force"),
        ("Population (count)", "pop"),
    ]
    available_cols = set(panel.columns)
    metric_options = [(label, base) for label, base in METRIC_BASES
                      if all(f"{base}_{g}" in available_cols for g in ["total", "male", "female"])]
    metric_label, metric_base = st.selectbox("Metric", metric_options, format_func=lambda x: x[0], index=0)

    group = st.radio("Group", ["Total", "Male", "Female"], horizontal=True)
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

# Filter geometry BEFORE merging to keep it light when a state is chosen
if state_choice != "All states":
    shp_filtered = shp_us[shp_us["STATEFP"] == state_choice].copy()
else:
    shp_filtered = shp_us

g = shp_filtered.merge(
    df_year[["fips", active_col]],
    on="fips", how="left"
)[["fips", "NAME", active_col, "geometry"]]   # <-- keep fips


# Map
st.subheader(f"{metric_label} — {group} · {year}")

# Choose a view: if a state is selected, fit to that state; otherwise use national view
if state_choice != "All states":
    current_view = view_for_gdf(shp_filtered)   # fits to that state's counties
else:
    current_view = pdk.ViewState(latitude=US_CENTER[0], longitude=US_CENTER[1],
                                 zoom=4, min_zoom=3, max_zoom=12, bearing=0, pitch=0)

deck, bins = make_deck_map(
    g,
    value_col=active_col,
    title=f"{metric_label} — {group}",
    n_bins=n_bins,
    view_state=current_view,   # <<< pass it in
)

st.pydeck_chart(deck, use_container_width=True, height=650)

st.markdown(
    legend_html(bins, COLOR_RAMP, title=f"{metric_label} — {group}"),
    unsafe_allow_html=True,
)



# Details + Distribution
left, right = st.columns([1, 1])

# ----- DETAILS (left) + WHERE-IT-STANDS (right) -----
left, right = st.columns([1, 1])

with left:
    st.markdown("### Details")

    county_opts = ["(None)"] + sorted(g["NAME"].dropna().unique().tolist())
    county_name = st.selectbox(
        "Select a county for details",
        county_opts,
        key="county_select"  # <-- only defined ONCE
    )

    if county_name == "(None)":
        st.caption("No county selected.")
    else:
        sel_row = g[g["NAME"] == county_name].head(1)
        cur_val = sel_row[active_col].iloc[0]
        fips_code = sel_row["fips"].iloc[0]

        st.markdown(f"**County:** {county_name}")
        if pd.isna(cur_val):
            st.markdown(f"**{metric_label} ({group}) — {year}:** —")
        else:
            fv = float(cur_val)
            st.markdown(f"**{metric_label} ({group}) — {year}:** {fv:,.2f}")

        # ----- 2010–2023 trend for this county -----
        trend = (
            panel.loc[panel["fips"] == fips_code, ["acs_year", "state_fips", active_col]]
            .dropna()
            .sort_values("acs_year")
        )

        if trend.empty:
            st.caption("No time series for this county.")
        else:
            show_state_avg = st.checkbox("Overlay state average", value=True, help="Dashed line = state average")

            state_trend = pd.DataFrame()
            if show_state_avg:
                state_code = str(trend["state_fips"].iloc[0])
                state_trend = (
                    panel.loc[panel["state_fips"] == state_code, ["acs_year", active_col]]
                    .groupby("acs_year", as_index=False)
                    .mean()
                    .sort_values("acs_year")
                )

            latest_year = int(trend["acs_year"].max())

            base = alt.Chart(trend).encode(
                x=alt.X("acs_year:O", title="Year"),
                y=alt.Y(f"{active_col}:Q", title=f"{metric_label} ({group})")
            ).properties(height=260)

            hover = alt.selection_point(fields=["acs_year"], nearest=True, on="mousemove", empty=False)

            county_line = base.mark_line()
            county_pts  = base.mark_point(size=40).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0))
            )
            hover_rule = alt.Chart(trend).mark_rule(strokeDash=[4, 3]).encode(
                x="acs_year:O"
            ).add_params(hover).transform_filter(hover)

            layers = [county_line, county_pts, hover_rule]

            if show_state_avg and not state_trend.empty:
                st_line = alt.Chart(state_trend).mark_line(strokeDash=[6, 4]).encode(
                    x="acs_year:O", y=f"{active_col}:Q"
                )
                layers.append(st_line)

            last_data = trend[trend["acs_year"] == latest_year]
            last_dot = alt.Chart(last_data).mark_point(size=70).encode(
                x="acs_year:O", y=f"{active_col}:Q"
            )
            last_label = alt.Chart(last_data).mark_text(align="left", dx=6).encode(
                x="acs_year:O",
                y=f"{active_col}:Q",
                text=alt.Text(f"{active_col}:Q", format=",.2f"),
            )

            chart = alt.layer(*layers, last_dot, last_label).resolve_scale(y="shared")
            st.altair_chart(chart, use_container_width=True)

with right:
    st.markdown("### Where does this county stand?")

    # Nation if All states, else the selected state
    if state_choice == "All states":
        scope_name = "U.S."
        comp_geo = shp_us
    else:
        scope_name = "state"
        comp_geo = shp_filtered

    comp_df = (
        comp_geo.merge(df_year[["fips", active_col]], on="fips", how="left")[[active_col]]
        .dropna()
        .rename(columns={active_col: "value"})
    )

    if comp_df.empty:
        st.caption("No data available to compute the distribution.")
    else:
        # Read the county selected on the left (no new widget here)
        county_name = st.session_state.get("county_select", "(None)")
        county_val = None
        if county_name and county_name != "(None)":
            sel_row = g[g["NAME"] == county_name][[active_col]]
            if not sel_row.empty and pd.notna(sel_row.iloc[0, 0]):
                county_val = float(sel_row.iloc[0, 0])

        scope_avg = float(comp_df["value"].mean())

        base = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=metric_label),
                y=alt.Y("count()", title="Counties"),
            )
            .properties(height=220)
        )

        layers = [base]

        # scope average (dashed)
        layers.append(
            alt.Chart(pd.DataFrame({"x": [scope_avg]}))
            .mark_rule(strokeDash=[4, 3])
            .encode(x="x:Q")
        )

        # selected county (solid)
        if county_val is not None:
            layers.append(
                alt.Chart(pd.DataFrame({"x": [county_val]}))
                .mark_rule()
                .encode(x="x:Q")
            )

        st.altair_chart(alt.layer(*layers), use_container_width=True)

        if county_val is not None:
            pct = (comp_df["value"] <= county_val).mean() * 100.0
            st.caption(
                f"{county_name} is at the **{pct:.1f}th** percentile within the **{scope_name}** in **{year}**."
            )
        else:
            st.caption("Pick a county on the left to see its position in the distribution.")


