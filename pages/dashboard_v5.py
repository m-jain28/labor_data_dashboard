# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import altair as alt
import pydeck as pdk
import altair as alt

import os, pathlib, streamlit as st




def atlas_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "grid": True,
                "gridColor": "#e9ecef",
                "tickColor": "#adb5bd",
                "domainColor": "#adb5bd",
                "labelColor": "#495057",
                "titleColor": "#343a40",
                "labelFontSize": 12,
                "titleFontSize": 13,
            },
            "axisX": {"labelAngle": 0, "grid": False},
            "legend": {
                "labelColor": "#495057",
                "titleColor": "#343a40",
                "orient": "top-left",
                "symbolType": "square",
            },
            "range": {
                "category": ["#0b7285", "#4c6ef5", "#fa5252", "#37b24d", "#f08c00"],
            },
            "background": "white",
        }
    }

alt.themes.register("atlas", atlas_theme)
alt.themes.enable("atlas")

# ============== CONFIG ==============
CSV_PATH = "us_county_unemployment_by_sex_2010_2023.csv"   # your panel CSV
LOCAL_SHP = "cb_2024_us_county_500k.shp"  # cartographic boundary counties

# U.S. view + bounds (deck.gl expects [[west, south], [east, north]])
US_CENTER = (37.8, -96.0)
US_BOUNDS_XY = [[-124.848974, 24.396308], [-66.885444, 49.384358]]

st.set_page_config(page_title="US County Labor Dashboard", layout="wide", initial_sidebar_state="expanded")
#alt.themes.enable("dark")

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

def legend_html(bins, ramp, title, subtitle="Total"):
    # Build the swatches/labels
    items = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        c = ramp[min(i, len(ramp) - 1)]
        color = f"rgb({c[0]}, {c[1]}, {c[2]})"
        items.append(
            f'<div class="leg-item"><span class="sw" style="background:{color}"></span>'
            f'<span class="lb">{lo:,.2f} – {hi:,.2f}</span></div>'
        )
    items_html = "\n".join(items)

    # Centered legend container (flex-center). One-row legend with no-wrap.
    return f"""
<div class="legend-wrap">
  <div class="oa-legend">
    <div class="legend-title">{title}</div>
    <div class="legend-sub">{subtitle}</div>
    <div class="legend-row">
      {items_html}
    </div>
  </div>
</div>

<style>
  /* Full-width wrapper that centers its child */
  .legend-wrap {{
    width: 100%;
    display: flex;
    justify-content: center;     /* center horizontally */
    margin-top: 10px;            /* space from the map */
  }}

  /* The legend card */
  .oa-legend {{
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
    max-width: 95vw;             /* keep it inside viewport on small screens */
    overflow-x: auto;            /* allow scroll if too many bins */
  }}

  .oa-legend .legend-title {{
    font-weight: 700;
    margin-bottom: 2px;
    color: #212529;
    text-align: center;
  }}

  .oa-legend .legend-sub {{
    font-size: 12px;
    color: #495057;
    margin-bottom: 6px;
    text-align: center;
  }}

  /* One single row, no wrapping; gap between items */
  .oa-legend .legend-row {{
    display: flex;
    flex-wrap: nowrap;           /* force single row */
    gap: 14px;
    align-items: center;
  }}

  .oa-legend .leg-item {{
    display: flex;
    align-items: center;
    white-space: nowrap;         /* keep each label on one line */
  }}

  .oa-legend .sw {{
    width: 16px;
    height: 12px;
    border: 1px solid #adb5bd;
    margin-right: 6px;
    flex: 0 0 16px;
  }}

  .oa-legend .lb {{
    font-size: 12px;
    color: #343a40;
  }}
</style>
"""




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
st.title("County Level Labor Market Statistics by Gender")
st.caption("Choose year, metric, group (overall/male/female), and (optional) state; hover counties for details.")

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
        ("Labor force participation rate (%)", "labor_force"),
        ("Population (count)", "pop"),
    ]
    available_cols = set(panel.columns)
    metric_options = [(label, base) for label, base in METRIC_BASES
                      if all(f"{base}_{g}" in available_cols for g in ["total", "male", "female"])]
    metric_label, metric_base = st.selectbox("Metric", metric_options, format_func=lambda x: x[0], index=0)

    group = st.radio("Group", ["Overall", "Male", "Female"], horizontal=True)
    n_bins = st.slider("Number of bins", 5, 9, 7)

    import us  # put this at the very top of your file with other imports

    # State filter (optional)
    STATE_FIPS_TO_NAME = {s.fips: s.name for s in us.states.STATES}
    STATE_NAME_TO_FIPS = {s.name: s.fips for s in us.states.STATES}

    states = sorted(panel["state_fips"].dropna().unique().tolist())
    state_names = [STATE_FIPS_TO_NAME.get(fips, fips) for fips in states]

    state_choice_name = st.selectbox("Filter by State (optional)", ["All states"] + state_names, index=0)

    if state_choice_name == "All states":
        state_choice = "All states"
    else:
        state_choice = STATE_NAME_TO_FIPS[state_choice_name]


# Active column
col_map = {"Overall": "total", "Male": "male", "Female": "female"}
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
    legend_html(bins, COLOR_RAMP, title=f"{metric_label} — {group}", subtitle="Total"),
    unsafe_allow_html=True,
)



# ===================== DETAILS + WHERE-IT-STANDS =====================
# Centered instruction line
st.markdown(
    """
    <div style="text-align:center;margin: 12px 0 6px 0;">
      <h3 style="margin:0;">Select a county for more details</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# County selector (center the widget on the page using 3 columns)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    county_opts = ["(None)"] + sorted(g["NAME"].dropna().unique().tolist())
    county_name = st.selectbox("Select a county for details", county_opts, key="county_select_centered")

# Two equal columns under the centered line
left_col, right_col = st.columns(2, gap="large")

# A consistent chart height for both sides
CHART_H = 280

# ---------------- LEFT: TIME TREND ----------------
with left_col:
    st.markdown(
        '<div style="text-align:center;"><h4 style="margin-top:0;">Time trend</h4></div>',
        unsafe_allow_html=True,
    )

    if not county_name or county_name == "(None)":
        st.caption("No county selected.")
    else:
        sel_row = g.loc[g["NAME"] == county_name].head(1)
        if sel_row.empty:
            st.caption("No data for the selected county.")
        else:
            cur_val = sel_row[active_col].iloc[0]
            fips_code = sel_row["fips"].iloc[0]

            # 2010–2023 trend for this county
            trend = (
                panel.loc[panel["fips"] == fips_code, ["acs_year", "state_fips", active_col]]
                .dropna()
                .sort_values("acs_year")
            )

            if trend.empty:
                st.caption("No time series for this county.")
            else:
                # Overlay US avg if "All states", else state avg (same color, dashed)
                avg_label = "Overlay U.S. average" if state_choice == "All states" else "Overlay state average"
                show_comp_avg = st.checkbox(avg_label, value=True, help="Dashed line = comparison average")

                comp_trend = pd.DataFrame()
                if show_comp_avg:
                    if state_choice == "All states":
                        comp_trend = (
                            panel[["acs_year", active_col]]
                            .groupby("acs_year", as_index=False).mean()
                            .sort_values("acs_year")
                        )
                    else:
                        state_code = str(trend["state_fips"].iloc[0])
                        comp_trend = (
                            panel.loc[panel["state_fips"] == state_code, ["acs_year", active_col]]
                            .groupby("acs_year", as_index=False).mean()
                            .sort_values("acs_year")
                        )

                latest_year = int(trend["acs_year"].max())

                base = (
                    alt.Chart(trend)
                    .encode(
                        x=alt.X("acs_year:O", title="Year"),
                        y=alt.Y(
                            f"{active_col}:Q",
                            title=f"{metric_label} ({group})",
                            scale=alt.Scale(zero=False, nice=True),
                        ),
                        tooltip=[
                            alt.Tooltip("acs_year:O", title="Year"),
                            alt.Tooltip(f"{active_col}:Q", title=metric_label, format=",.2f"),
                        ],
                    )
                    .properties(height=CHART_H)
                )

                county_line = base.mark_line(color="#1023cf", strokeWidth=3)

                hover = alt.selection_point(fields=["acs_year"], nearest=True, on="mousemove", empty=False)
                county_pts = base.mark_point(size=60, color="#1023cf").encode(
                    opacity=alt.condition(hover, alt.value(1), alt.value(0))
                )
                hover_rule = (
                    alt.Chart(trend)
                    .mark_rule(strokeDash=[4, 3], color="#868e96")
                    .encode(x="acs_year:O")
                    .add_params(hover)
                    .transform_filter(hover)
                )

                layers = [county_line, county_pts, hover_rule]

                if show_comp_avg and not comp_trend.empty:
                    st_line = (
                        alt.Chart(comp_trend)
                        .mark_line(strokeDash=[6, 4], color="#1023cf", strokeWidth=2, opacity=0.9)
                        .encode(x="acs_year:O", y=f"{active_col}:Q")
                    )
                    layers.append(st_line)

                last_data = trend[trend["acs_year"] == latest_year]
                last_dot = alt.Chart(last_data).mark_point(size=90, color="#0b7285").encode(
                    x="acs_year:O", y=f"{active_col}:Q"
                )
                last_label = (
                    alt.Chart(last_data)
                    .mark_text(align="left", dx=8, dy=-4, color="#1023cf", fontWeight="bold")
                    .encode(x="acs_year:O", y=f"{active_col}:Q", text=alt.Text(f"{active_col}:Q", format=",.2f"))
                )

                chart_left = alt.layer(*layers, last_dot, last_label).resolve_scale(y="shared")
                st.altair_chart(chart_left, use_container_width=True)

# ---------------- RIGHT: WHERE DOES THIS COUNTY STAND ----------------
with right_col:
    st.markdown(
        '<div style="text-align:center;"><h4 style="margin-top:0;">Where does this county stand?</h4></div>',
        unsafe_allow_html=True,
    )

    if not county_name or county_name == "(None)":
        st.caption("Select a county on the left to see where it stands.")
    else:
        # Comparison scope: Nation if All states, else selected state
        scope_name = "U.S." if state_choice == "All states" else "state"
        comp_geo = shp_us if state_choice == "All states" else shp_filtered

        comp_df = (
            comp_geo.merge(df_year[["fips", active_col]], on="fips", how="left")[[active_col]]
            .dropna()
            .rename(columns={active_col: "value"})
        )

        if comp_df.empty:
            st.caption("No data available to compute the distribution.")
        else:
            r = g.loc[g["NAME"] == county_name, active_col]
            county_val = float(r.iloc[0]) if (not r.empty and pd.notna(r.iloc[0])) else None

            scope_avg = float(comp_df["value"].mean())

            hist = (
                alt.Chart(comp_df)
                .mark_bar(color="#A4A6A8", opacity=0.95, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=metric_label),
                    y=alt.Y("count()", title="Counties"),
                    tooltip=[alt.Tooltip("count()", title="# of counties")],
                )
                .properties(height=CHART_H)
            )

            avg_rule = (
                alt.Chart(pd.DataFrame({"x": [scope_avg]}))
                .mark_rule(strokeDash=[5, 4], color="#1023cf", strokeWidth=2)
                .encode(x="x:Q")
            )

            layers_r = [hist, avg_rule]

            if county_val is not None:
                county_rule = (
                    alt.Chart(pd.DataFrame({"x": [county_val]}))
                    .mark_rule(color="#1023cf", strokeWidth=2.5)
                    .encode(x="x:Q")
                )
                layers_r.append(county_rule)

            chart_right = alt.layer(*layers_r)
            st.altair_chart(chart_right, use_container_width=True)

            def ordinal(pct: float) -> str:
                n = int(round(pct))
                return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

            if county_val is not None:
                pct = (comp_df["value"] <= county_val).mean() * 100.0
                st.caption(f"{county_name} is at the **{ordinal(pct)} percentile** within the **{scope_name}** in **{year}**.")
            else:
                st.caption("No value for the selected county in this year.")
