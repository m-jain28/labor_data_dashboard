# app.py — S2401-only
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import altair as alt
import pydeck as pdk
from pathlib import Path

# ---------- Paths (work from /pages/*) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs_s2401"
MERGED_CSV = ROOT / "merged_2010_2023.csv"      # your merged panel with S2401_*E columns
S2401_LABELS_CSV = DATA_DIR / "S2401_labels.csv"    # two cols: variable,label
LOCAL_SHP = ROOT / "cb_2024_us_county_500k.shp"

# ---------- Theme ----------
def atlas_theme():
    return {"config": {"view": {"stroke": "transparent"},
        "axis": {"grid": True, "gridColor": "#e9ecef", "tickColor": "#adb5bd",
                 "domainColor": "#adb5bd", "labelColor": "#495057", "titleColor": "#343a40",
                 "labelFontSize": 12, "titleFontSize": 13},
        "axisX": {"labelAngle": 0, "grid": False},
        "legend": {"labelColor": "#495057", "titleColor": "#343a40", "orient": "top-left", "symbolType": "square"},
        "range": {"category": ["#0b7285", "#4c6ef5", "#fa5252", "#37b24d", "#f08c00"]},
        "background": "white"}}
alt.themes.register("atlas", atlas_theme)
alt.themes.enable("atlas")

st.set_page_config(page_title="US County Labor — S2401", layout="wide", initial_sidebar_state="expanded")

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_panel(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={"state_fips":"string","county_fips":"string","fips":"string"}, low_memory=False)

@st.cache_data(show_spinner=False)
def load_counties(shp_path: str) -> gpd.GeoDataFrame:
    shp = gpd.read_file(shp_path)[["STATEFP","COUNTYFP","NAME","geometry"]]
    shp["fips"] = (shp["STATEFP"] + shp["COUNTYFP"]).astype(str)
    return shp.to_crs(epsg=4326)[["fips","STATEFP","NAME","geometry"]]

@st.cache_data(show_spinner=False)
def load_s2401_labels(path: str, panel_columns: list[str] | None = None) -> pd.DataFrame:
    lab = pd.read_csv(path, dtype=str)
    lab.columns = [str(c).strip().lower() for c in lab.columns]
    if "variable" not in lab.columns or "label" not in lab.columns:
        raise ValueError("S2401_labels.csv must have columns: variable,label")

    # Only keep S2401 estimate vars that exist in your data
    lab["variable"] = lab["variable"].astype(str).str.strip()
    lab = lab[lab["variable"].str.startswith("S2401_") & lab["variable"].str.endswith("E")].copy()

    # Split on '!!', strip trailing colons
    toks = lab["label"].fillna("").str.split("!!").apply(
        lambda xs: [str(x).strip().rstrip(":") for x in xs if str(x).strip() != ""]
    )

    # Keep only rows whose first token is Estimate
    stat0 = toks.str[0].str.lower()
    lab = lab[stat0.str.startswith("estimate")].copy()
    toks = toks.loc[lab.index]

    # Sex (look in first few tokens)
    def get_sex(tt):
        for t in tt[:3]:
            t = t.lower()
            if t.startswith("male"): return "Male"
            if t.startswith("female"): return "Female"
            if t.startswith("total"): return "Total"
        return "Total"

    # Find the anchor "...Civilian employed population 16 years and over"
    def parse_path(tt):
        anchor_i = None
        for i, t in enumerate(tt):
            if t.lower().startswith("civilian employed population"):
                anchor_i = i
                break
        if anchor_i is None:
            return pd.Series({"sex": get_sex(tt), "major": "", "sub": "", "item": ""})
        major = tt[anchor_i + 1] if len(tt) > anchor_i + 1 else ""
        sub   = tt[anchor_i + 2] if len(tt) > anchor_i + 2 else ""
        item  = tt[anchor_i + 3] if len(tt) > anchor_i + 3 else ""
        return pd.Series({"sex": get_sex(tt), "major": major, "sub": sub, "item": item})

    parsed = toks.apply(parse_path)
    lab = pd.concat([lab.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    # Leaf label to display
    lab["leaf"] = np.where(lab["item"] != "", lab["item"],
                   np.where(lab["sub"]  != "", lab["sub"], lab["major"]))

    # Keep only variables that exist in your merged panel (if provided)
    if panel_columns is not None:
        lab = lab[lab["variable"].isin(set(panel_columns))].copy()

    return lab[["variable", "label", "sex", "major", "sub", "item", "leaf"]]


# ---------- Color / legend ----------
def compute_quantile_bins(values: pd.Series, n=7):
    vals = values.dropna().astype(float).to_numpy()
    if vals.size == 0 or np.all(vals == vals[0]): return np.linspace(0, 1, n+1)
    bins = np.unique(np.quantile(vals, np.linspace(0, 1, n+1)))
    return bins if bins.size >= 3 else np.linspace(vals.min(), vals.max(), n+1)

COLOR_RAMP = [[255,255,204],[255,237,160],[254,217,118],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[177,0,38]]
def pick_rgba(v, bins, ramp=COLOR_RAMP):
    if v is None or pd.isna(v): return [217,217,217,200]
    v = float(v)
    for i in range(len(bins)-1):
        if v <= bins[i+1]: return ramp[min(i,len(ramp)-1)]+[220]
    return ramp[-1]+[220]

def legend_html(bins, ramp, title, subtitle=""):
    items = "".join(
        f'<div class="leg-item"><span class="sw" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>'
        f'<span class="lb">{lo:,.2f} – {hi:,.2f}</span></div>'
        for (lo,hi),c in zip(zip(bins[:-1],bins[1:]), [ramp[min(i,len(ramp)-1)] for i in range(len(bins)-1)])
    )
    return f"""
<div class="legend-wrap"><div class="oa-legend">
  <div class="legend-title">{title}</div>
  {'<div class="legend-sub">'+subtitle+'</div>' if subtitle else ''}
  <div class="legend-row">{items}</div>
</div></div>
<style>
.legend-wrap{{width:100%;display:flex;justify-content:center;margin-top:10px}}
.oa-legend{{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.legend-title{{font-weight:700;margin-bottom:2px;color:#212529;text-align:center}}
.legend-sub{{font-size:12px;color:#495057;margin-bottom:6px;text-align:center}}
.legend-row{{display:flex;flex-wrap:nowrap;gap:14px;align-items:center}}
.leg-item{{display:flex;align-items:center;white-space:nowrap}}
.sw{{width:16px;height:12px;border:1px solid #adb5bd;margin-right:6px;flex:0 0 16px}}
.lb{{font-size:12px;color:#343a40}}
</style>"""

# ---------- Map ----------
US_CENTER = (37.8, -96.0)
US_BOUNDS_XY = [[-124.848974, 24.396308], [-66.885444, 49.384358]]

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


# ================= APP =================
st.title("County-Level Occupational Structure")

# Load data & geometry
if not MERGED_CSV.exists():
    st.error(f"CSV not found: {MERGED_CSV}")
    st.stop()
panel = load_panel(str(MERGED_CSV))

if not S2401_LABELS_CSV.exists():
    st.error(f"Labels file not found: {S2401_LABELS_CSV}")
    st.stop()
s2401_labels = load_s2401_labels(str(S2401_LABELS_CSV), panel_columns=list(panel.columns))

if not LOCAL_SHP.exists():
    st.error(f"Shapefile not found: {LOCAL_SHP}")
    st.stop()
shp_us = load_counties(str(LOCAL_SHP))

# -------- Sidebar: S2401 selectors only --------
with st.sidebar:
    st.header("Controls")

    years = sorted(panel["acs_year"].dropna().unique().tolist())
    year = st.selectbox("Year", years, index=len(years)-1)

    sex = st.radio("Sex", ["Total", "Male", "Female"], index=0, horizontal=True)
    lab_sex = s2401_labels[s2401_labels["sex"] == sex].copy()

    # Canonical 5 major groups (filter + order)
    MAJOR_ORDER = [
        "Management, business, science, and arts occupations",
        "Service occupations",
        "Sales and office occupations",
        "Natural resources, construction, and maintenance occupations",
        "Production, transportation, and material moving occupations",
    ]
    # Some label files have a trailing colon in the major name; normalize it:
    lab_sex["major"] = lab_sex["major"].str.rstrip(":")

    available_majors = [m for m in MAJOR_ORDER if m in set(lab_sex["major"])]
    if not available_majors:
        st.error("No S2401 main groups found in labels matching the 5 major groups.")
        st.stop()

    major_choice = st.selectbox("Major occupation group", available_majors, index=0)

    # Subcategories under the chosen major
    lab_major = lab_sex[lab_sex["major"] == major_choice]
    subs = sorted([s for s in lab_major["sub"].unique() if s])
    has_subs = len(subs) > 0
    if has_subs:
        sub_choice = st.selectbox("Subcategory (optional)", ["(none)"] + subs, index=0)
        lab_sub = lab_major if sub_choice == "(none)" else lab_major[lab_major["sub"] == sub_choice]
    else:
        lab_sub = lab_major

        # Items under the chosen subcategory (if any). Make selection OPTIONAL.
    items = sorted([i for i in lab_sub["item"].unique() if i])

    if items:
        item_choice = st.selectbox("Item (optional)", ["(none)"] + items, index=0)
        if item_choice == "(none)":
            # choose the subtotal row at the sub-level if it exists (item == "")
            pick = lab_sub[(lab_sub["sub"] == lab_sub["sub"].iloc[0]) & (lab_sub["item"] == "")].head(1)
            # fallback: if no “sub total” row exists, use the first item
            if pick.empty:
                pick = lab_sub[lab_sub["item"] == items[0]].head(1)
                leaf_choice = items[0]
            else:
                leaf_choice = lab_sub["sub"].iloc[0]
        else:
            pick = lab_sub[lab_sub["item"] == item_choice].head(1)
            leaf_choice = item_choice
    else:
        # No item level — select within sub-level
        leaves = sorted([l for l in lab_sub["leaf"].unique() if l])
        leaf_choice = st.selectbox("Measure", leaves, index=0)
        pick = lab_sub[(lab_sub["leaf"] == leaf_choice)].head(1)


    var_code = pick["variable"].iloc[0]
    active_col = var_code
    metric_label = leaf_choice
    group = sex

    n_bins = st.slider("Number of bins", 5, 9, 7)
    state_fips_input = st.text_input("Filter by state FIPS (optional, e.g., 06 for CA)", value="").strip()
    state_choice = state_fips_input if state_fips_input else "All states"


# -------- Data slice & map --------
df_year = panel[panel["acs_year"] == year].copy()
if state_choice != "All states":
    df_year = df_year[df_year["state_fips"] == state_choice]
shp_filtered = shp_us if state_choice == "All states" else shp_us[shp_us["STATEFP"] == state_choice]

g = shp_filtered.merge(df_year[["fips", active_col]], on="fips", how="left")[["fips","NAME",active_col,"geometry"]]

st.subheader(f"{metric_label} · {group} · {year}")

# v5 behavior: national view when "All states"; fit to bbox when a state is chosen
if state_choice != "All states":
    current_view = view_for_gdf(shp_filtered)
else:
    current_view = pdk.ViewState(latitude=US_CENTER[0], longitude=US_CENTER[1],
                                 zoom=4, min_zoom=3, max_zoom=12, bearing=0, pitch=0)

deck, bins = make_deck_map(
    g,
    value_col=active_col,
    title=f"{metric_label} — {group}",
    n_bins=n_bins,
    view_state=current_view,
)

# IMPORTANT: give the map a key so the camera resets when inputs change
st.pydeck_chart(
    deck,
    use_container_width=True,
    height=650,
    key=f"map-{year}-{state_choice}-{active_col}"
)

st.markdown(legend_html(bins, COLOR_RAMP, title=metric_label, subtitle=f"S2401 — {group}"), unsafe_allow_html=True)


# ---------- Small helpers ----------
def _fmt_value(v):
    import math
    if v is None or pd.isna(v):
        return "—"
    try:
        v = float(v)
        # integer-like values as ints, otherwise 2 decimals
        if math.isfinite(v) and abs(v - round(v)) < 1e-9:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)

def show_selected_value_sentence(county_name: str, g: pd.DataFrame, active_col: str,
                                 major_choice: str, sub_choice: str | None,
                                 leaf_choice: str, group: str, year: int):
    """Render a sentence with county, value, major/sub/leaf occupation, group, and year."""
    if not county_name or county_name == "(None)":
        return
    row = g.loc[g["NAME"] == county_name]
    val = None if row.empty else row[active_col].iloc[0]

    if val is None or pd.isna(val):
        val_str = "—"
    else:
        val_str = f"{val:,.0f}" if float(val).is_integer() else f"{val:,.2f}"

    # Construct occupation phrase
    occ_phrase = major_choice
    if sub_choice and sub_choice != "(none)":
        occ_phrase += f" → {sub_choice}"
    if leaf_choice and leaf_choice not in [major_choice, sub_choice]:
        occ_phrase += f" → {leaf_choice}"

    st.markdown(
        f"""{county_name} has **{val_str}** **{group}** individuals in **{occ_phrase}** in **{year}**.""",
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

# Show the selected county's current value (metric/group/year)
show_selected_value_sentence(
    county_name,
    g,
    active_col,
    major_choice,
    sub_choice if has_subs else None,
    leaf_choice,
    group,
    year
)




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
