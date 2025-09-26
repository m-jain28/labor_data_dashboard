# Home.py — put this at repo root (same level as /pages)
import base64
from pathlib import Path
import streamlit as st

# ---------- PAGE ----------
st.set_page_config(
    page_title="Opportunity Atlas — Home",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapse the left sidebar like the mock
)

# ---------- OPTIONAL BACKGROUND ----------

# ---- Robust background image injector ----
import base64
from pathlib import Path
import streamlit as st

# point to the image relative to THIS file, not the CWD
BG_IMAGE = (Path(__file__).parent / "assets" / "newcomb.png").resolve()

def inject_bg(image_path: Path, overlay_alpha: float = 0.65, zoom: float = 0.5):
    """
    overlay_alpha: 0 = fully transparent overlay, 1 = fully white
    zoom: scale factor for the background image (1 = normal, >1 zooms in)
    """
    if not image_path.exists():
        st.warning(f"Background not found: {image_path}")
        return

    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
          .oa-bg {{
            position: fixed;
            inset: 0;
            z-index: -1;
            background:
              linear-gradient(180deg,
                rgba(255,255,255,{overlay_alpha}),
                rgba(255,255,255,{overlay_alpha})),
              url("data:{mime};base64,{b64}") no-repeat center center fixed;
            background-size: {zoom*100}% auto;   /* zoom in, keep centered */
          }}
          .stApp {{ background: transparent !important; }}
          html, body {{ background: #ffffff; }}
        </style>
        <div class="oa-bg"></div>
        """,
        unsafe_allow_html=True,
    )



inject_bg(BG_IMAGE)


# ---------- GLOBAL STYLES ----------
st.markdown(
    """
    <style>
    /* widen page and reduce default paddings */
    .block-container { max-width: 1400px; padding-top: 1.0rem; padding-bottom: 2rem; }

    /* Typography (use system stack so it works everywhere) */
    :root {
      --ink: #112E45;
      --teal: #0b7285;
      --ink-2: #1f2937;
      --tile-border: #e9ecef;
      --tile-shadow: 0 12px 36px rgba(16,32,65,.06), 0 1px 2px rgba(0,0,0,.04);
      --tile-shadow-hover: 0 18px 48px rgba(16,32,65,.14);
    }
    html, body, [class^="css"] {
      font-feature-settings: "ss01" on, "liga" on;
    }

    /* Left hero */
    .hero-title {
      font-size: clamp(48px, 6vw, 96px);
      font-weight: 900;
      letter-spacing: .04em;
      line-height: 1.04;
      color: var(--ink);
      margin: 0 0 1rem 0;
    }
    .hero-kicker {
      margin: 0 0 .8rem 0;
      font-size: clamp(18px, 2.7vw, 28px);
      font-weight: 800;
      color: var(--teal);
    }
    .lead {
      font-size: 18px;
      line-height: 1.6;
      color: var(--ink-2);
      background: #fff;
      padding: 1rem 1.25rem;
      border-radius: 12px;
      box-shadow: 0 8px 28px rgba(16,32,65,.08);
    }

    /* Right column header */
    .explore-title {
      font-size: 22px; font-weight: 900; letter-spacing: .06em;
      color: #102A43; text-transform: uppercase; margin: .4rem 0 .6rem 0;
    }

    /* ---- CARD (tile) styling for st.page_link across Streamlit versions ---- */
    #oa-cards [data-testid="stPageLinkList"],
    #oa-cards ul { list-style: none; margin: 0; padding: 0; }

    /* Match all known renderers of page_link anchors */
    #oa-cards [data-testid^="stPageLink"] a,
    #oa-cards li > a[kind="page-link"],
    #oa-cards a[data-baseweb="link"] {
      display: grid !important;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 12px;

      width: 100%;
      margin: 14px 0;
      padding: 22px 24px;

      background: #fff;
      border: 1px solid var(--tile-border);
      border-radius: 16px;
      box-shadow: var(--tile-shadow);

      text-decoration: none !important;
      color: inherit !important;

      transition: box-shadow .15s ease, transform .1s ease;
      line-height: 1.28;
      white-space: pre-wrap;  /* keep \\n in labels */
      font-size: 20px;        /* base size for title/subtitle lines */
      font-weight: 700;       /* the “title” line pops */
    }
    #oa-cards [data-testid^="stPageLink"] a:hover,
    #oa-cards li > a[kind="page-link"]:hover,
    #oa-cards a[data-baseweb="link"]:hover {
      box-shadow: var(--tile-shadow-hover);
      transform: translateY(-1px);
    }

    /* Hide any default icons; we draw our own chevron */
    #oa-cards [data-testid^="stPageLink"] a svg,
    #oa-cards li > a[kind="page-link"] svg,
    #oa-cards a[data-baseweb="link"] svg { display: none !important; }

    /* Right chevron */
    #oa-cards [data-testid^="stPageLink"] a::after,
    #oa-cards li > a[kind="page-link"]::after,
    #oa-cards a[data-baseweb="link"]::after {
      content: "›";
      font-size: 28px;
      font-weight: 800;
      color: var(--teal);
      line-height: 1;
      padding-left: 10px;
    }

    /* Make the first visual line (the “Neighborhood” kicker) look small & teal */
    #oa-cards [data-testid^="stPageLink"] a::first-line,
    #oa-cards li > a[kind="page-link"]::first-line,
    #oa-cards a[data-baseweb="link"]::first-line {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--teal);
      opacity: .95;
      font-weight: 800;
    }

        /***** OA card tiles rendered as buttons (robust across Streamlit versions) *****/
    #oa-cards [data-testid^="baseButton"],
    #oa-cards button {
    display: grid !important;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 12px;

    width: 100%;
    margin: 14px 0;
    padding: 22px 24px;

    background: #fff !important;
    border: 1px solid #e9ecef !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 36px rgba(16,32,65,.06), 0 1px 2px rgba(0,0,0,.04) !important;

    color: inherit !important;
    text-align: left !important;
    white-space: pre-wrap !important;    /* keep \n in the label */
    line-height: 1.28;
    font-size: 20px !important;
    font-weight: 700 !important;
    cursor: pointer;
    transition: box-shadow .15s ease, transform .1s ease;
    }

    /* hover */
    #oa-cards [data-testid^="baseButton"]:hover,
    #oa-cards button:hover {
    box-shadow: 0 18px 48px rgba(16,32,65,.14) !important;
    transform: translateY(-1px);
    }

    /* remove default focus ring styles that clash */
    #oa-cards [data-testid^="baseButton"]:focus,
    #oa-cards button:focus { outline: none !important; }

    /* Chevron at right */
    #oa-cards [data-testid^="baseButton"]::after,
    #oa-cards button::after {
    content: "›";
    font-size: 28px;
    font-weight: 800;
    color: #0b7285;
    line-height: 1;
    padding-left: 10px;
    justify-self: end;
    }

    /* “Neighborhood” kicker on the first line */
    #oa-cards [data-testid^="baseButton"]::first-line,
    #oa-cards button::first-line {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #0b7285;
    opacity: .95;
    font-weight: 800;
    }

        /* give room under Streamlit's top toolbar */
    .block-container{ 
    max-width:1400px; 
    padding-top:3.5rem;   /* was ~1rem */
    padding-bottom:2rem;
    overflow:visible;      /* avoid any accidental clipping */
    }

    /* tame the hero size & spacing just a bit */
    .hero-title{
    font-size:clamp(44px,5.6vw,88px);   /* slightly smaller max prevents overlap */
    line-height:1.06;
    margin:0 0 1.1rem 0;
    }

    /* start the right column lower so it aligns visually with the hero block */
    #oa-cards{ margin-top: 2.2rem; }

    /* base visual for the tile button already added; now adjust typography feel */
    #oa-cards button {
    line-height:1.28;
    font-size:22px !important;   /* a bit larger for the “title” feel */
    font-weight:800 !important;  /* title line will look boldest */
    letter-spacing:0.1px;
    }

    /* small teal kicker on first visual line (“Neighborhood”) */
    #oa-cards button::first-line{
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:.08em;
    color:#0b7285;
    opacity:.95;
    font-weight:800;
    }

    /* make the VERY end of the label look lighter (authors line) */
    #oa-cards button{
    /* nothing here; the trick is below using a non-breaking space + en dash */
    }

    /* optional: slightly tighter vertical rhythm on hover to mimic the mock */
    #oa-cards button:hover { transform: translateY(-1px); }

    /* Right rail: push all the way right and drop it lower */
    .right-rail{
    max-width: 620px;     /* controls tile column width */
    margin-left: auto;    /* hug the right edge inside its Streamlit column */
    margin-top: 3.2rem;   /* so it never sits under the big hero word */
    }

    /* optional: align the section title visually to the right */
    .right-rail .explore-title{ text-align: right; margin-right: 6px; }

    /* small spacing tweak between the title and the first tile */
    #oa-cards{ margin-top: .6rem; }




    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LAYOUT ----------
left, right = st.columns([1.65, 0.45], gap="large")

with left:
    st.markdown('<div class="hero-title">LABOR MARKET <br>DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-kicker">MAPPING LABOR MARKET OUTCOMES ACROSS THE U.S. BY GENDER</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="lead">
          <p>Which counties in America have the lowest unemployment rates for females?</p>
         
         How has occupational structure for females changed over time?</p>
         <p>Where do counties stand relative to state performance?</p>
          <p><strong>The Labor Market Dasboard by Newcomb Institute provides policy and program leaders with the latest evidence on labor statistics by gender,
          helping them develop and test solutions to enhance economic outcomes in their communities.</p></strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="right-rail">', unsafe_allow_html=True)
    st.markdown('<div class="explore-title">Explore the Data</div>', unsafe_allow_html=True)
    st.markdown('<div id="oa-cards">', unsafe_allow_html=True)

    # --- tiles (unchanged) ---
    if st.button(
        "County\n"
        "Labor Statistics\n"
        "— Overall, Male, Female\n",
        key="tile_outcomes"
    ):
        st.switch_page("pages/dashboard_v5.py")

    if st.button(
        "County\n"
        "Occupation Structure\n"
        "— Overall, Male, Female\n",
        key="tile_trends"
    ):
        st.switch_page("pages/dashboard_v6.py")

    st.markdown("</div></div>", unsafe_allow_html=True)  # closes #oa-cards and .right-rail
