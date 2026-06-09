"""
Nordek · NRK Price Analysis Dashboard
World-class redesign — IBM Plex Mono terminal aesthetic pushed to the extreme.
Every pixel intentional. Every interaction deliberate.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
import base64
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRK · Nordek Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — Refined terminal noir palette
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#050D18"       # near-black navy
PANEL    = "#080F1C"       # sidebar / panel
CARD     = "#0C1929"       # card surface
CARD2    = "#0E1E30"       # elevated card
BORDER   = "#142236"       # subtle border
BORDER2  = "#1C3050"       # active border
BLUE     = "#1E90FF"       # vivid electric blue
CYAN     = "#00D4FF"       # ice cyan
TEAL     = "#00C4A0"       # teal accent
GREEN    = "#00E5A0"       # signal green
RED      = "#FF4060"       # hot red
AMBER    = "#FFAA00"       # gold amber
PURPLE   = "#9B6DFF"       # electric purple
TEXT     = "#D8E8F8"       # cool white
SUBTEXT  = "#4A6885"       # muted blue-gray
DIM      = "#243B55"       # very dim
GLOW_B   = "rgba(30,144,255,0.15)"
GLOW_C   = "rgba(0,212,255,0.10)"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Meticulous every-detail styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root reset ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"], .main, .stApp {{
    background-color: {BG} !important;
    color: {TEXT};
    font-family: 'IBM Plex Mono', monospace;
}}

/* ── Kill Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {PANEL} !important;
    border-right: 1px solid {BORDER2} !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* ── Main content area ── */
.block-container {{
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {BLUE}; }}

/* ── Radio nav ── */
div[role="radiogroup"] {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0;
}}
div[role="radiogroup"] label {{
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 4px !important;
    padding: 9px 14px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    color: {SUBTEXT} !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    position: relative !important;
}}
div[role="radiogroup"] label:hover {{
    border-color: {BORDER2} !important;
    color: {TEXT} !important;
    background: rgba(30,144,255,0.05) !important;
}}
div[role="radiogroup"] label:has(input:checked) {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
    background: rgba(30,144,255,0.08) !important;
    font-weight: 600 !important;
}}
div[role="radiogroup"] label::before {{
    content: "";
    display: none;
}}
div[role="radiogroup"] input[type="radio"] {{
    display: none !important;
}}

/* ── Plotly charts: remove white backgrounds ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER2} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
    background: {CARD2} !important;
    color: {SUBTEXT} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid {BORDER2} !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: {TEXT} !important;
    background: {CARD} !important;
    padding: 9px 12px !important;
    border-bottom: 1px solid {BORDER} !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {{
    background: transparent !important;
    border: 1px solid {BORDER2} !important;
    color: {SUBTEXT} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    padding: 8px 16px !important;
    border-radius: 4px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
    background: rgba(30,144,255,0.06) !important;
}}

/* ═══════════════════════════
   CUSTOM COMPONENT CLASSES
═══════════════════════════ */

/* Topbar */
.topbar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER2};
    padding: 0 2rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -2rem 2rem -2rem;
    position: sticky;
    top: 0;
    z-index: 100;
}}
.topbar-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.topbar-badge {{
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {CYAN};
    border: 1px solid rgba(0,212,255,0.3);
    padding: 3px 8px;
    border-radius: 3px;
    background: rgba(0,212,255,0.05);
}}
.topbar-right {{
    display: flex;
    align-items: center;
    gap: 20px;
    font-size: 11px;
    color: {SUBTEXT};
    letter-spacing: 0.04em;
}}
.topbar-dot {{
    width: 6px;
    height: 6px;
    background: {GREEN};
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px {GREEN};
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

/* Page header */
.page-header {{
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {BORDER};
}}
.page-eyebrow {{
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {BLUE};
    margin-bottom: 8px;
}}
.page-title {{
    font-size: 28px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.02em;
    line-height: 1.1;
}}
.page-title span {{
    color: {BLUE};
}}
.page-sub {{
    font-size: 12px;
    color: {SUBTEXT};
    margin-top: 8px;
    line-height: 1.6;
    max-width: 600px;
}}

/* KPI row */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 2rem;
}}
.kpi-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.kpi-box:hover {{ border-color: {BORDER2}; }}
.kpi-box::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {BLUE}, {CYAN});
    opacity: 0;
    transition: opacity 0.2s;
}}
.kpi-box:hover::before {{ opacity: 1; }}
.kpi-label {{
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 10px;
}}
.kpi-value {{
    font-size: 22px;
    font-weight: 700;
    color: {TEXT};
    line-height: 1;
    letter-spacing: -0.02em;
}}
.kpi-meta {{
    font-size: 10px;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}}
.up   {{ color: {GREEN}; }}
.dn   {{ color: {RED}; }}
.neu  {{ color: {SUBTEXT}; }}

/* Section divider */
.sec {{
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {DIM};
    padding: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.sec::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

/* Metric card (model results) */
.m-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 24px 20px;
    text-align: center;
}}
.m-label {{
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 12px;
}}
.m-value {{
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}}
.m-desc {{
    font-size: 10px;
    color: {SUBTEXT};
    line-height: 1.6;
}}

/* Intern spotlight card */
.intern-card {{
    background: linear-gradient(135deg, {CARD} 0%, rgba(30,144,255,0.06) 100%);
    border: 1px solid {BORDER2};
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 2rem;
    display: flex;
    gap: 24px;
    align-items: flex-start;
    position: relative;
    overflow: hidden;
}}
.intern-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, {BLUE}, {CYAN});
}}
.intern-badge {{
    background: linear-gradient(135deg, {BLUE}, {CYAN});
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
    min-width: 80px;
    flex-shrink: 0;
}}
.intern-badge-title {{
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {BG};
    line-height: 1.4;
    text-transform: uppercase;
}}
.intern-name {{
    font-size: 18px;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 4px;
}}
.intern-sub {{
    font-size: 12px;
    color: {CYAN};
    margin-bottom: 10px;
    letter-spacing: 0.02em;
}}
.intern-desc {{
    font-size: 12px;
    color: {SUBTEXT};
    line-height: 1.7;
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* Bullet list */
.bullet-list {{ margin: 0; padding: 0; list-style: none; }}
.bullet-item {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid {BORDER};
    align-items: start;
}}
.bullet-item:last-child {{ border-bottom: none; }}
.bullet-icon {{
    width: 28px;
    height: 28px;
    border-radius: 4px;
    background: rgba(30,144,255,0.12);
    border: 1px solid rgba(30,144,255,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
    margin-top: 1px;
}}
.bullet-title {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 4px;
    letter-spacing: 0.01em;
}}
.bullet-desc {{
    font-size: 12px;
    color: {SUBTEXT};
    line-height: 1.7;
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* Step cards */
.step-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 2rem;
}}
.step-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.step-card:hover {{ border-color: {BORDER2}; }}
.step-num {{
    font-size: 64px;
    font-weight: 700;
    color: {BORDER};
    line-height: 1;
    position: absolute;
    top: 12px; right: 20px;
    user-select: none;
    pointer-events: none;
}}
.step-tag {{
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {BLUE};
    margin-bottom: 10px;
}}
.step-title {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 10px;
}}
.step-desc {{
    font-size: 12px;
    color: {SUBTEXT};
    line-height: 1.8;
    font-family: 'IBM Plex Sans', sans-serif;
    position: relative;
    z-index: 1;
}}

/* Timeline */
.timeline {{
    display: flex;
    gap: 0;
    margin: 0 0 2rem 0;
    border: 1px solid {BORDER};
    border-radius: 6px;
    overflow: hidden;
}}
.tl-item {{
    flex: 1;
    padding: 16px 20px;
    border-right: 1px solid {BORDER};
    position: relative;
}}
.tl-item:last-child {{ border-right: none; }}
.tl-month {{
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 8px;
}}
.tl-bar {{
    height: 3px;
    border-radius: 2px;
    margin-bottom: 10px;
}}
.tl-desc {{
    font-size: 11px;
    color: {TEXT};
    line-height: 1.6;
}}

/* Disclaimer */
.disclaimer {{
    background: rgba(255,64,96,0.06);
    border: 1px solid rgba(255,64,96,0.2);
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 11px;
    color: rgba(255,180,190,0.9);
    line-height: 1.7;
    margin: 1rem 0;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.disclaimer-icon {{ flex-shrink: 0; margin-top: 1px; font-size: 12px; }}

/* Sidebar nav section label */
.nav-section {{
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {DIM};
    padding: 16px 14px 6px 14px;
}}

/* Sidebar info panel */
.sidebar-info {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 16px;
    margin: 12px 0;
}}
.sidebar-info-label {{
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTEXT};
    margin-bottom: 8px;
}}
.sidebar-info-text {{
    font-size: 11px;
    color: {TEXT};
    line-height: 1.7;
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* Status pill */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    padding: 3px 9px;
    border-radius: 99px;
    letter-spacing: 0.04em;
}}
.pill-green {{
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.25);
    color: {GREEN};
}}
.pill-amber {{
    background: rgba(255,170,0,0.1);
    border: 1px solid rgba(255,170,0,0.25);
    color: {AMBER};
}}

/* Forecast table wrapper */
.fc-table-wrap {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1rem;
}}

/* Sidebar logo area */
.sidebar-logo {{
    padding: 20px 16px 0 16px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0;
    padding-bottom: 16px;
}}

/* Chart container */
.chart-wrap {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    margin-bottom: 1rem;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_logo_b64(height=40):
    path = "nordek_logo.png"
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{height}" style="display:block;" />'
    # SVG fallback — clean wordmark
    return f"""
    <svg width="120" height="{height}" viewBox="0 0 120 {height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="{BLUE}"/>
          <stop offset="100%" stop-color="{CYAN}"/>
        </linearGradient>
      </defs>
      <rect x="0" y="6" width="22" height="28" rx="3" fill="url(#lg)" opacity="0.9"/>
      <text x="4" y="25" font-family="IBM Plex Mono" font-size="11" font-weight="700"
            fill="{BG}">NRK</text>
      <text x="30" y="27" font-family="IBM Plex Mono" font-size="14" font-weight="700"
            fill="{TEXT}" letter-spacing="1">NORDEK</text>
    </svg>"""


def pbase(title="", h=420):
    """Shared Plotly layout factory."""
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=12, color=SUBTEXT),
                   x=0, xanchor="left", pad=dict(l=4, t=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", size=10, color=SUBTEXT),
        height=h,
        margin=dict(l=52, r=20, t=40, b=44),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER2, showgrid=True,
            tickfont=dict(size=10, color=SUBTEXT), zeroline=False,
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER2, showgrid=True,
            tickfont=dict(size=10, color=SUBTEXT), zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=SUBTEXT),
            bordercolor=BORDER, borderwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=CARD2, bordercolor=BORDER2,
            font=dict(family="IBM Plex Mono", size=11, color=TEXT),
        ),
    )


def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


def chart(fig):
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (all cached)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_price(v):
    if pd.isna(v):
        return np.nan
    return float(str(v).replace("$", "").strip())

def _parse_volume(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("$", "").strip().lower()
    if s.endswith("m"):   return float(s[:-1]) * 1_000_000
    if s.endswith("k"):   return float(s[:-1]) * 1_000
    return float(s)

@st.cache_data
def load_data():
    df = pd.read_csv("price_history_filtered.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y")
    df = (df[(df["Date"] >= "2023-06-01") & (df["Date"] <= "2023-08-31")]
          .copy().sort_values("Date").reset_index(drop=True))
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].apply(_parse_price).astype(float)
    df["Volume"]     = df["Volume"].apply(_parse_volume).astype(float)
    df["Volume_NRK"] = df["Volume(NRK)"].astype(float)
    df["High"]       = df["High"].fillna(df["Close"])
    df["Low"]        = df["Low"].fillna(df["Close"])
    return df.reset_index(drop=True)

@st.cache_data
def engineer_features(_df):
    df = _df.copy()
    c = df["Close"]
    df["MA3"]   = c.rolling(3).mean()
    df["MA7"]   = c.rolling(7).mean()
    df["MA14"]  = c.rolling(14).mean()
    df["EMA7"]  = c.ewm(span=7, adjust=False).mean()
    df["EMA14"] = c.ewm(span=14, adjust=False).mean()
    df["Vol7"]  = c.rolling(7).std()
    df["Ret1"]  = c.pct_change(1)
    df["Ret3"]  = c.pct_change(3)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"]   = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["HLSpread"] = df["High"] - df["Low"]
    for lag in range(1, 6):
        df[f"Lag{lag}"] = c.shift(lag)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"]     = df["Date"].dt.month

    FEATS = ["MA3","MA7","MA14","EMA7","EMA14","Vol7","Ret1","Ret3",
             "RSI14","HLSpread","Lag1","Lag2","Lag3","Lag4","Lag5",
             "DayOfWeek","Month","Volume"]
    df_feat = df.dropna(subset=FEATS).copy()
    for col in FEATS:
        df_feat[col] = df_feat[col].astype("float64")
    return df_feat, FEATS

@st.cache_data
def train_models(_df_feat, feat_cols):
    n     = len(_df_feat)
    split = int(n * 0.80)
    train = _df_feat.iloc[:split]
    test  = _df_feat.iloc[split:]

    X_tr  = train[feat_cols].values
    y_tr  = train["Close"].values
    X_te  = test[feat_cols].values
    y_te  = test["Close"].values

    sc = MinMaxScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, subsample=0.8, random_state=42)
    gb.fit(X_tr_sc, y_tr)

    y_pred_gb = gb.predict(X_te_sc)
    y_pred_bl = np.concatenate([[y_tr[-1]], y_te[:-1]])   # naive carry-forward

    mae_gb = mean_absolute_error(y_te, y_pred_gb)
    mae_bl = mean_absolute_error(y_te, y_pred_bl)

    actual_dir  = np.sign(np.diff(y_te))
    pred_dir_gb = np.sign(np.diff(y_pred_gb))
    dir_acc     = int(np.sum(actual_dir == pred_dir_gb))

    return dict(
        gb=gb, scaler=sc,
        train=train, test=test,
        y_test=y_te, y_pred_gb=y_pred_gb, y_pred_bl=y_pred_bl,
        mae_gb=mae_gb, mae_bl=mae_bl,
        dir_acc=dir_acc, n_test=len(y_te),
        feat_importances=dict(zip(feat_cols, gb.feature_importances_)),
    )

@st.cache_data
def build_forecast(_df, _results, feat_cols, days=30):
    df     = _df.copy()
    gb     = _results["gb"]
    sc     = _results["scaler"]
    hist   = list(df["Close"].values)
    highs  = list(df["High"].values)
    lows   = list(df["Low"].values)
    vols   = list(df["Volume"].values)
    last_d = pd.Timestamp(df["Date"].iloc[-1])
    preds  = []
    for i in range(days):
        c     = pd.Series(hist)
        nd    = last_d + pd.Timedelta(days=i+1)
        gain  = c.diff().clip(lower=0).rolling(14).mean().iloc[-1]
        loss  = (-c.diff().clip(upper=0)).rolling(14).mean().iloc[-1]
        row   = [c.rolling(3).mean().iloc[-1],
                 c.rolling(7).mean().iloc[-1],
                 c.rolling(14).mean().iloc[-1],
                 c.ewm(span=7, adjust=False).mean().iloc[-1],
                 c.ewm(span=14, adjust=False).mean().iloc[-1],
                 c.rolling(7).std().iloc[-1],
                 c.pct_change(1).iloc[-1],
                 c.pct_change(3).iloc[-1],
                 100 - 100 / (1 + gain / (loss + 1e-10)),
                 highs[-1] - lows[-1]] + \
                [hist[-j] for j in range(1, 6)] + \
                [nd.dayofweek, nd.month, float(np.mean(vols[-7:]))]
        p = float(gb.predict(sc.transform(np.array(row, dtype="float64").reshape(1,-1)))[0])
        preds.append((nd, p))
        hist.append(p); highs.append(p*1.01); lows.append(p*0.99)
        vols.append(float(np.mean(vols[-7:])))

    fc_prices = [x[1] for x in preds]
    roll_std  = float(pd.Series(df["Close"].values).rolling(7).std().iloc[-1])
    return dict(
        dates=[x[0] for x in preds], prices=fc_prices,
        upper=[p + 1.5*roll_std for p in fc_prices],
        lower=[p - 1.5*roll_std for p in fc_prices],
        hist_dates=list(df["Date"]), hist_prices=list(df["Close"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
df_raw             = load_data()
df_feat, feat_cols = engineer_features(df_raw)
results            = train_models(df_feat, feat_cols)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sidebar-logo">{get_logo_b64(36)}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["◈  My Work at Nordek",
         "◈  Price & Model Results",
         "◈  30-Day Forecast",
         "◈  How It Works"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="padding: 0 4px;">
      <div class="sidebar-info">
        <div class="sidebar-info-label">About</div>
        <div class="sidebar-info-text">
          ML price analysis built during a summer 2023 internship at Nordek.
          Feature engineering · model evaluation · data storytelling.
        </div>
      </div>
      <div style="font-size:10px;color:{DIM};padding:8px 4px;line-height:1.7;">
        Dataset · Jun–Aug 2023<br/>
        73 trading days · 18 features<br/>
        Model · Gradient Boosting
      </div>
    </div>
    """, unsafe_allow_html=True)

# Resolve page name
_page = page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR (rendered once, outside page logic)
# ─────────────────────────────────────────────────────────────────────────────
open_p  = df_raw["Open"].iloc[0]
close_p = df_raw["Close"].iloc[-1]
chg     = (close_p - open_p) / open_p * 100
chg_col = GREEN if chg >= 0 else RED
chg_sym = "▲" if chg >= 0 else "▼"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <span style="font-size:14px;font-weight:700;color:{TEXT};letter-spacing:0.02em;">NRK / USD</span>
    <span class="topbar-badge">NORDEK TOKEN</span>
    <span style="font-size:13px;font-weight:700;color:{TEXT};">${close_p:.4f}</span>
    <span style="font-size:12px;color:{chg_col};">{chg_sym} {abs(chg):.2f}% · period</span>
  </div>
  <div class="topbar-right">
    <span><span class="topbar-dot"></span>&nbsp; Data loaded</span>
    <span style="color:{BORDER2};">|</span>
    <span>Jun – Aug 2023</span>
    <span style="color:{BORDER2};">|</span>
    <span>73 days</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — My Work at Nordek
# ═════════════════════════════════════════════════════════════════════════════
if _page == "My Work at Nordek":

    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Portfolio · Summer 2023</div>
      <div class="page-title">My Work at <span>Nordek</span></div>
      <div class="page-sub">
        Applied ML internship — analysing on-chain NRK token data,
        engineering predictive features, and building a full ML pipeline
        for short-term price movement forecasting.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Intern spotlight
    st.markdown(f"""
    <div class="intern-card">
      <div class="intern-badge">
        <div class="intern-badge-title">Applied<br/>ML<br/>Intern<br/>2023</div>
      </div>
      <div>
        <div class="intern-name">Applied Machine Learning Intern</div>
        <div class="intern-sub">June – August 2023 &nbsp;·&nbsp; 3 months &nbsp;·&nbsp; Blockchain / DeFi</div>
        <div class="intern-desc">
          Collected and cleaned daily OHLCV data for the NRK token across a full summer window.
          Engineered 18 predictive signals, trained a Gradient Boosting model, and packaged
          everything into this interactive analytics dashboard for both technical and non-technical audiences.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    trading_days = len(df_raw)
    high_p = df_raw["High"].max()
    low_p  = df_raw["Low"].min()

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-box">
        <div class="kpi-label">Period Open</div>
        <div class="kpi-value">${open_p:.4f}</div>
        <div class="kpi-meta neu">1 Jun 2023</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Period Close</div>
        <div class="kpi-value">${close_p:.4f}</div>
        <div class="kpi-meta {'up' if chg>=0 else 'dn'}">{chg_sym} {abs(chg):.2f}% total return</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Highest Price</div>
        <div class="kpi-value">${high_p:.4f}</div>
        <div class="kpi-meta up">▲ Period peak</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Lowest Price</div>
        <div class="kpi-value">${low_p:.4f}</div>
        <div class="kpi-meta dn">▼ Period trough</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Trading Days</div>
        <div class="kpi-value">{trading_days}</div>
        <div class="kpi-meta neu">Jun – Aug 2023</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Candlestick + volume subplot
    sec("Price History · Candlestick + Moving Averages + Volume")

    ma7  = df_raw["Close"].rolling(7).mean()
    ma14 = df_raw["Close"].rolling(14).mean()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_raw["Date"], open=df_raw["Open"],
        high=df_raw["High"], low=df_raw["Low"], close=df_raw["Close"],
        name="NRK/USD",
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED, width=1),   fillcolor=RED),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_raw["Date"], y=ma7, name="MA 7",
        line=dict(color=BLUE, width=1.2, dash="dot"), opacity=0.9,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_raw["Date"], y=ma14, name="MA 14",
        line=dict(color=AMBER, width=1.2, dash="dash"), opacity=0.9,
    ), row=1, col=1)

    # Volume bars
    vol_colors = [GREEN if df_raw["Close"].iloc[i] >= df_raw["Open"].iloc[i] else RED
                  for i in range(len(df_raw))]
    fig.add_trace(go.Bar(
        x=df_raw["Date"], y=df_raw["Volume"],
        marker_color=vol_colors, opacity=0.7, name="Volume",
        showlegend=True,
    ), row=2, col=1)

    base = pbase(h=520)
    fig.update_layout(
        **base,
        xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=BORDER, linecolor=BORDER2, tickfont=dict(size=9, color=SUBTEXT)),
        yaxis=dict(tickprefix="$", gridcolor=BORDER, linecolor=BORDER2, tickfont=dict(size=10, color=SUBTEXT)),
        yaxis2=dict(tickformat=".2s", gridcolor=BORDER, linecolor=BORDER2, tickfont=dict(size=9, color=SUBTEXT)),
    )
    chart(fig)

    # What I did
    sec("Deliverables · What I Built")
    bullets = [
        ("📥", "Data Collection & Cleaning",
         "Gathered daily OHLCV data for the NRK token across the full summer. Parsed non-standard formats — dollar signs, K/M volume suffixes — and validated the dataset for consistency across 73 trading days."),
        ("⚙️", "Feature Engineering (18 signals)",
         "Built rolling averages (3/7/14-day), exponential smoothing, rolling volatility, RSI momentum, price-change rates, and 5 price lags. Each of these transforms raw OHLCV data into inputs a model can learn from."),
        ("🤖", "Model Training & Evaluation",
         "Trained a Gradient Boosting regressor on the first 80% of the dataset and evaluated on a held-out 15-day test window. Benchmarked against a naive carry-forward baseline to confirm real predictive value."),
        ("📊", "Interactive Dashboard",
         "Packaged the entire pipeline into this Streamlit app — showing candlestick charts, model performance in plain English, a 30-day forecast, and a walkthrough accessible to non-technical stakeholders."),
    ]

    st.markdown('<div class="bullet-list">', unsafe_allow_html=True)
    for icon, title, desc in bullets:
        st.markdown(f"""
        <div class="bullet-item">
          <div class="bullet-icon">{icon}</div>
          <div>
            <div class="bullet-title">{title}</div>
            <div class="bullet-desc">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price & Model Results
# ═════════════════════════════════════════════════════════════════════════════
elif _page == "Price & Model Results":

    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Evaluation · 15-day hold-out test window</div>
      <div class="page-title">Price &amp; <span>Model Results</span></div>
      <div class="page-sub">
        Gradient Boosting vs. naive carry-forward baseline —
        direction accuracy, mean absolute error, and residual analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    y_te      = results["y_test"]
    y_gb      = results["y_pred_gb"]
    y_bl      = results["y_pred_bl"]
    mae_gb    = results["mae_gb"]
    mae_bl    = results["mae_bl"]
    dir_acc   = results["dir_acc"]
    n_test    = results["n_test"]
    test_df   = results["test"]
    pct_better = (mae_bl - mae_gb) / mae_bl * 100

    # Metric trio
    c1, c2, c3 = st.columns(3)
    with c1:
        pct_dir = dir_acc / (n_test - 1) * 100
        d_col = GREEN if pct_dir >= 50 else AMBER
        st.markdown(f"""
        <div class="m-card">
          <div class="m-label">Direction Accuracy</div>
          <div class="m-value" style="color:{d_col};">{dir_acc}/{n_test-1}</div>
          <div class="m-desc">Correct UP / DOWN calls<br/>{pct_dir:.0f}% hit rate</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="m-card">
          <div class="m-label">Mean Abs. Error</div>
          <div class="m-value" style="color:{AMBER};">${mae_gb:.5f}</div>
          <div class="m-desc">Avg price gap — predicted<br/>vs actual on test set</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        v_col = GREEN if pct_better > 0 else RED
        st.markdown(f"""
        <div class="m-card">
          <div class="m-label">vs Naive Baseline</div>
          <div class="m-value" style="color:{v_col};">{pct_better:+.1f}%</div>
          <div class="m-desc">Gradient Boosting MAE<br/>improvement over baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Actual vs predicted
    sec("Predicted vs Actual · 15-day Test Window")
    fig_p = go.Figure()

    # Subtle fill between curves
    fig_p.add_trace(go.Scatter(
        x=list(test_df["Date"]) + list(test_df["Date"])[::-1],
        y=list(y_te) + list(y_gb)[::-1],
        fill="toself",
        fillcolor="rgba(30,144,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig_p.add_trace(go.Scatter(
        x=test_df["Date"], y=y_te,
        name="Actual", line=dict(color=CYAN, width=2.5),
        mode="lines+markers",
        marker=dict(size=5, color=CYAN, line=dict(width=1.5, color=BG)),
    ))
    fig_p.add_trace(go.Scatter(
        x=test_df["Date"], y=y_gb,
        name="Gradient Boosting", line=dict(color=BLUE, width=2, dash="dot"),
    ))
    fig_p.add_trace(go.Scatter(
        x=test_df["Date"], y=y_bl,
        name="Naive Baseline", line=dict(color=AMBER, width=1.5, dash="dash"), opacity=0.7,
    ))
    fig_p.update_layout(**pbase(h=380))
    fig_p.update_yaxes(tickprefix="$")
    chart(fig_p)

    # Feature importance + residuals side by side
    col_a, col_b = st.columns(2)

    with col_a:
        sec("Top 6 Features · Importance Score")
        label_map = {
            "Lag1":"Yesterday's close","Lag2":"2-day ago close","Lag3":"3-day ago close",
            "Lag4":"4-day ago close","Lag5":"5-day ago close",
            "MA3":"3-day avg","MA7":"7-day avg","MA14":"14-day avg",
            "EMA7":"7-day exp avg","EMA14":"14-day exp avg",
            "Vol7":"7-day volatility","Ret1":"1-day return","Ret3":"3-day return",
            "RSI14":"RSI momentum","HLSpread":"Daily range",
            "DayOfWeek":"Day of week","Month":"Month","Volume":"Volume",
        }
        fi_df = (pd.DataFrame(list(results["feat_importances"].items()),
                              columns=["feat","imp"])
                 .sort_values("imp", ascending=False).head(6))
        fi_df["label"] = fi_df["feat"].map(label_map).fillna(fi_df["feat"])

        fig_fi = go.Figure(go.Bar(
            x=fi_df["imp"], y=fi_df["label"],
            orientation="h",
            marker=dict(
                color=fi_df["imp"],
                colorscale=[[0, BORDER2], [1, CYAN]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in fi_df["imp"]],
            textposition="outside",
            textfont=dict(size=9, color=SUBTEXT, family="IBM Plex Mono"),
        ))
        fig_fi.update_layout(**pbase(h=300))
        fig_fi.update_xaxes(title_text="Importance")
        chart(fig_fi)

    with col_b:
        sec("Residuals · Prediction Error per Day")
        residuals = y_te - y_gb
        r_colors  = [GREEN if r >= 0 else RED for r in residuals]

        fig_r = go.Figure()
        fig_r.add_hline(y=0, line_color=SUBTEXT, line_dash="dot", line_width=1)
        fig_r.add_trace(go.Bar(
            x=test_df["Date"], y=residuals,
            marker_color=r_colors,
            marker_line_width=0,
            opacity=0.85,
            name="Error (actual − predicted)",
        ))
        fig_r.update_layout(
            **pbase("green = under-predicted · red = over-predicted", h=300)
        )
        fig_r.update_yaxes(tickprefix="$")
        chart(fig_r)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — 30-Day Forecast
# ═════════════════════════════════════════════════════════════════════════════
elif _page == "30-Day Forecast":

    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Projection · From 31 Aug 2023 · Portfolio demo only</div>
      <div class="page-title">30-Day <span>Forecast</span></div>
      <div class="page-sub">
        Gradient Boosting model rolled forward 30 days with ±1.5σ confidence band
        derived from 7-day rolling volatility.
      </div>
    </div>
    """, unsafe_allow_html=True)

    fc = build_forecast(df_raw, results, feat_cols, days=30)
    fc_arr   = np.array(fc["prices"])
    fc_start = fc_arr[0]; fc_end = fc_arr[-1]
    fc_max   = fc_arr.max(); fc_min = fc_arr.min()
    fc_pct   = (fc_end - fc_start) / fc_start * 100
    fc_sym   = "▲" if fc_pct >= 0 else "▼"
    fc_col   = GREEN if fc_pct >= 0 else RED

    st.markdown(f"""
    <div class="kpi-row" style="grid-template-columns: repeat(4, 1fr);">
      <div class="kpi-box">
        <div class="kpi-label">Forecast Start</div>
        <div class="kpi-value">${fc_start:.4f}</div>
        <div class="kpi-meta neu">1 Sep 2023</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Forecast End</div>
        <div class="kpi-value">${fc_end:.4f}</div>
        <div class="kpi-meta {'up' if fc_pct>=0 else 'dn'}">{fc_sym} {abs(fc_pct):.2f}% projected</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Projected High</div>
        <div class="kpi-value">${fc_max:.4f}</div>
        <div class="kpi-meta up">▲ Upper band peak</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-label">Projected Low</div>
        <div class="kpi-value">${fc_min:.4f}</div>
        <div class="kpi-meta dn">▼ Lower band trough</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sec("Full Context + 30-Day Outlook")

    fig_fc = go.Figure()

    # History area
    fig_fc.add_trace(go.Scatter(
        x=fc["hist_dates"], y=fc["hist_prices"],
        name="Historical", line=dict(color=CYAN, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
    ))

    # Confidence band
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"] + fc["dates"][::-1],
        y=fc["upper"] + fc["lower"][::-1],
        fill="toself",
        fillcolor="rgba(30,144,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1.5σ band",
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"], y=fc["prices"],
        name="Forecast", line=dict(color=BLUE, width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=3.5, color=BLUE),
    ))

    # Internship end vline
    fig_fc.add_vline(
        x=pd.Timestamp("2023-08-31"),
        line_color=AMBER, line_dash="dash", line_width=1.2,
        annotation_text="internship end",
        annotation_font_color=AMBER,
        annotation_font_size=10,
        annotation_font_family="IBM Plex Mono",
        annotation_position="top right",
    )

    fig_fc.update_layout(**pbase(h=460))
    fig_fc.update_yaxes(tickprefix="$")
    chart(fig_fc)

    st.markdown(f"""
    <div class="disclaimer">
      <span class="disclaimer-icon">⚠</span>
      <span>
        <strong>Disclaimer:</strong> This forecast is a portfolio demonstration only.
        Nordek ceased operations after August 2023 and the NRK token is no longer actively traded.
        These projections carry no investment advice and should not be used as the basis for any financial decision.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Day-by-day table
    sec("Day-by-Day Forecast Table")

    fc_table = pd.DataFrame({
        "Date":             [d.strftime("%d %b %Y") for d in fc["dates"]],
        "Projected Price":  [f"${p:.5f}" for p in fc["prices"]],
        "Upper Band (+1.5σ)": [f"${u:.5f}" for u in fc["upper"]],
        "Lower Band (−1.5σ)": [f"${l:.5f}" for l in fc["lower"]],
    })
    st.dataframe(fc_table, hide_index=True, use_container_width=True)

    csv_bytes = fc_table.to_csv(index=False).encode()
    st.download_button(
        "⬇  Export forecast as CSV",
        data=csv_bytes,
        file_name="nrk_30day_forecast.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — How It Works
# ═════════════════════════════════════════════════════════════════════════════
elif _page == "How It Works":

    st.markdown(f"""
    <div class="page-header">
      <div class="page-eyebrow">Methodology · Plain-English Walkthrough</div>
      <div class="page-title">How It <span>Works</span></div>
      <div class="page-sub">
        A non-technical breakdown of the data pipeline, feature engineering,
        model training, and evaluation process.
      </div>
    </div>
    """, unsafe_allow_html=True)

    sec("Project Steps")

    steps = [
        ("01", "STEP 1", "Gathering the Data",
         "Each day of the internship I recorded the NRK token's open, high, low, close, and trading volume. "
         "By the end of August I had 73 clean days — the full Jun–Aug 2023 window. "
         "Raw quirks like dollar signs and K/M volume suffixes were parsed and normalised."),
        ("02", "STEP 2", "Teaching the Model to Read Patterns",
         "Raw prices alone are noisy. I created 18 derived signals: rolling averages (3/7/14-day) that "
         "smooth out noise, momentum indicators (RSI) that detect whether the price is gaining or losing steam, "
         "price-change rates, and 5 lag features — 'what was the price N days ago?'"),
        ("03", "STEP 3", "Training & Testing",
         "I trained a Gradient Boosting regressor on the first ~80% of days and evaluated on the last 15 "
         "(never seen during training). To verify the result was meaningful, a naive carry-forward baseline "
         "was also evaluated — the GB model had to beat it to be considered useful."),
        ("04", "STEP 4", "Turning It Into a Story",
         "Numbers alone don't help stakeholders. I built this dashboard to show the historical chart, "
         "model accuracy in plain language, a 30-day forward projection, and this walkthrough. "
         "The goal: accessible to anyone, not just data scientists."),
    ]

    st.markdown('<div class="step-grid">', unsafe_allow_html=True)
    for num, tag, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div class="step-tag">{tag}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Timeline
    sec("Project Timeline · Summer 2023")

    tl_items = [
        ("June 2023", BLUE,   "Data collection begins · baseline exploration · initial cleaning"),
        ("July 2023", CYAN,   "Feature engineering · model architecture · first training runs"),
        ("August 2023", GREEN, "Model evaluation · dashboard build · stakeholder demo · internship end"),
    ]
    tl_html = '<div class="timeline">'
    for month, color, desc in tl_items:
        tl_html += f"""
        <div class="tl-item">
          <div class="tl-month">{month}</div>
          <div class="tl-bar" style="background:{color};"></div>
          <div class="tl-desc">{desc}</div>
        </div>"""
    tl_html += '</div>'
    st.markdown(tl_html, unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer">
      <span class="disclaimer-icon">⚠</span>
      <span>
        <strong>Important:</strong> This project was built as a portfolio demonstration.
        Nordek ceased operations after August 2023. Nothing here constitutes investment advice.
        All predictions are based solely on historical patterns in a 73-day dataset.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack summary
    sec("Tech Stack")
    tech = [
        ("Python 3.11",    "Runtime"),
        ("Streamlit",      "Dashboard framework"),
        ("scikit-learn",   "Gradient Boosting · MinMaxScaler"),
        ("Pandas / NumPy", "Data wrangling · feature engineering"),
        ("Plotly",         "Interactive charts"),
    ]
    cols = st.columns(len(tech))
    for col, (name, role) in zip(cols, tech):
        with col:
            st.markdown(f"""
            <div class="kpi-box" style="text-align:center;padding:14px 10px;">
              <div class="kpi-label">{role}</div>
              <div style="font-size:12px;font-weight:600;color:{BLUE};margin-top:6px;">{name}</div>
            </div>
            """, unsafe_allow_html=True)
