"""
Nordek · NRK Price Analysis  ·  v3 — Designer Edition
Single logo · rich sidebar · cinematic hero · no redundant topbar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings, base64, os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="NRK · Nordek", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Tokens ────────────────────────────────────────────────────────────────────
BG      = "#04090F"
PANEL   = "#060D16"
CARD    = "#0A1520"
CARD2   = "#0D1B28"
BORDER  = "#112030"
BORDER2 = "#1A3248"
BLUE    = "#1A8FFF"
CYAN    = "#00CFFF"
TEAL    = "#00B8A0"
GREEN   = "#00E5A0"
RED     = "#FF3D5C"
AMBER   = "#FFA500"
PURPLE  = "#8B5CF6"
TEXT    = "#DCE9F8"
MUTED   = "#8BA8C4"
SUBTEXT = "#3E5A75"
DIM     = "#1A2E42"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,[class*="css"],.main,.stApp{{
  background:{BG}!important;color:{TEXT};
  font-family:'IBM Plex Mono',monospace;
}}
#MainMenu,footer,header,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"]{{
  display:none!important;height:0!important;visibility:hidden!important
}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:3px;height:3px}}
::-webkit-scrollbar-track{{background:{BG}}}
::-webkit-scrollbar-thumb{{background:{BORDER2};border-radius:2px}}

/* ── Main content ── */
.block-container{{padding:0 2.5rem 4rem 2.5rem!important;max-width:1440px!important}}

/* ────────────────────────────
   SIDEBAR
──────────────────────────── */
[data-testid="stSidebar"]{{
  background:{PANEL}!important;
  border-right:1px solid {BORDER2}!important;
  padding:0!important;
  width:270px!important;
}}
[data-testid="stSidebar"]>div:first-child{{padding-top:0!important}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;font-family:'IBM Plex Mono',monospace!important}}

/* Radio nav */
div[role="radiogroup"]{{display:flex;flex-direction:column;gap:2px;padding:0 12px}}
div[role="radiogroup"] label{{
  background:transparent!important;
  border:1px solid transparent!important;
  border-radius:6px!important;
  padding:10px 14px!important;
  margin:0!important;cursor:pointer!important;
  transition:all 0.18s ease!important;
  font-size:11px!important;letter-spacing:0.04em!important;
  color:{MUTED}!important;display:flex!important;align-items:center!important;
}}
div[role="radiogroup"] label:hover{{
  border-color:{BORDER2}!important;color:{TEXT}!important;
  background:rgba(26,143,255,0.06)!important;
}}
div[role="radiogroup"] label:has(input:checked){{
  border-color:rgba(26,143,255,0.5)!important;color:{BLUE}!important;
  background:rgba(26,143,255,0.1)!important;font-weight:600!important;
}}
div[role="radiogroup"] input[type="radio"]{{display:none!important}}

/* Download button */
[data-testid="stDownloadButton"] button{{
  background:transparent!important;border:1px solid {BORDER2}!important;
  color:{MUTED}!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:11px!important;letter-spacing:0.06em!important;
  padding:9px 18px!important;border-radius:5px!important;
  transition:all 0.15s!important;cursor:pointer!important;
}}
[data-testid="stDownloadButton"] button:hover{{
  border-color:{BLUE}!important;color:{BLUE}!important;
  background:rgba(26,143,255,0.07)!important;
}}

/* Dataframe */
[data-testid="stDataFrame"]{{border:1px solid {BORDER2}!important;border-radius:8px!important;overflow:hidden}}
[data-testid="stDataFrame"] th{{background:{CARD2}!important;color:{SUBTEXT}!important;
  font-family:'IBM Plex Mono',monospace!important;font-size:9px!important;
  letter-spacing:0.1em!important;text-transform:uppercase!important;
  padding:10px 14px!important;border-bottom:1px solid {BORDER2}!important}}
[data-testid="stDataFrame"] td{{font-family:'IBM Plex Mono',monospace!important;
  font-size:12px!important;color:{TEXT}!important;background:{CARD}!important;
  padding:9px 14px!important;border-bottom:1px solid {BORDER}!important}}

/* ════════════════════════════
   SIDEBAR COMPONENTS
════════════════════════════ */

/* Hero logo block — full width, immersive */
.sb-hero{{
  background:linear-gradient(160deg,{CARD2} 0%,{BG} 100%);
  border-bottom:1px solid {BORDER2};
  padding:28px 20px 22px;
  position:relative;overflow:hidden;
}}
.sb-hero::before{{
  content:"";position:absolute;top:-40px;right:-40px;
  width:140px;height:140px;border-radius:50%;
  background:radial-gradient(circle,rgba(26,143,255,0.12) 0%,transparent 70%);
  pointer-events:none;
}}
.sb-hero::after{{
  content:"NRK";position:absolute;bottom:-12px;right:16px;
  font-size:72px;font-weight:700;letter-spacing:-0.04em;
  color:rgba(26,143,255,0.05);pointer-events:none;line-height:1;
  font-family:'IBM Plex Mono',monospace;
}}
.sb-token-pill{{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(0,207,255,0.08);border:1px solid rgba(0,207,255,0.2);
  border-radius:99px;padding:4px 10px;margin-top:10px;
}}
.sb-token-dot{{
  width:5px;height:5px;border-radius:50%;background:{CYAN};
  box-shadow:0 0 6px {CYAN};animation:pulse 2s infinite;
}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}
.sb-token-label{{font-size:9px;letter-spacing:0.1em;color:{CYAN};text-transform:uppercase}}

.sb-price-block{{
  margin-top:16px;padding-top:16px;border-top:1px solid {BORDER};
  display:flex;justify-content:space-between;align-items:flex-end;
}}
.sb-price-main{{font-size:26px;font-weight:700;color:{TEXT};letter-spacing:-0.03em;line-height:1}}
.sb-price-chg{{font-size:11px;margin-top:4px}}

/* Stat rows in sidebar */
.sb-stats{{padding:0 20px;margin-top:4px}}
.sb-stat-row{{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid {BORDER};
  font-size:11px;
}}
.sb-stat-row:last-child{{border-bottom:none}}
.sb-stat-label{{color:{SUBTEXT};letter-spacing:0.04em}}
.sb-stat-value{{color:{TEXT};font-weight:500}}

/* Nav label */
.sb-nav-label{{
  font-size:8px;letter-spacing:0.14em;text-transform:uppercase;
  color:{SUBTEXT};padding:16px 20px 6px;
}}

/* Sidebar footer */
.sb-footer{{
  padding:14px 20px;border-top:1px solid {BORDER};
  margin-top:auto;
}}
.sb-footer-text{{font-size:10px;color:{SUBTEXT};line-height:1.7;font-family:'IBM Plex Sans',sans-serif}}
.sb-version{{
  display:inline-flex;gap:6px;align-items:center;
  margin-top:8px;font-size:9px;letter-spacing:0.08em;
}}

/* ════════════════════════════
   PAGE COMPONENTS
════════════════════════════ */

/* Cinematic page hero — full bleed */
.pg-hero{{
  background:linear-gradient(135deg,{CARD2} 0%,rgba(4,9,15,0) 60%);
  border:1px solid {BORDER2};border-radius:12px;
  padding:36px 40px;margin-bottom:2rem;
  position:relative;overflow:hidden;
}}
.pg-hero::before{{
  content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{BLUE},{CYAN},transparent);
}}
.pg-hero-grid{{display:grid;grid-template-columns:1fr auto;gap:32px;align-items:start}}
.pg-eyebrow{{
  font-size:9px;letter-spacing:0.16em;text-transform:uppercase;
  color:{BLUE};margin-bottom:10px;
  display:flex;align-items:center;gap:8px;
}}
.pg-eyebrow::before{{content:"";width:20px;height:1px;background:{BLUE};opacity:0.5}}
.pg-title{{font-size:32px;font-weight:700;color:{TEXT};letter-spacing:-0.03em;line-height:1.05}}
.pg-title em{{color:{BLUE};font-style:normal}}
.pg-desc{{font-size:12px;color:{MUTED};margin-top:10px;line-height:1.75;max-width:560px;font-family:'IBM Plex Sans',sans-serif}}
.pg-hero-meta{{
  text-align:right;padding-top:4px;
}}
.pg-hero-badge{{
  background:linear-gradient(135deg,rgba(26,143,255,0.15),rgba(0,207,255,0.08));
  border:1px solid rgba(26,143,255,0.3);border-radius:8px;
  padding:16px 20px;min-width:160px;
}}
.pg-badge-val{{font-size:22px;font-weight:700;color:{TEXT};letter-spacing:-0.02em}}
.pg-badge-label{{font-size:9px;color:{MUTED};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px}}
.pg-badge-sub{{font-size:10px;margin-top:4px}}

/* KPI strip */
.kpi-strip{{display:grid;gap:10px;margin-bottom:1.75rem}}
.kpi{{
  background:{CARD};border:1px solid {BORDER};border-radius:8px;
  padding:18px 20px;position:relative;overflow:hidden;
  transition:border-color 0.2s,transform 0.2s;
  cursor:default;
}}
.kpi:hover{{border-color:{BORDER2};transform:translateY(-1px)}}
.kpi::after{{
  content:"";position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,{BLUE},transparent);
  opacity:0;transition:opacity 0.2s;
}}
.kpi:hover::after{{opacity:1}}
.kpi-label{{font-size:8px;letter-spacing:0.14em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:10px}}
.kpi-value{{font-size:24px;font-weight:700;color:{TEXT};line-height:1;letter-spacing:-0.025em}}
.kpi-meta{{font-size:10px;margin-top:7px;display:flex;align-items:center;gap:5px}}
.up{{color:{GREEN}}}.dn{{color:{RED}}}.neu{{color:{MUTED}}}

/* Section header */
.sec{{
  font-size:8px;letter-spacing:0.16em;text-transform:uppercase;color:{SUBTEXT};
  padding:1.75rem 0 0.85rem;display:flex;align-items:center;gap:14px;
}}
.sec::after{{content:"";flex:1;height:1px;background:linear-gradient(90deg,{BORDER},{BG})}}

/* Chart card */
.chart-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:10px;
  padding:6px 6px 2px;margin-bottom:1rem;
}}

/* Metric cards */
.m-grid{{display:grid;gap:10px;margin-bottom:1.5rem}}
.m-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:8px;
  padding:24px 22px;position:relative;overflow:hidden;
}}
.m-card::before{{
  content:"";position:absolute;left:0;top:0;bottom:0;width:2px;
  background:linear-gradient(180deg,{BLUE},{CYAN});
}}
.m-label{{font-size:8px;letter-spacing:0.14em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:14px}}
.m-value{{font-size:40px;font-weight:700;line-height:1;letter-spacing:-0.04em;margin-bottom:8px}}
.m-desc{{font-size:10px;color:{MUTED};line-height:1.65;font-family:'IBM Plex Sans',sans-serif}}

/* Model table */
.model-table{{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:1rem}}
.model-table th{{
  font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};
  padding:10px 14px;border-bottom:1px solid {BORDER2};text-align:left;background:{CARD2};
}}
.model-table td{{padding:11px 14px;border-bottom:1px solid {BORDER};color:{TEXT};vertical-align:middle}}
.model-table tr:last-child td{{border-bottom:none}}
.model-table tr.winner{{background:rgba(0,229,160,0.03)}}
.model-table tr.winner td{{color:{TEXT}}}
.model-table tr.dim td{{color:{SUBTEXT};font-style:italic}}
.badge{{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:3px;font-size:8px;letter-spacing:0.06em;font-weight:600}}
.badge-green{{background:rgba(0,229,160,0.12);color:{GREEN};border:1px solid rgba(0,229,160,0.25)}}
.badge-blue{{background:rgba(26,143,255,0.12);color:{BLUE};border:1px solid rgba(26,143,255,0.25)}}
.badge-red{{background:rgba(255,61,92,0.12);color:{RED};border:1px solid rgba(255,61,92,0.25)}}
.badge-purple{{background:rgba(139,92,246,0.12);color:{PURPLE};border:1px solid rgba(139,92,246,0.25)}}

/* Intern card */
.intern-strip{{
  background:linear-gradient(135deg,{CARD2},rgba(26,143,255,0.04));
  border:1px solid {BORDER2};border-radius:10px;padding:28px 32px;
  margin-bottom:1.75rem;display:flex;gap:28px;align-items:flex-start;
  position:relative;overflow:hidden;
}}
.intern-strip::before{{
  content:"";position:absolute;top:0;left:0;width:2px;height:100%;
  background:linear-gradient(180deg,{BLUE},{CYAN});
}}
.intern-strip::after{{
  content:"'23";position:absolute;right:28px;bottom:-8px;
  font-size:96px;font-weight:700;color:rgba(26,143,255,0.04);
  font-family:'IBM Plex Mono',monospace;line-height:1;pointer-events:none;
}}
.ib{{
  background:linear-gradient(135deg,{BLUE},{CYAN});border-radius:8px;
  padding:16px;text-align:center;min-width:76px;flex-shrink:0;
}}
.ib-text{{font-size:8px;font-weight:700;letter-spacing:0.1em;color:{BG};line-height:1.5;text-transform:uppercase}}

/* Bullets */
.bullet{{display:grid;grid-template-columns:32px 1fr;gap:14px;padding:15px 0;border-bottom:1px solid {BORDER};align-items:start}}
.bullet:last-child{{border-bottom:none}}
.bullet-icon{{
  width:32px;height:32px;border-radius:6px;display:flex;align-items:center;
  justify-content:center;font-size:13px;flex-shrink:0;
  background:rgba(26,143,255,0.1);border:1px solid rgba(26,143,255,0.2);
}}
.bullet-head{{font-size:13px;font-weight:600;color:{TEXT};margin-bottom:5px}}
.bullet-body{{font-size:12px;color:{MUTED};line-height:1.75;font-family:'IBM Plex Sans',sans-serif}}

/* Step grid */
.step-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:2rem}}
.step-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:8px;
  padding:26px 26px 22px;position:relative;overflow:hidden;
  transition:border-color 0.2s;
}}
.step-card:hover{{border-color:{BORDER2}}}
.step-bg{{
  position:absolute;top:8px;right:16px;font-size:72px;font-weight:700;
  color:{BORDER};line-height:1;pointer-events:none;user-select:none;
}}
.step-tag{{font-size:8px;letter-spacing:0.14em;text-transform:uppercase;color:{BLUE};margin-bottom:10px}}
.step-title{{font-size:15px;font-weight:600;color:{TEXT};margin-bottom:9px}}
.step-body{{font-size:12px;color:{MUTED};line-height:1.8;font-family:'IBM Plex Sans',sans-serif;position:relative;z-index:1}}

/* Timeline */
.tl{{display:flex;border:1px solid {BORDER};border-radius:8px;overflow:hidden;margin-bottom:2rem}}
.tl-cell{{flex:1;padding:18px 22px;border-right:1px solid {BORDER};position:relative}}
.tl-cell:last-child{{border-right:none}}
.tl-cell::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px}}
.tl-month{{font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};margin-bottom:10px}}
.tl-desc{{font-size:11px;color:{TEXT};line-height:1.65;font-family:'IBM Plex Sans',sans-serif}}

/* Alerts */
.alert{{border-radius:8px;padding:14px 18px;font-size:11px;line-height:1.75;margin:1rem 0;display:flex;gap:12px;align-items:flex-start;font-family:'IBM Plex Sans',sans-serif}}
.alert-warn{{background:rgba(255,61,92,0.06);border:1px solid rgba(255,61,92,0.18);color:rgba(255,175,185,0.9)}}
.alert-info{{background:rgba(26,143,255,0.06);border:1px solid rgba(26,143,255,0.2);color:rgba(150,195,255,0.9)}}
.alert-icon{{font-size:14px;flex-shrink:0;margin-top:1px}}

/* Insight callout block */
.callout{{
  background:{CARD2};border:1px solid {BORDER2};border-radius:8px;
  padding:20px 22px;margin-bottom:1.5rem;
  display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:start;
}}
.callout-icon{{
  width:36px;height:36px;border-radius:7px;display:flex;align-items:center;
  justify-content:center;font-size:16px;flex-shrink:0;
  background:rgba(26,143,255,0.12);border:1px solid rgba(26,143,255,0.22);
}}
.callout-head{{font-size:12px;font-weight:600;color:{TEXT};margin-bottom:6px}}
.callout-body{{font-size:11px;color:{MUTED};line-height:1.75;font-family:'IBM Plex Sans',sans-serif}}
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _pp(v):
    return float(str(v).replace("$","").strip()) if not pd.isna(v) else np.nan
def _pv(v):
    if pd.isna(v): return np.nan
    s = str(v).replace("$","").strip().lower()
    return float(s[:-1])*1e6 if s.endswith("m") else float(s[:-1])*1e3 if s.endswith("k") else float(s)

@st.cache_data
def load_data():
    df = pd.read_csv("price_history_filtered.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d %Y")
    df = df[(df["Date"]>="2023-06-01")&(df["Date"]<="2023-08-31")].sort_values("Date").reset_index(drop=True)
    for c in ["Open","High","Low","Close"]: df[c]=df[c].apply(_pp).astype(float)
    df["Volume"]=df["Volume"].apply(_pv).astype(float)
    df["Volume_NRK"]=df["Volume(NRK)"].astype(float)
    df["High"]=df["High"].fillna(df["Close"]); df["Low"]=df["Low"].fillna(df["Close"])
    return df.reset_index(drop=True)

@st.cache_data
def engineer(_df):
    df=_df.copy(); c=df["Close"]; sc_=c.shift(1)
    delta=sc_.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    df["MA3"]=sc_.rolling(3).mean(); df["MA7"]=sc_.rolling(7).mean(); df["MA14"]=sc_.rolling(14).mean()
    df["EMA7"]=sc_.ewm(span=7,adjust=False).mean(); df["EMA14"]=sc_.ewm(span=14,adjust=False).mean()
    df["Vol7"]=sc_.rolling(7).std(); df["Ret1"]=sc_.pct_change(1); df["Ret3"]=sc_.pct_change(3)
    df["RSI14"]=100-(100/(1+gain/(loss+1e-10)))
    df["HLSpread"]=df["High"].shift(1)-df["Low"].shift(1)
    for lag in range(1,8): df[f"Lag{lag}"]=c.shift(lag)
    df["DayOfWeek"]=df["Date"].dt.dayofweek; df["Month"]=df["Date"].dt.month
    df["VolMA7"]=df["Volume"].shift(1).rolling(7).mean()
    FEATS=["MA3","MA7","MA14","EMA7","EMA14","Vol7","Ret1","Ret3","RSI14",
           "HLSpread","Lag1","Lag2","Lag3","Lag4","Lag5","Lag6","Lag7","DayOfWeek","Month","VolMA7"]
    df_f=df.dropna(subset=FEATS+["Close"]).copy().reset_index(drop=True)
    for col in FEATS: df_f[col]=df_f[col].astype("float64")
    return df_f, FEATS

@st.cache_data
def train(_df_feat, feat_cols):
    n=len(_df_feat); split=int(n*0.80)
    tr=_df_feat.iloc[:split]; te=_df_feat.iloc[split:]
    Xtr=tr[feat_cols].values; ytr=tr["Close"].values
    Xte=te[feat_cols].values; yte=te["Close"].values
    sc=MinMaxScaler(); Xtr_sc=sc.fit_transform(Xtr); Xte_sc=sc.transform(Xte)
    ridge=Ridge(alpha=0.5); gb=GradientBoostingRegressor(n_estimators=300,learning_rate=0.03,max_depth=2,subsample=0.7,min_samples_leaf=2,random_state=42)
    ridge.fit(Xtr_sc,ytr); gb.fit(Xtr_sc,ytr)
    ypr=ridge.predict(Xte_sc); ypg=gb.predict(Xte_sc)
    ypn=np.concatenate([[ytr[-1]],yte[:-1]])
    tscv=TimeSeriesSplit(n_splits=4)
    def cv_mae(m):
        s=[]
        for ti,vi in tscv.split(Xtr_sc):
            m.fit(Xtr_sc[ti],ytr[ti]); s.append(mean_absolute_error(ytr[vi],m.predict(Xtr_sc[vi])))
        return float(np.mean(s)),float(np.std(s))
    def met(yt,yp):
        return dict(mae=mean_absolute_error(yt,yp),rmse=float(np.sqrt(mean_squared_error(yt,yp))),
                    da=int(np.sum(np.sign(np.diff(yt))==np.sign(np.diff(yp)))))
    return dict(
        ridge=ridge,gb=gb,scaler=sc,train=tr,test=te,
        y_test=yte,y_pred_ridge=ypr,y_pred_gb=ypg,y_pred_naive=ypn,
        m_ridge=met(yte,ypr),m_gb=met(yte,ypg),m_naive=met(yte,ypn),
        cv_ridge=cv_mae(Ridge(alpha=0.5)),
        cv_gb=cv_mae(GradientBoostingRegressor(n_estimators=300,learning_rate=0.03,max_depth=2,subsample=0.7,min_samples_leaf=2,random_state=42)),
        n_test=len(yte),
        feat_importances=dict(zip(feat_cols,gb.feature_importances_)),
    )

@st.cache_data
def forecast(_df,_results,feat_cols,days=30):
    df=_df.copy(); ridge=_results["ridge"]; sc=_results["scaler"]
    hist=list(df["Close"].values); highs=list(df["High"].values)
    lows=list(df["Low"].values); vols=list(df["Volume"].values)
    last_d=pd.Timestamp(df["Date"].iloc[-1]); preds=[]
    for i in range(days):
        c_s=pd.Series(hist); nd=last_d+pd.Timedelta(days=i+1)
        delta=c_s.diff(); gain=delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss=(-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        row=[c_s.rolling(3).mean().iloc[-1],c_s.rolling(7).mean().iloc[-1],c_s.rolling(14).mean().iloc[-1],
             c_s.ewm(span=7,adjust=False).mean().iloc[-1],c_s.ewm(span=14,adjust=False).mean().iloc[-1],
             c_s.rolling(7).std().iloc[-1],c_s.pct_change(1).iloc[-1],c_s.pct_change(3).iloc[-1],
             100-100/(1+gain/(loss+1e-10)),highs[-1]-lows[-1]]+\
            [hist[-j] for j in range(1,8)]+[nd.dayofweek,nd.month,float(np.mean(vols[-7:]))]
        p=float(ridge.predict(sc.transform(np.array(row,dtype="float64").reshape(1,-1)))[0])
        preds.append((nd,p)); hist.append(p); highs.append(p*1.01); lows.append(p*0.99); vols.append(float(np.mean(vols[-7:])))
    fcp=[x[1] for x in preds]; rs=float(pd.Series(df["Close"].values).rolling(7).std().iloc[-1])
    return dict(dates=[x[0] for x in preds],prices=fcp,upper=[p+1.5*rs for p in fcp],lower=[p-1.5*rs for p in fcp],
                hist_dates=list(df["Date"]),hist_prices=list(df["Close"]))

def get_logo(h=34):
    p="nordek_logo.png"
    if os.path.exists(p):
        with open(p,"rb") as f: b64=base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{h}" style="display:block;object-fit:contain;"/>'
    return f"""<svg width="130" height="{h}" viewBox="0 0 130 {h}" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{BLUE}"/><stop offset="100%" stop-color="{CYAN}"/></linearGradient></defs>
      <rect x="0" y="5" width="24" height="24" rx="4" fill="url(#g)"/>
      <text x="5" y="22" font-family="IBM Plex Mono" font-size="10" font-weight="700" fill="{BG}">NRK</text>
      <text x="32" y="24" font-family="IBM Plex Mono" font-size="15" font-weight="700" fill="{TEXT}" letter-spacing="1.5">NORDEK</text>
    </svg>"""

def pb(title="",h=420):
    return dict(
        title=dict(text=title,font=dict(family="IBM Plex Mono",size=11,color=SUBTEXT),x=0,xanchor="left",pad=dict(l=6,t=4)),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono",size=10,color=MUTED),height=h,
        margin=dict(l=52,r=16,t=36,b=44),
        xaxis=dict(gridcolor=BORDER,linecolor=BORDER2,showgrid=True,tickfont=dict(size=9,color=SUBTEXT),zeroline=False),
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER2,showgrid=True,tickfont=dict(size=9,color=SUBTEXT),zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=MUTED),bordercolor=BORDER,borderwidth=1),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD2,bordercolor=BORDER2,font=dict(family="IBM Plex Mono",size=11,color=TEXT)),
    )

def sec(label): st.markdown(f'<div class="sec">{label}</div>',unsafe_allow_html=True)
def cc(fig):
    st.markdown('<div class="chart-card">',unsafe_allow_html=True)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)


# ── Load ──────────────────────────────────────────────────────────────────────
df_raw            = load_data()
df_feat,feat_cols = engineer(df_raw)
results           = train(df_feat,feat_cols)

open_p  = df_raw["Open"].iloc[0]
close_p = df_raw["Close"].iloc[-1]
high_p  = df_raw["High"].max()
low_p   = df_raw["Low"].min()
chg     = (close_p-open_p)/open_p*100
chg_col = GREEN if chg>=0 else RED
chg_sym = "▲" if chg>=0 else "▼"
avg_vol = df_raw["Volume"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — rich, single logo, data-dense
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Hero block — ONE logo, immersive
    st.markdown(f"""
    <div class="sb-hero">
      {get_logo(30)}
      <div class="sb-token-pill">
        <span class="sb-token-dot"></span>
        <span class="sb-token-label">NRK Token · Live</span>
      </div>
      <div class="sb-price-block">
        <div>
          <div class="sb-price-main">${close_p:.4f}</div>
          <div class="sb-price-chg {'up' if chg>=0 else 'dn'}">{chg_sym} {abs(chg):.2f}% &nbsp;·&nbsp; period</div>
        </div>
        <div style="text-align:right;font-size:9px;color:{SUBTEXT};letter-spacing:0.04em;line-height:1.9;">
          Jun – Aug 2023<br/>73 trading days
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # Live stats
    st.markdown(f"""
    <div class="sb-stats">
      <div class="sb-stat-row">
        <span class="sb-stat-label">HIGH</span>
        <span class="sb-stat-value up">${high_p:.4f}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-label">LOW</span>
        <span class="sb-stat-value dn">${low_p:.4f}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-label">AVG VOLUME</span>
        <span class="sb-stat-value">${avg_vol/1e6:.2f}M</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-label">MODEL</span>
        <span class="sb-stat-value" style="color:{BLUE};">Ridge · α=0.5</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-label">FEATURES</span>
        <span class="sb-stat-value">20 signals</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-label">LEAK-FREE</span>
        <span class="sb-stat-value" style="color:{GREEN};">✓ Verified</span>
      </div>
    </div>""",unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Navigation</div>',unsafe_allow_html=True)

    page = st.radio("nav",[
        "◈  My Work at Nordek",
        "◈  Price & Model Results",
        "◈  30-Day Forecast",
        "◈  How It Works",
    ],label_visibility="collapsed")

    # Footer
    st.markdown(f"""
    <div class="sb-footer">
      <div class="sb-footer-text">
        Applied ML internship project · Summer 2023.<br/>
        Blockchain / DeFi · Nordek Protocol.
      </div>
      <div class="sb-version">
        <span style="background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.25);
          color:{GREEN};padding:2px 7px;border-radius:3px;letter-spacing:0.06em;">v3</span>
        <span style="color:{SUBTEXT};">Leak-free · Ridge model</span>
      </div>
    </div>""",unsafe_allow_html=True)

_page = page.split("  ",1)[-1].strip()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — My Work at Nordek
# ══════════════════════════════════════════════════════════════════════════════
if _page == "My Work at Nordek":

    # Cinematic hero
    st.markdown(f"""
    <div class="pg-hero">
      <div class="pg-hero-grid">
        <div>
          <div class="pg-eyebrow">Portfolio · Summer 2023 · Blockchain / DeFi</div>
          <div class="pg-title">NRK Token<br/><em>Price Analysis</em></div>
          <div class="pg-desc">Applied ML internship at Nordek — collecting on-chain OHLCV data,
          engineering 20 leak-free predictive features, benchmarking Ridge Regression against
          Gradient Boosting with TimeSeriesSplit CV, and packaging everything into this dashboard.</div>
        </div>
        <div class="pg-hero-meta">
          <div class="pg-hero-badge">
            <div class="pg-badge-label">Period Return</div>
            <div class="pg-badge-val" style="color:{chg_col};">{chg_sym}{abs(chg):.2f}%</div>
            <div class="pg-badge-sub {'up' if chg>=0 else 'dn'}">${open_p:.4f} → ${close_p:.4f}</div>
          </div>
          <div style="margin-top:10px;font-size:9px;color:{SUBTEXT};text-align:right;letter-spacing:0.04em;line-height:1.9;">
            Jun 2023 – Aug 2023<br/>73 trading days<br/>5 OHLCV fields
          </div>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # Intern strip
    st.markdown(f"""
    <div class="intern-strip">
      <div class="ib"><div class="ib-text">Applied<br/>ML<br/>Intern<br/>2023</div></div>
      <div>
        <div style="font-size:19px;font-weight:700;color:{TEXT};margin-bottom:5px;letter-spacing:-0.01em;">
          Applied Machine Learning Intern
        </div>
        <div style="font-size:11px;color:{CYAN};margin-bottom:12px;letter-spacing:0.03em;">
          June – August 2023 &nbsp;·&nbsp; 3 months &nbsp;·&nbsp; Blockchain / DeFi
        </div>
        <div style="font-size:12px;color:{MUTED};line-height:1.8;font-family:'IBM Plex Sans',sans-serif;max-width:580px;">
          Collected and cleaned daily OHLCV data for the NRK token across a full summer window.
          Engineered 20 leak-free predictive signals, compared Ridge Regression against Gradient
          Boosting using 4-fold TimeSeriesSplit cross-validation, and packaged the full pipeline
          into this interactive analytics dashboard for both technical and non-technical audiences.
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # KPI strip
    st.markdown(f"""
    <div class="kpi-strip" style="grid-template-columns:repeat(5,1fr);">
      <div class="kpi"><div class="kpi-label">Period Open</div>
        <div class="kpi-value">${open_p:.4f}</div><div class="kpi-meta neu">1 Jun 2023</div></div>
      <div class="kpi"><div class="kpi-label">Period Close</div>
        <div class="kpi-value">${close_p:.4f}</div>
        <div class="kpi-meta {'up' if chg>=0 else 'dn'}">{chg_sym} {abs(chg):.2f}%</div></div>
      <div class="kpi"><div class="kpi-label">Highest Price</div>
        <div class="kpi-value">${high_p:.4f}</div><div class="kpi-meta up">▲ Peak</div></div>
      <div class="kpi"><div class="kpi-label">Lowest Price</div>
        <div class="kpi-value">${low_p:.4f}</div><div class="kpi-meta dn">▼ Trough</div></div>
      <div class="kpi"><div class="kpi-label">Trading Days</div>
        <div class="kpi-value">{len(df_raw)}</div><div class="kpi-meta neu">Jun–Aug 2023</div></div>
    </div>""",unsafe_allow_html=True)

    # Candlestick + volume
    sec("Price History · Candlestick + Moving Averages + Volume")
    ma7=df_raw["Close"].rolling(7).mean(); ma14=df_raw["Close"].rolling(14).mean()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df_raw["Date"],open=df_raw["Open"],high=df_raw["High"],
        low=df_raw["Low"],close=df_raw["Close"],name="NRK/USD",
        increasing=dict(line=dict(color=GREEN,width=1),fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,width=1),fillcolor=RED)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df_raw["Date"],y=ma7,name="MA 7",
        line=dict(color=BLUE,width=1.3,dash="dot"),opacity=0.9),row=1,col=1)
    fig.add_trace(go.Scatter(x=df_raw["Date"],y=ma14,name="MA 14",
        line=dict(color=AMBER,width=1.3,dash="dash"),opacity=0.9),row=1,col=1)
    vc=[GREEN if df_raw["Close"].iloc[i]>=df_raw["Open"].iloc[i] else RED for i in range(len(df_raw))]
    fig.add_trace(go.Bar(x=df_raw["Date"],y=df_raw["Volume"],marker_color=vc,opacity=0.65,name="Volume"),row=2,col=1)
    base=pb(h=520)
    fig.update_layout(**base,xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=9,color=SUBTEXT)),
        yaxis=dict(tickprefix="$",gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=10,color=SUBTEXT)),
        yaxis2=dict(tickformat=".2s",gridcolor=BORDER,linecolor=BORDER2,tickfont=dict(size=9,color=SUBTEXT)))
    cc(fig)

    sec("Deliverables · What I Built")
    for icon,title,body in [
        ("📥","Data Collection & Cleaning","73 days of daily OHLCV data. Parsed dollar signs, K/M volume suffixes, validated consistency across the full Jun–Aug 2023 window."),
        ("⚙️","20 Leak-Free Features","All rolling signals (MA, EMA, RSI, volatility) are shifted 1 day forward — no look-ahead bias. A common mistake fixed in this pipeline."),
        ("🤖","Model Training & CV Evaluation","Ridge vs Gradient Boosting with 4-fold TimeSeriesSplit CV. Ridge wins on this small dataset — regularisation beats complexity at 46 training rows."),
        ("📊","Interactive Dashboard","Full pipeline in Streamlit — candlestick charts, model comparison table, CV scores, feature importance, 30-day forecast with confidence band."),
    ]:
        st.markdown(f"""<div class="bullet">
          <div class="bullet-icon">{icon}</div>
          <div><div class="bullet-head">{title}</div><div class="bullet-body">{body}</div></div>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price & Model Results
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "Price & Model Results":

    mr=results["m_ridge"]; mg=results["m_gb"]; mn=results["m_naive"]
    nte=results["n_test"]; cvr=results["cv_ridge"]; cvg=results["cv_gb"]
    yte=results["y_test"]; ypr=results["y_pred_ridge"]
    ypg=results["y_pred_gb"]; ypn=results["y_pred_naive"]
    tdf=results["test"]
    best="ridge" if mr["mae"]<=mg["mae"] else "gb"
    bm=mr if best=="ridge" else mg
    pct_naive=(mn["mae"]-bm["mae"])/mn["mae"]*100
    dir_pct=bm["da"]/(nte-1)*100

    st.markdown(f"""
    <div class="pg-hero">
      <div class="pg-hero-grid">
        <div>
          <div class="pg-eyebrow">Evaluation · Leak-free features · 4-fold TimeSeriesSplit CV</div>
          <div class="pg-title">Price &amp;<br/><em>Model Results</em></div>
          <div class="pg-desc">Ridge Regression vs Gradient Boosting vs naive baseline —
          MAE, RMSE, direction accuracy, and cross-validation on 73 rows of NRK price data.</div>
        </div>
        <div class="pg-hero-meta">
          <div class="pg-hero-badge">
            <div class="pg-badge-label">Best Model MAE</div>
            <div class="pg-badge-val" style="color:{AMBER};">${bm['mae']:.5f}</div>
            <div class="pg-badge-sub" style="color:{GREEN if pct_naive>0 else RED};">
              {'+' if pct_naive>0 else ''}{pct_naive:.1f}% vs naive
            </div>
          </div>
          <div style="margin-top:10px;font-size:9px;color:{SUBTEXT};text-align:right;letter-spacing:0.04em;line-height:1.9;">
            Test window · {nte} days<br/>Dir acc · {bm['da']}/{nte-1} ({dir_pct:.0f}%)<br/>Models · Ridge · GB
          </div>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # 3 metric cards
    c1,c2,c3=st.columns(3)
    dc=GREEN if dir_pct>=55 else AMBER if dir_pct>=45 else RED
    with c1:
        st.markdown(f"""<div class="m-card"><div class="m-label">Direction Accuracy · Best Model</div>
          <div class="m-value" style="color:{dc};">{bm['da']}<span style="font-size:20px;color:{MUTED};font-weight:400;">/{nte-1}</span></div>
          <div class="m-desc">{dir_pct:.0f}% correct UP/DOWN calls on the held-out test window.
          Random guessing = 50%.</div></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="m-card"><div class="m-label">Mean Absolute Error · Ridge</div>
          <div class="m-value" style="color:{AMBER};">${bm['mae']:.5f}</div>
          <div class="m-desc">Average dollar gap between predicted and actual price
          on days the model had never seen.</div></div>""",unsafe_allow_html=True)
    with c3:
        vc=GREEN if pct_naive>0 else RED
        st.markdown(f"""<div class="m-card"><div class="m-label">Improvement vs Naive Baseline</div>
          <div class="m-value" style="color:{vc};">{pct_naive:+.1f}<span style="font-size:20px;font-weight:400;">%</span></div>
          <div class="m-desc">Carry-forward baseline MAE: ${mn['mae']:.5f}.
          Simpler model still needs to beat this to be useful.</div></div>""",unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)

    # Model comparison table
    sec("Model Comparison · Test Set + 4-fold CV")
    st.markdown(f"""
    <table class="model-table">
      <thead><tr>
        <th>Model</th><th>MAE ↓</th><th>RMSE ↓</th>
        <th>Direction Acc ↑</th><th>CV MAE (4-fold)</th><th>Notes</th>
      </tr></thead>
      <tbody>
        <tr class="dim">
          <td>Naive carry-forward</td>
          <td>${mn['mae']:.5f}</td><td>${mn['rmse']:.5f}</td><td>—</td><td>—</td>
          <td>Predict tomorrow = today</td>
        </tr>
        <tr class="{'winner' if best=='gb' else ''}">
          <td>Gradient Boosting <span class="badge badge-purple">300 trees</span></td>
          <td>${mg['mae']:.5f}</td><td>${mg['rmse']:.5f}</td>
          <td>{mg['da']}/{nte-1} &nbsp;<span style="color:{SUBTEXT};">({mg['da']/(nte-1)*100:.0f}%)</span></td>
          <td>{cvg[0]:.5f} <span style="color:{SUBTEXT};">± {cvg[1]:.5f}</span></td>
          <td>High variance on small data</td>
        </tr>
        <tr class="{'winner' if best=='ridge' else ''}">
          <td>Ridge Regression <span class="badge {'badge-green' if best=='ridge' else 'badge-blue'}">{'★ best' if best=='ridge' else 'α=0.5'}</span></td>
          <td>${mr['mae']:.5f}</td><td>${mr['rmse']:.5f}</td>
          <td>{mr['da']}/{nte-1} &nbsp;<span style="color:{SUBTEXT};">({mr['da']/(nte-1)*100:.0f}%)</span></td>
          <td>{cvr[0]:.5f} <span style="color:{SUBTEXT};">± {cvr[1]:.5f}</span></td>
          <td>Regularisation wins at N=46</td>
        </tr>
      </tbody>
    </table>""",unsafe_allow_html=True)

    # Insight callout
    st.markdown(f"""
    <div class="callout">
      <div class="callout-icon">💡</div>
      <div>
        <div class="callout-head">Why Ridge beats Gradient Boosting on this dataset</div>
        <div class="callout-body">
          With only ~46 training samples, Gradient Boosting's 300 trees memorise noise rather than
          learning signal — its CV std is ±{cvg[1]:.5f} vs Ridge's ±{cvr[1]:.5f}.
          Ridge's L2 regularisation shrinks noisy coefficients toward zero, giving stable,
          generalisable predictions. This is a textbook small-data result: simpler models win.
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # Actual vs predicted
    sec("Actual vs Predicted · Test Window")
    fig_p=go.Figure()
    fig_p.add_trace(go.Scatter(x=list(tdf["Date"])+list(tdf["Date"])[::-1],
        y=list(yte)+list(ypr)[::-1],fill="toself",fillcolor="rgba(26,143,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),showlegend=False,hoverinfo="skip"))
    fig_p.add_trace(go.Scatter(x=tdf["Date"],y=yte,name="Actual",
        line=dict(color=CYAN,width=2.5),mode="lines+markers",
        marker=dict(size=5,color=CYAN,line=dict(width=1.5,color=BG))))
    fig_p.add_trace(go.Scatter(x=tdf["Date"],y=ypr,name="Ridge (best)",
        line=dict(color=BLUE,width=2,dash="dot")))
    fig_p.add_trace(go.Scatter(x=tdf["Date"],y=ypg,name="Gradient Boosting",
        line=dict(color=PURPLE,width=1.5,dash="dot"),opacity=0.75))
    fig_p.add_trace(go.Scatter(x=tdf["Date"],y=ypn,name="Naive baseline",
        line=dict(color=AMBER,width=1.2,dash="dash"),opacity=0.55))
    fig_p.update_layout(**pb(h=380))
    fig_p.update_yaxes(tickprefix="$")
    cc(fig_p)

    cA,cB=st.columns(2)
    with cA:
        sec("Feature Importance · GB Model")
        lm={"Lag1":"Yesterday close","Lag2":"2-day lag","Lag3":"3-day lag",
            "Lag4":"4-day lag","Lag5":"5-day lag","Lag6":"6-day lag","Lag7":"7-day lag",
            "MA3":"MA 3d","MA7":"MA 7d","MA14":"MA 14d","EMA7":"EMA 7d","EMA14":"EMA 14d",
            "Vol7":"Volatility 7d","Ret1":"1-day return","Ret3":"3-day return",
            "RSI14":"RSI 14","HLSpread":"Daily range","DayOfWeek":"Day of week",
            "Month":"Month","VolMA7":"Vol MA 7d"}
        fi_df=(pd.DataFrame(list(results["feat_importances"].items()),columns=["feat","imp"])
               .sort_values("imp",ascending=False).head(8))
        fi_df["label"]=fi_df["feat"].map(lm).fillna(fi_df["feat"])
        fig_fi=go.Figure(go.Bar(x=fi_df["imp"],y=fi_df["label"],orientation="h",
            marker=dict(color=fi_df["imp"],colorscale=[[0,BORDER2],[1,CYAN]],showscale=False,line=dict(width=0)),
            text=[f"{v:.3f}" for v in fi_df["imp"]],textposition="outside",
            textfont=dict(size=9,color=SUBTEXT,family="IBM Plex Mono")))
        fig_fi.update_layout(**pb(h=310))
        fig_fi.update_xaxes(title_text="Importance score")
        cc(fig_fi)
    with cB:
        sec("Residuals · Ridge Model")
        res=yte-ypr; rc=[GREEN if r>=0 else RED for r in res]
        fig_r=go.Figure()
        fig_r.add_hline(y=0,line_color=SUBTEXT,line_dash="dot",line_width=1)
        fig_r.add_trace(go.Bar(x=tdf["Date"],y=res,marker_color=rc,marker_line_width=0,opacity=0.85,name="Error"))
        fig_r.update_layout(**pb("green = under-predicted · red = over-predicted",h=310))
        fig_r.update_yaxes(tickprefix="$")
        cc(fig_r)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — 30-Day Forecast
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "30-Day Forecast":

    fc=forecast(df_raw,results,feat_cols,days=30)
    fca=np.array(fc["prices"]); fcs,fce=fca[0],fca[-1]
    fcmax,fcmin=fca.max(),fca.min(); fcp=(fce-fcs)/fcs*100
    fcsym="▲" if fcp>=0 else "▼"; fccol=GREEN if fcp>=0 else RED

    st.markdown(f"""
    <div class="pg-hero">
      <div class="pg-hero-grid">
        <div>
          <div class="pg-eyebrow">Projection · From 31 Aug 2023 · Portfolio demo only</div>
          <div class="pg-title">30-Day<br/><em>Forecast</em></div>
          <div class="pg-desc">Ridge Regression rolled forward 30 days using autoregressive
          features. Confidence band = ±1.5σ from 7-day rolling volatility at period end.</div>
        </div>
        <div class="pg-hero-meta">
          <div class="pg-hero-badge">
            <div class="pg-badge-label">Projected Change</div>
            <div class="pg-badge-val" style="color:{fccol};">{fcsym}{abs(fcp):.2f}%</div>
            <div class="pg-badge-sub" style="color:{fccol};">${fcs:.4f} → ${fce:.4f}</div>
          </div>
          <div style="margin-top:10px;font-size:9px;color:{SUBTEXT};text-align:right;letter-spacing:0.04em;line-height:1.9;">
            High · ${fcmax:.4f}<br/>Low &nbsp;· ${fcmin:.4f}<br/>Band · ±1.5σ
          </div>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-strip" style="grid-template-columns:repeat(4,1fr);">
      <div class="kpi"><div class="kpi-label">Forecast Start</div>
        <div class="kpi-value">${fcs:.4f}</div><div class="kpi-meta neu">1 Sep 2023</div></div>
      <div class="kpi"><div class="kpi-label">Forecast End</div>
        <div class="kpi-value">${fce:.4f}</div>
        <div class="kpi-meta {'up' if fcp>=0 else 'dn'}">{fcsym} {abs(fcp):.2f}%</div></div>
      <div class="kpi"><div class="kpi-label">Projected High</div>
        <div class="kpi-value">${fcmax:.4f}</div><div class="kpi-meta up">▲ Upper peak</div></div>
      <div class="kpi"><div class="kpi-label">Projected Low</div>
        <div class="kpi-value">${fcmin:.4f}</div><div class="kpi-meta dn">▼ Lower trough</div></div>
    </div>""",unsafe_allow_html=True)

    sec("Full History + 30-Day Outlook · Ridge Model")
    fig_fc=go.Figure()
    fig_fc.add_trace(go.Scatter(x=fc["hist_dates"],y=fc["hist_prices"],name="Historical",
        line=dict(color=CYAN,width=2),fill="tozeroy",fillcolor="rgba(0,207,255,0.04)"))
    fig_fc.add_trace(go.Scatter(
        x=fc["dates"]+fc["dates"][::-1],y=fc["upper"]+fc["lower"][::-1],
        fill="toself",fillcolor="rgba(26,143,255,0.09)",
        line=dict(color="rgba(0,0,0,0)"),name="±1.5σ band"))
    fig_fc.add_trace(go.Scatter(x=fc["dates"],y=fc["prices"],name="Forecast (Ridge)",
        line=dict(color=BLUE,width=2,dash="dot"),mode="lines+markers",
        marker=dict(size=3.5,color=BLUE)))
    fig_fc.add_vline(x=pd.Timestamp("2023-08-31"),line_color=AMBER,line_dash="dash",
        line_width=1.2,annotation_text="internship end",
        annotation_font_color=AMBER,annotation_font_size=10,
        annotation_font_family="IBM Plex Mono",annotation_position="top right")
    fig_fc.update_layout(**pb(h=460))
    fig_fc.update_yaxes(tickprefix="$")
    cc(fig_fc)

    st.markdown(f"""
    <div class="alert alert-warn">
      <span class="alert-icon">⚠</span>
      <span><strong>Disclaimer:</strong> Portfolio demonstration only. Nordek ceased operations
      after August 2023 and the NRK token is no longer actively traded. Not financial advice.
      Autoregressive forecasts degrade rapidly beyond ~5 steps — treat this as a trajectory
      illustration, not a reliable price target.</span>
    </div>""",unsafe_allow_html=True)

    sec("Day-by-Day Forecast Table")
    fc_table=pd.DataFrame({
        "Date":[d.strftime("%d %b %Y") for d in fc["dates"]],
        "Projected Price":[f"${p:.5f}" for p in fc["prices"]],
        "Upper (+1.5σ)":[f"${u:.5f}" for u in fc["upper"]],
        "Lower (−1.5σ)":[f"${l:.5f}" for l in fc["lower"]],
    })
    st.dataframe(fc_table,hide_index=True,use_container_width=True)
    st.download_button("⬇  Export forecast as CSV",
        data=fc_table.to_csv(index=False).encode(),
        file_name="nrk_30day_forecast.csv",mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "How It Works":

    st.markdown(f"""
    <div class="pg-hero">
      <div class="pg-hero-grid">
        <div>
          <div class="pg-eyebrow">Methodology · Plain-English Walkthrough</div>
          <div class="pg-title">How It<br/><em>Works</em></div>
          <div class="pg-desc">A non-technical breakdown of the data pipeline, leak-free
          feature engineering, model selection rationale, and honest evaluation methodology.</div>
        </div>
        <div class="pg-hero-meta">
          <div class="pg-hero-badge">
            <div class="pg-badge-label">Pipeline Steps</div>
            <div class="pg-badge-val" style="color:{BLUE};">4</div>
            <div class="pg-badge-sub neu">Data → Features → Model → Story</div>
          </div>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    sec("Project Steps")
    st.markdown('<div class="step-grid">',unsafe_allow_html=True)
    for num,tag,title,body in [
        ("01","STEP 1","Gathering the Data",
         "73 days of daily OHLCV data across Jun–Aug 2023. Raw quirks like dollar signs and K/M volume suffixes were parsed. Dataset validated for consistency before any modelling."),
        ("02","STEP 2","Leak-Free Feature Engineering",
         "All 20 signals — rolling averages, EMA, RSI, volatility, returns — are shifted 1 day forward. This prevents look-ahead bias: a subtle but critical error that inflates apparent model performance."),
        ("03","STEP 3","Model Selection with CV",
         "Ridge and Gradient Boosting trained on 80% of data. 4-fold TimeSeriesSplit CV used to measure generalisation. Ridge won — at N=46 training rows, regularisation beats complexity."),
        ("04","STEP 4","Honest Evaluation",
         "Every model is benchmarked against a naive carry-forward baseline. CV variance is also reported — a low mean with high variance means the model got lucky on one fold, not genuinely good."),
    ]:
        st.markdown(f"""<div class="step-card">
          <div class="step-bg">{num}</div>
          <div class="step-tag">{tag}</div>
          <div class="step-title">{title}</div>
          <div class="step-body">{body}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    sec("Project Timeline · Summer 2023")
    st.markdown(f"""
    <div class="tl">
      <div class="tl-cell" style="--c:{BLUE};">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{BLUE};"></div>
        <div class="tl-month">June 2023</div>
        <div class="tl-desc">Data collection begins · baseline exploration · initial cleaning</div>
      </div>
      <div class="tl-cell">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{CYAN};"></div>
        <div class="tl-month">July 2023</div>
        <div class="tl-desc">Leak-free feature engineering · model architecture · first CV runs</div>
      </div>
      <div class="tl-cell">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{GREEN};"></div>
        <div class="tl-month">August 2023</div>
        <div class="tl-desc">Model evaluation · dashboard build · stakeholder demo · internship end</div>
      </div>
    </div>""",unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert alert-warn">
      <span class="alert-icon">⚠</span>
      <span><strong>73 rows is a small dataset.</strong> Most production financial ML systems
      use years of data. Results here demonstrate methodology, not a production trading signal.
      Direction accuracy on 11 test days is not statistically significant — ~100+ test points
      are needed to draw firm conclusions.</span>
    </div>""",unsafe_allow_html=True)

    sec("Tech Stack")
    cols=st.columns(5)
    for col,(name,role) in zip(cols,[("Python 3.11","Runtime"),("Streamlit","Dashboard"),
        ("scikit-learn","Ridge · GB · CV"),("Pandas / NumPy","Data pipeline"),("Plotly","Charts")]):
        with col:
            st.markdown(f"""<div class="kpi" style="text-align:center;padding:14px 10px;">
              <div class="kpi-label">{role}</div>
              <div style="font-size:11px;font-weight:600;color:{BLUE};margin-top:6px;">{name}</div>
            </div>""",unsafe_allow_html=True)
