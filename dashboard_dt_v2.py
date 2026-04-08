#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC-BTES Digital Twin — Dashboard di Live Monitoring
=====================================================
Avvio:  streamlit run dashboard_dt_v2.py
Poi aprire nel browser:  http://localhost:8501

Legge:
  - RTC_BTES_output_MPC/live_results.csv          (live, aggiornato ogni ora)
  - RTC_BTES_output_MPC/RTC_BTES_MPC_v2_*.xlsx   (output finale)
  - sensori_live.csv                               (dati sensori, opzionale)
  - Open-Meteo API                                 (previsioni 24h, opzionale)

Installazione dipendenze (una-tantum):
  pip install streamlit plotly openpyxl requests
"""

import os
import glob
import time as _time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

# =============================================================================
# CONFIGURAZIONE  ← modifica qui
# =============================================================================
OUTPUT_DIR    = "RTC_BTES_output_MPC"
SENSORI_CSV   = "sensori_live.csv"
REFRESH_SEC   = 60
LAT           = 38.10094
LON           = 13.34657
H_FORECAST    = 24
GIORNI_STORIA = 14

T_BTES_MIN    = 18.0
T_BTES_MAX    = 40.0
T_BTES_INDIST = 20.0

COLORI_MODO = {
    'charge'   : '#F5A623',
    'solar_dc' : '#2E86C1',
    'discharge': '#E74C3C',
    'idle'     : '#7F8C8D',
}

# =============================================================================
# LAYOUT PAGINA — configurazione base
# =============================================================================
st.set_page_config(
    page_title            = "SMARTEP Digital Twin",
    page_icon             = "🌡️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CSS GLOBALE — apparenza avanzata
# =============================================================================
st.markdown("""
<style>
/* ── Body / sfondo ───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #F0F4F8;
}
[data-testid="stSidebar"] {
    background-color: #1C2833;
    color: #ECF0F1;
}
[data-testid="stSidebar"] * {
    color: #ECF0F1 !important;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stCheckbox > label {
    color: #AED6F1 !important;
    font-size: 0.95rem;
}

/* ── Intestazione principale ─────────────────────────────────────────── */
.smartep-header {
    background: linear-gradient(135deg, #1A3A5C 0%, #1F618D 60%, #2980B9 100%);
    border-radius: 12px;
    padding: 28px 40px 20px 40px;
    margin-bottom: 6px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.smartep-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 0.06em;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.smartep-subtitle {
    font-size: 1.15rem;
    color: #AED6F1;
    margin-top: 4px;
    letter-spacing: 0.12em;
    font-weight: 400;
}
.smartep-timestamp {
    font-size: 0.82rem;
    color: #85C1E9;
    margin-top: 8px;
}

/* ── Sezioni ─────────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 5px solid #1F618D;
    padding-left: 12px;
    margin: 30px 0 12px 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1C2833;
}

/* ── KPI card ────────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
    text-align: center;
    height: 100%;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.0rem;
    font-weight: 800;
    color: #1A3A5C;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 0.82rem;
    color: #888;
    margin-top: 4px;
}

/* Colore badge modalità */
.badge-charge    { background:#FEF3E2; border-left: 4px solid #F5A623; }
.badge-solar_dc  { background:#EAF4FB; border-left: 4px solid #2E86C1; }
.badge-discharge { background:#FDEDEC; border-left: 4px solid #E74C3C; }
.badge-idle      { background:#F2F3F4; border-left: 4px solid #7F8C8D; }

/* ── Tabella ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INTESTAZIONE
# =============================================================================
st.markdown(f"""
<div class="smartep-header">
  <div class="smartep-title">SMARTEP Digital Twin</div>
  <div class="smartep-subtitle">RTC &nbsp;·&nbsp; BTES &nbsp;·&nbsp; Dry Cooler &nbsp;—&nbsp; Palermo</div>
  <div class="smartep-timestamp">🕐 Aggiornamento automatico ogni {REFRESH_SEC}s &nbsp;|&nbsp;
       Ultima lettura: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Impostazioni")
    giorni_vis    = st.slider("Giorni di storico", 1, GIORNI_STORIA, 7)
    mostra_soglie = st.checkbox("Mostra confronto MPC vs Soglie", value=True)
    mostra_forecast = st.checkbox("Mostra previsioni 24h", value=True)
    mostra_sensori  = st.checkbox("Mostra dati sensori", value=True)
    st.markdown("---")
    st.markdown("**Limiti operativi BTES**")
    st.markdown(f"- T min scarica: **{T_BTES_MIN} °C**")
    st.markdown(f"- T max carica:  **{T_BTES_MAX} °C**")
    st.markdown(f"- T indisturbata: **{T_BTES_INDIST} °C**")
    st.markdown("---")
    if st.button("🔄 Aggiorna ora"):
        st.rerun()

# helper: layout comune per tutti i grafici
def _layout(fig, title="", height=420, y_title="", legend_y=-0.18, **kw):
    fig.update_layout(
        title      = dict(text=title, font=dict(size=17, color="#1C2833"), x=0.02),
        height     = height,
        yaxis_title= y_title,
        margin     = dict(t=52, b=8, l=8, r=8),
        paper_bgcolor = "#FFFFFF",
        plot_bgcolor  = "#FAFBFC",
        legend        = dict(orientation="h", y=legend_y,
                             font=dict(size=13), bgcolor="rgba(0,0,0,0)"),
        font          = dict(size=13, color="#2C3E50"),
        **kw,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8EDF2",
                     tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_yaxes(showgrid=True, gridcolor="#E8EDF2",
                     tickfont=dict(size=12), title_font=dict(size=13))
    return fig

def _section(icon, text):
    st.markdown(f'<div class="section-header">{icon}&nbsp;{text}</div>',
                unsafe_allow_html=True)

# =============================================================================
# CARICAMENTO DATI
# =============================================================================
@st.cache_data(ttl=REFRESH_SEC)
def carica_risultati_mpc() -> pd.DataFrame | None:
    live_path = os.path.join(OUTPUT_DIR, "live_results.csv")
    if os.path.exists(live_path):
        df    = pd.read_csv(live_path)
        fonte = "live"
    else:
        pattern = os.path.join(OUTPUT_DIR, "RTC_BTES_MPC_v2_*.xlsx")
        files   = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(OUTPUT_DIR, "RTC_BTES_MPC_v1_*.xlsx")
            files   = sorted(glob.glob(pattern))
        if not files:
            return None
        df    = pd.read_excel(files[-1])
        fonte = "excel"

    ref = datetime(2025, 8, 1)
    df["datetime"] = [ref + timedelta(hours=float(h)) for h in df["time_h"]]

    n_ore_sim = len(df)
    st.sidebar.metric("Ore simulate finora", n_ore_sim,
                      delta="in corso..." if fonte == "live" else "completato")
    if fonte == "live":
        mtime = datetime.fromtimestamp(os.path.getmtime(live_path)).strftime('%H:%M:%S')
        st.sidebar.markdown(f"🟢 **Live** — ultimo aggiornamento: `{mtime}`")
    else:
        st.sidebar.markdown("📁 Lettura da file finale Excel")
    return df


@st.cache_data(ttl=REFRESH_SEC)
def carica_sensori() -> pd.DataFrame | None:
    if not os.path.exists(SENSORI_CSV):
        return None
    try:
        df = pd.read_csv(SENSORI_CSV, sep=';', parse_dates=['datetime'])
        return df.sort_values('datetime').reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def carica_forecast() -> dict | None:
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT, "longitude": LON,
            "hourly"  : "temperature_2m,shortwave_radiation,wind_speed_10m,relative_humidity_2m",
            "forecast_days": 2, "timezone": "Europe/Rome", "models": "best_match",
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        d     = r.json()["hourly"]
        now_h = datetime.now().hour
        return {
            "ora"    : d["time"][now_h : now_h + H_FORECAST],
            "T_aria" : d["temperature_2m"][now_h : now_h + H_FORECAST],
            "Q_irr"  : d["shortwave_radiation"][now_h : now_h + H_FORECAST],
            "WSpeed" : d["wind_speed_10m"][now_h : now_h + H_FORECAST],
            "RH"     : d["relative_humidity_2m"][now_h : now_h + H_FORECAST],
        }
    except Exception:
        return None


# --- Caricamento ---
df_mpc = carica_risultati_mpc()
df_sen = carica_sensori() if mostra_sensori else None
fcast  = carica_forecast() if mostra_forecast else None

if df_mpc is None:
    st.error("⚠️  Nessun file di output MPC trovato in `RTC_BTES_output_MPC/`.\n"
             "Avvia prima `RTC_BTES_MPC_v2.py`.")
    st.stop()

t_max_dt  = df_mpc["datetime"].max()
t_min_vis = t_max_dt - timedelta(days=giorni_vis)
df_vis    = df_mpc[df_mpc["datetime"] >= t_min_vis].copy()

ultima     = df_mpc.iloc[-1]
T_btes_now = float(ultima["T_btes_C"])
mode_now   = str(ultima["mode"])
Q_rtc_now  = float(ultima["Q_rtc_W"])
Q_dc_now   = float(ultima["Q_dc_W"])
T_aria_now = float(ultima["T_aria_C"])

# =============================================================================
# RIGA 1 — KPI CARDS + GAUGE
# =============================================================================
_section("📊", "Stato corrente dell'impianto")

pct_btes   = max(0.0, min(100.0, (T_btes_now - T_BTES_MIN) / (T_BTES_MAX - T_BTES_MIN) * 100))
delta_btes = T_btes_now - T_BTES_INDIST
badge_cls  = f"kpi-card badge-{mode_now}" if mode_now in COLORI_MODO else "kpi-card"

col_gauge, col_kpis = st.columns([1, 3])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = T_btes_now,
        number= {"suffix": " °C", "font": {"size": 34, "color": "#1A3A5C"}},
        delta = {"reference": T_BTES_INDIST, "suffix": " °C",
                 "font": {"size": 16}},
        title = {"text": "T fluido BTES", "font": {"size": 17, "color": "#2C3E50"}},
        gauge = {
            "axis"  : {"range": [T_BTES_MIN - 3, T_BTES_MAX + 3],
                       "tickfont": {"size": 12}},
            "bar"   : {"color": COLORI_MODO.get(mode_now, "#888"), "thickness": 0.28},
            "bgcolor": "#F0F4F8",
            "borderwidth": 0,
            "steps" : [
                {"range": [T_BTES_MIN - 3, T_BTES_MIN], "color": "#FADBD8"},
                {"range": [T_BTES_MIN, T_BTES_INDIST],  "color": "#D6EAF8"},
                {"range": [T_BTES_INDIST, T_BTES_MAX],  "color": "#D5F5E3"},
                {"range": [T_BTES_MAX, T_BTES_MAX + 3], "color": "#FADBD8"},
            ],
            "threshold": {"line": {"color": "#E74C3C", "width": 4},
                          "value": T_BTES_MAX},
        },
    ))
    fig_gauge.update_layout(height=310, margin=dict(t=30, b=10, l=20, r=20),
                            paper_bgcolor="#FFFFFF")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_kpis:
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    modo_label = mode_now.upper().replace("_", " ")
    modo_col   = COLORI_MODO.get(mode_now, "#888")

    with r1c1:
        st.markdown(f"""
        <div class="{badge_cls}">
          <div class="kpi-label">Modalità operativa</div>
          <div class="kpi-value" style="color:{modo_col}; font-size:1.6rem;">{modo_label}</div>
        </div>""", unsafe_allow_html=True)

    with r1c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">T fluido BTES</div>
          <div class="kpi-value">{T_btes_now:.1f} °C</div>
          <div class="kpi-delta">{'▲' if delta_btes>0 else '▼'} {delta_btes:+.1f}°C vs indisturbata</div>
        </div>""", unsafe_allow_html=True)

    with r1c3:
        arrow = "▲ carica" if mode_now == 'charge' else ("▼ scarica" if mode_now == 'discharge' else "─ stabile")
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Capacità BTES</div>
          <div class="kpi-value">{pct_btes:.0f} %</div>
          <div class="kpi-delta">{arrow}</div>
        </div>""", unsafe_allow_html=True)

    with r2c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Potenza RTC</div>
          <div class="kpi-value">{Q_rtc_now/1000:.2f} kW</div>
        </div>""", unsafe_allow_html=True)

    with r2c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Potenza Dry Cooler</div>
          <div class="kpi-value">{Q_dc_now/1000:.2f} kW</div>
        </div>""", unsafe_allow_html=True)

    with r2c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">T aria esterna</div>
          <div class="kpi-value">{T_aria_now:.1f} °C</div>
        </div>""", unsafe_allow_html=True)

# piccolo spazio
st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

# =============================================================================
# RIGA 2 — TEMPERATURE SISTEMA
# =============================================================================
_section("📈", f"Temperature sistema — ultimi {giorni_vis} giorni")

fig_T = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.06,
    subplot_titles=("Temperature [°C]", "Modalità operativa"),
)

# Linee temperature
for y_col, label, col, w, dash in [
    ("T_btes_C",   "T fluido BTES",   "#E67E22", 3.0, "solid"),
    ("Tw_out_C",   "T uscita RTC",    "#E74C3C", 2.5, "solid"),
    ("Tw_in_C",    "T ingresso RTC",  "#2980B9", 2.5, "solid"),
    ("T_aria_C",   "T aria esterna",  "#27AE60", 2.0, "dot"),
    ("T_dc_ret_C", "T ritorno DC",    "#8E44AD", 2.0, "dash"),
]:
    if y_col in df_vis.columns:
        fig_T.add_trace(go.Scatter(
            x=df_vis["datetime"], y=df_vis[y_col],
            name=label, line=dict(color=col, width=w, dash=dash),
            connectgaps=False), row=1, col=1)

# Linee riferimento BTES
for val, col, lbl in [
    (T_BTES_MIN,    "#2980B9", f"T min {T_BTES_MIN}°C"),
    (T_BTES_MAX,    "#E74C3C", f"T max {T_BTES_MAX}°C"),
    (T_BTES_INDIST, "#7F8C8D", f"T ind. {T_BTES_INDIST}°C"),
]:
    fig_T.add_hline(y=val, line_dash="dot", line_color=col,
                    annotation_text=lbl,
                    annotation_font=dict(size=12, color=col),
                    annotation_position="right", row=1, col=1)

# Sensori sovrapposti
if df_sen is not None and "T_btes_mis" in df_sen.columns:
    df_sen_vis = df_sen[df_sen["datetime"] >= t_min_vis]
    fig_T.add_trace(go.Scatter(
        x=df_sen_vis["datetime"], y=df_sen_vis["T_btes_mis"],
        name="T BTES misurata ★", mode="markers+lines",
        line=dict(color="#E67E22", width=1.5, dash="dash"),
        marker=dict(size=7, symbol="star")), row=1, col=1)

# Barre modalità operativa
for modo, col in COLORI_MODO.items():
    mask = df_vis["mode"] == modo
    if mask.any():
        fig_T.add_trace(go.Bar(
            x=df_vis.loc[mask, "datetime"], y=[1] * mask.sum(),
            name=modo, marker_color=col, opacity=0.85,
            showlegend=True), row=2, col=1)

_layout(fig_T, height=600, legend_y=-0.09)
fig_T.update_layout(barmode="stack")
fig_T.update_yaxes(title_text="Temperatura [°C]", title_font=dict(size=14), row=1, col=1)
fig_T.update_yaxes(title_text="Modo", tickvals=[0,1], ticktext=["",""], row=2, col=1)
for i in range(1, 3):
    fig_T.layout.annotations[i-1].update(font=dict(size=14, color="#2C3E50"))

st.plotly_chart(fig_T, use_container_width=True)

# =============================================================================
# RIGA 3 — POTENZE E BILANCIO ENERGETICO
# =============================================================================
_section("⚡", "Potenze e bilancio energetico")
col_L, col_R = st.columns(2)

with col_L:
    fig_P = go.Figure()
    fig_P.add_trace(go.Scatter(
        x=df_vis["datetime"], y=df_vis["Q_rtc_W"] / 1000,
        name="Q RTC [kW]", line=dict(color="#27AE60", width=2.5),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.18)"))
    fig_P.add_trace(go.Scatter(
        x=df_vis["datetime"], y=df_vis["Q_dc_W"] / 1000,
        name="Q Dry Cooler [kW]", line=dict(color="#E74C3C", width=2.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.18)"))
    fig_P.add_trace(go.Scatter(
        x=df_vis["datetime"], y=df_vis["Q_solar_W"] / 1000,
        name="Q solare incidente [kW]", line=dict(color="#F39C12", width=2.0, dash="dot")))
    _layout(fig_P, title="Potenze istantanee", height=420, y_title="Potenza [kW]")
    st.plotly_chart(fig_P, use_container_width=True)

with col_R:
    E_btes = np.cumsum([max(q, 0) / 1000 if m == 'charge' else 0
                        for q, m in zip(df_vis["Q_rtc_W"], df_vis["mode"])])
    E_dc   = np.cumsum([q / 1000 if m in ('solar_dc', 'discharge') else 0
                        for q, m in zip(df_vis["Q_dc_W"], df_vis["mode"])])
    fig_E = go.Figure()
    fig_E.add_trace(go.Scatter(
        x=df_vis["datetime"], y=E_btes,
        name="E cumulativa BTES (carica)",
        line=dict(color="#E67E22", width=3),
        fill="tozeroy", fillcolor="rgba(230,126,34,0.15)"))
    fig_E.add_trace(go.Scatter(
        x=df_vis["datetime"], y=E_dc,
        name="E cumulativa Dry Cooler",
        line=dict(color="#E74C3C", width=3),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.12)"))
    _layout(fig_E, title="Energia cumulativa", height=420, y_title="Energia [kWh]")
    st.plotly_chart(fig_E, use_container_width=True)

# =============================================================================
# RIGA 4 — CONFRONTO MPC vs SOGLIE
# =============================================================================
if mostra_soglie and "mpc_gain" in df_vis.columns:
    _section("🔍", "Valore del MPC rispetto allo scheduler a soglie")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        cum_gain = np.cumsum(df_vis["mpc_gain"].fillna(0))
        colors_fill = ["rgba(26,82,118,0.25)" if g >= 0
                       else "rgba(231,76,60,0.25)" for g in cum_gain]
        fig_cg = go.Figure()
        fig_cg.add_trace(go.Scatter(
            x=df_vis["datetime"], y=cum_gain,
            name="Guadagno cumulativo MPC",
            line=dict(color="#1A5276", width=3),
            fill="tozeroy",
            fillcolor=colors_fill[0] if cum_gain.iloc[-1] >= 0
                      else "rgba(231,76,60,0.20)",
        ))
        fig_cg.add_hline(y=0, line_dash="dash", line_color="#E74C3C", line_width=2)
        val_fin = float(cum_gain.iloc[-1]) if len(cum_gain) > 0 else 0
        _layout(fig_cg,
                title=f"Guadagno cumulativo MPC vs Soglie  (finale: {val_fin:+.2f} kWh-eq)",
                height=380, y_title="kWh-eq")
        st.plotly_chart(fig_cg, use_container_width=True)

        # Istogramma guadagno orario (sotto il cumulativo)
        fig_hist = px.histogram(df_vis, x="mpc_gain", nbins=30,
            color_discrete_sequence=["#2E86C1"],
            labels={"mpc_gain": "Guadagno orario [kWh-eq]"},
            title="Distribuzione guadagno orario MPC vs Soglie")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#E74C3C", line_width=2)
        _layout(fig_hist, height=320, y_title="Conteggio ore")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        if "mpc_changed" in df_vis.columns:
            n_div = int(df_vis["mpc_changed"].sum())
            n_tot = len(df_vis)
            g_pos = float(df_vis.loc[df_vis["mpc_gain"] > 0, "mpc_gain"].sum())
            g_neg = float(df_vis.loc[df_vis["mpc_gain"] < 0, "mpc_gain"].sum())

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Ore con scelta diversa dalle soglie",
                      f"{n_div} / {n_tot}",
                      delta=f"{100*n_div/n_tot:.1f}%")
            st.metric("Guadagno reward positivo",  f"+{g_pos:.2f} kWh-eq")
            st.metric("Guadagno reward negativo",  f"{g_neg:.2f} kWh-eq")
            st.metric("Guadagno netto",            f"{g_pos+g_neg:+.2f} kWh-eq",
                      delta_color="normal")

# =============================================================================
# RIGA 5 — PREVISIONI METEO 24H
# =============================================================================
if mostra_forecast and fcast is not None:
    _section("☀️", "Previsioni meteo prossime 24h  —  Open-Meteo")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fc.add_trace(go.Bar(
            x=fcast["ora"], y=fcast["Q_irr"],
            name="Irraggiamento [W/m²]",
            marker_color="#F39C12", opacity=0.75), secondary_y=False)
        fig_fc.add_trace(go.Scatter(
            x=fcast["ora"], y=fcast["T_aria"],
            name="T aria [°C]",
            line=dict(color="#E74C3C", width=3)), secondary_y=True)
        fig_fc.update_yaxes(title_text="W/m²", title_font=dict(size=13),
                            secondary_y=False)
        fig_fc.update_yaxes(title_text="°C", title_font=dict(size=13),
                            secondary_y=True)
        _layout(fig_fc, title="Irraggiamento e temperatura", height=360, legend_y=-0.25)
        st.plotly_chart(fig_fc, use_container_width=True)

    with col_f2:
        fig_rh = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rh.add_trace(go.Bar(
            x=fcast["ora"], y=fcast["RH"],
            name="Umidità relativa [%]",
            marker_color="#5DADE2", opacity=0.75), secondary_y=False)
        fig_rh.add_trace(go.Scatter(
            x=fcast["ora"], y=fcast["WSpeed"],
            name="Vento [m/s]",
            line=dict(color="#8E44AD", width=3)), secondary_y=True)
        fig_rh.update_yaxes(title_text="%",   title_font=dict(size=13),
                            secondary_y=False)
        fig_rh.update_yaxes(title_text="m/s", title_font=dict(size=13),
                            secondary_y=True)
        _layout(fig_rh, title="Umidità relativa e vento", height=360, legend_y=-0.25)
        st.plotly_chart(fig_rh, use_container_width=True)

elif mostra_forecast and fcast is None:
    st.info("⚠️ Previsioni Open-Meteo non disponibili (nessuna connessione internet?).")

# =============================================================================
# RIGA 6 — CONFRONTO MODELLO vs SENSORI
# =============================================================================
if mostra_sensori and df_sen is not None:
    _section("🔬", "Confronto modello vs misure sensori")

    df_sen_h = df_sen.copy()
    df_sen_h["datetime"] = df_sen_h["datetime"].dt.floor("h")
    df_merge = pd.merge(
        df_vis[["datetime", "T_btes_C"]],
        df_sen_h[["datetime", "T_btes_mis"]],
        on="datetime", how="inner")

    if len(df_merge) > 0:
        errore = df_merge["T_btes_mis"] - df_merge["T_btes_C"]

        fig_err = make_subplots(rows=1, cols=2,
            subplot_titles=("T_btes: Modello vs Misurata [°C]",
                            "Errore di stima (mis − mod) [°C]"),
            horizontal_spacing=0.10)

        fig_err.add_trace(go.Scatter(
            x=df_merge["datetime"], y=df_merge["T_btes_C"],
            name="Modello",
            line=dict(color="#E67E22", width=2.5)), row=1, col=1)
        fig_err.add_trace(go.Scatter(
            x=df_merge["datetime"], y=df_merge["T_btes_mis"],
            name="Misurata", mode="markers",
            marker=dict(color="#8E44AD", size=7, symbol="star")), row=1, col=1)
        fig_err.add_trace(go.Bar(
            x=df_merge["datetime"], y=errore,
            name="Errore",
            marker_color=["#E74C3C" if e > 0 else "#2980B9" for e in errore]),
            row=1, col=2)
        fig_err.add_hline(y=0, row=1, col=2,
                          line_dash="dash", line_color="black", line_width=1.5)

        _layout(fig_err, height=400)
        for ann in fig_err.layout.annotations:
            ann.update(font=dict(size=14, color="#2C3E50"))
        st.plotly_chart(fig_err, use_container_width=True)

        rmse = float(np.sqrt(np.mean(errore**2)))
        bias = float(errore.mean())
        ce1, ce2, ce3 = st.columns(3)
        ce1.metric("RMSE modello-misura", f"{rmse:.2f} °C")
        ce2.metric("Bias sistematico",    f"{bias:+.2f} °C")
        ce3.metric("N ore confrontate",   f"{len(df_merge)}")
    else:
        st.info("Nessuna sovrapposizione temporale tra modello e sensori.")

elif mostra_sensori and df_sen is None:
    st.info(f"ℹ️  File sensori `{SENSORI_CSV}` non trovato. "
            "Quando i sensori saranno collegati, il file verrà letto automaticamente.")

# =============================================================================
# RIGA 7 — TABELLA ULTIME 48 ORE
# =============================================================================
_section("📋", "Tabella valori — ultime 48 ore")
with st.expander("Espandi tabella"):
    cols_tab = ["datetime", "mode", "T_btes_C", "T_aria_C",
                "Q_rtc_W", "Q_dc_W", "n_iter"]
    for c in ["mpc_best_reward", "threshold_mode", "mpc_gain"]:
        if c in df_mpc.columns:
            cols_tab.append(c)

    df_tab = df_mpc.tail(48)[cols_tab].copy()
    df_tab["Q_rtc_kW"] = (df_tab["Q_rtc_W"] / 1000).round(2)
    df_tab["Q_dc_kW"]  = (df_tab["Q_dc_W"]  / 1000).round(2)
    df_tab["T_btes_C"] = df_tab["T_btes_C"].round(2)
    if "mpc_gain" in df_tab.columns:
        df_tab["mpc_gain"] = df_tab["mpc_gain"].round(3)
    df_tab = df_tab.drop(columns=["Q_rtc_W", "Q_dc_W"])

    def colora_modo(val):
        return (f"background-color: {COLORI_MODO.get(val,'#fff')}80; "
                "color: #1C2833; font-weight: 600;")

    style_cols = [c for c in ["mode", "threshold_mode"] if c in df_tab.columns]
    styled = df_tab.style.applymap(colora_modo, subset=style_cols)
    st.dataframe(styled, use_container_width=True, height=420)

# =============================================================================
# AUTO-REFRESH
# =============================================================================
_time.sleep(REFRESH_SEC)
st.rerun()
