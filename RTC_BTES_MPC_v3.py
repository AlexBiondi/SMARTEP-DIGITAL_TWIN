#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC-BTES MPC Digital Twin — v3  (modello radiativo Brutsaert)
==============================================================
Identico a v2 tranne per il modello di temperatura del cielo:

  v2 usava Swinbank (1963):  T_sky = 0.0552 * T_aria^1.5
    → dipende solo dalla temperatura dell'aria, ignora l'umidità

  v3 usa Brutsaert (1975):   ε_a = 1.24 * (e_a / T_aria_K)^(1/7)
                              T_sky = ε_a^(1/4) * T_aria_K
    → incorpora la pressione di vapore e_a [hPa], calcolata da RH
      tramite la formula di Magnus-Tetens:
        e_sat = 6.112 * exp(17.67 * T_C / (T_C + 243.5))   [hPa]
        e_a   = (RH / 100) * e_sat

  La formula di Brutsaert è più accurata in condizioni di alta umidità
  (es. Palermo in estate) perché il vapore acqueo è il principale
  assorbitore di radiazione infrarossa atmosferica. Con RH elevata
  l'emissività del cielo aumenta e T_sky risulta più alta, riducendo
  le perdite radiative nette del collettore stradale.

  Riferimenti:
    Swinbank W.C. (1963), Q.J.R. Meteorol. Soc., 89, 339–348
    Brutsaert W. (1975), Water Resour. Res., 11(5), 742–744

Controllo predittivo (MPC) con rollout — invariato rispetto a v2.
Output salvato in RTC_BTES_output_MPC_v3/ per non sovrascrivere v2.
"""

import os
import math
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from datetime import datetime, timedelta, timezone
from scipy.interpolate import interp1d

# =============================================================================
# TIMING
# =============================================================================
t_start = time.perf_counter()

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'legend.fontsize': 14, 'figure.dpi': 300, 'lines.linewidth': 2
})

current_dir = os.getcwd()
output_dir  = os.path.join(current_dir, "RTC_BTES_output_MPC_v3")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# PARAMETRI SIMULAZIONE  ← modifica qui
# =============================================================================
SIM_MODE   = '4months'
START_HOUR = 5088   # 1° agosto 2025

_SIM_HOURS = {'3days': 3*24, '1month': 30*24, '1year': 365*24, '4months': 122*24}
n_ore_sim  = _SIM_HOURS[SIM_MODE]
t_max      = float(n_ore_sim) * 3600.0

_SIM_REF_DATE = datetime(2025, 1, 1, 0, 0)
sim_start_dt  = _SIM_REF_DATE + timedelta(hours=START_HOUR)
sim_end_dt    = sim_start_dt  + timedelta(hours=n_ore_sim)
sim_giorni    = n_ore_sim / 24

MESI_IT = ['','gennaio','febbraio','marzo','aprile','maggio','giugno',
           'luglio','agosto','settembre','ottobre','novembre','dicembre']
def fmt_data(dt):
    return f"{dt.day} {MESI_IT[dt.month]} {dt.year}"

SIM_TAG = f"{sim_start_dt.strftime('%Y%m%d')}_{SIM_MODE}_MPC_v3"

# --- Parametri MPC ---
H_MPC              = 12    # orizzonte previsionale [ore]
MAX_PICARD_EXEC    = 20    # iterazioni Picard per esecuzione reale
MAX_PICARD_ROLLOUT = 1     # iterazioni Picard nel rollout (velocità > precisione)

# --- Sorgente meteo ---
# USE_FORECAST = True  → chiama Open-Meteo API (richiede connessione internet)
# USE_FORECAST = False → usa i dati storici dal CSV locale (modalità offline)
USE_FORECAST       = False
LAT                = 38.10094   # 38° 6' 3.406" N — sito SMARTEP, Palermo
LON                = 13.34657   # 13° 20' 47.67" E
FORECAST_REFRESH_H = 6          # ore tra aggiornamenti API

# =============================================================================
# PARAMETRI FISICI RTC  (identici al v3)
# =============================================================================
spessori = np.array([0.11, 0.07, 0.5, 1.0])
lambda_i = np.array([0.95, 1.83, 0.95, 1.175])
rho_sol  = np.array([1545, 2250, 2055, 1804])
Cp_sol   = np.array([1314,  950,  838,  1302])

Ca = float(rho_sol[0] * Cp_sol[0] * spessori[0])
Cb = float(rho_sol[1] * Cp_sol[1] * spessori[1])
Cc = float(rho_sol[2] * Cp_sol[2] * spessori[2])
Cd = float(rho_sol[3] * Cp_sol[3] * spessori[3])

r              = spessori / lambda_i
ra, rb, rc, rd = map(float, r)

sigma   = 5.67e-8
epsilon = 0.88

rho_water_ref = 996.0
cp_water_ref  = 4182.0
lambda_w_ref  = 0.6097
lambda_tubo   = 0.35
De            = 0.032
Di            = 0.0291
r_c           = 0.0
alpha_rad     = 0.932

u_avg          = 1.24
mass_flow_rate = rho_water_ref * (math.pi * Di**2 / 4.0) * u_avg

ni_w   = 8.8e-7
Alpha  = lambda_w_ref / (rho_water_ref * cp_water_ref)
Pr     = ni_w / Alpha
Re     = Di * u_avg / ni_w
Nu     = 0.023 * Re**0.8 * Pr**0.4
h_i    = float(Nu * lambda_w_ref / Di)
r_tubo = float(De / (2 * lambda_tubo) * math.log(De / Di))
U      = float(1.0 / (1.0/h_i + r_tubo + r_c))

print(f"m_dot = {mass_flow_rate:.4f} kg/s  |  U = {U:.1f} W/m2K  |  Re = {Re:.0f}")

m_dot_dc = 2973.0 / 3600.0   # [kg/s]
cp_dc    = 4200.0             # [J/(kg·K)]
DC_COEFF = 0.548e3            # [W]
DC_ESPON = 0.8733             # [-]

# =============================================================================
# GRIGLIA SPAZIALE / TEMPORALE  (identica al v3)
# =============================================================================
dt      = 0.1       # passo temporale FD [s]
Delta_z = 0.2       # passo spaziale [m]
P       = 0.125     # interasse tubi [m]
L_tot   = 19.0      # lunghezza collettore [m]
Area    = L_tot * P
n_z     = int(L_tot / Delta_z)

print("Courant:", u_avg * dt / Delta_z)

dt_btes    = 3600.0
save_every = int(round(dt_btes / dt))
n_ore      = int(t_max / dt_btes)

lunghezza_tubo       = n_z * Delta_z
tempo_di_percorrenza = lunghezza_tubo / u_avg
buffer_size          = int(round(tempo_di_percorrenza / dt)) + 2
A_seg                = P * Delta_z

print(f"n_z = {n_z}  |  buffer_size = {buffer_size}  |  n_ore = {n_ore}")

# =============================================================================
# CARICAMENTO CSV  (sempre caricato — usato come fallback o unica sorgente)
# =============================================================================
weather_path = "Datiy2025.csv"
dataW        = pd.read_csv(weather_path, sep=';', parse_dates=['time'])
i_end_csv    = min(START_HOUR + n_ore_sim + H_MPC + 2, len(dataW))
dataW_win    = dataW.iloc[START_HOUR:i_end_csv].reset_index(drop=True)

csv_T_aria = dataW_win["temperature_2m (°C)"].to_numpy(dtype=float)
csv_Q_irr  = np.clip(dataW_win["shortwave_radiation (W/m2)"].to_numpy(dtype=float), 0, None)
csv_WSpeed = np.clip(dataW_win["wind_speed_10m (km/h)"].to_numpy(dtype=float) / 3.6, 0, None)
csv_RH     = dataW_win["relative_humidity_2m (%)"].to_numpy(dtype=float)
# diffuse_radiation caricata per future estensioni (non usata nel FD attuale)
csv_Q_diff = np.clip(dataW_win["diffuse_radiation (W/m2)"].to_numpy(dtype=float), 0, None)

# =============================================================================
# CARICAMENTO G-FUNCTION BTES  (identico al v3)
# =============================================================================
df_gf        = pd.read_excel("MIFT_ref.xlsx", sheet_name="MIFT")
gfunc        = pd.DataFrame({'t_ts': df_gf.iloc[1:655, 0], 'gfunc': df_gf.iloc[1:655, 1]})
gfunc_interp = interp1d(gfunc['t_ts'], gfunc['gfunc'],
                        kind='linear', fill_value='extrapolate')

# =============================================================================
# PARAMETRI BTES  (identici al v3)
# =============================================================================
H_btes  = 15.0
R_field = 0.1187
k_s     = 1.68
Cp_s    = 2.351e6
alpha_s = k_s / Cp_s
ts      = H_btes**2 / (9.0 * alpha_s)
T0_btes = 20.0
n_bores = 24
H_tot   = H_btes * n_bores
tol_btes = 0.5

# =============================================================================
# SOGLIE SCHEDULER  (usate come policy di rollout nell'MPC)
# =============================================================================
T_aria_soglia_estate = 15.0   # [°C]
Q_solare_soglia_rtc  = 100.0  # [W/m²]
T_btes_min_scarica   = 18.0   # [°C]
T_btes_max_carica    = 40.0   # [°C]

# =============================================================================
# FUNZIONI FISICHE  (identiche al v3 — aggrega_carico_btes refactored)
# =============================================================================
def water_props(Tk: float):
    """Densità e cp acqua in funzione di T [K]."""
    rho_w = -0.005128 * Tk**2 + 2.8156 * Tk + 613.4536
    cp_w  = -0.00065  * Tk**3 + 0.6118 * Tk**2 - 191.7925 * Tk + 24212.55
    return float(rho_w), float(cp_w)


def T_dc_ritorno(T_aria_C: float, T_in_C: float, m_dot: float = None) -> float:
    """Temperatura uscita dry cooler [°C] — modello empirico SMARTEP."""
    if m_dot is None:
        m_dot = m_dot_dc
    delta_T = T_in_C - T_aria_C
    if delta_T <= 0.0:
        return float(T_in_C)
    T_out = T_in_C - (DC_COEFF / (m_dot * cp_dc)) * (delta_T ** DC_ESPON)
    return float(min(max(T_out, T_aria_C + 0.1), T_in_C))


def Q_dc_demand(T_aria_C: float, T_in_C: float = None, m_dot: float = None) -> float:
    """Potenza dissipata dal dry cooler [W] — modello empirico SMARTEP."""
    if m_dot  is None: m_dot  = m_dot_dc
    if T_in_C is None: T_in_C = T0_btes
    delta_T = T_in_C - T_aria_C
    if delta_T <= 0.0:
        return 0.0
    return float(max(DC_COEFF * (delta_T ** DC_ESPON), 0.0))


def aggrega_carico_btes(carichi: list, q_W: float,
                        t_in_h: float, t_end_h: float) -> None:
    """
    Aggiunge un carico e aggrega i vecchi con schema a fusione a coppie.
    REFACTORED rispetto al v3: opera sulla lista passata, non sulla globale.
    """
    carichi.append({"q": q_W, "delta_t": 1, "t_in": t_in_h, "t_end": t_end_h})
    if len(carichi) <= 47:
        return
    non_agg = carichi[-47:]
    agg     = carichi[:-47]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(agg) - 1:
            if agg[i]["delta_t"] == agg[i+1]["delta_t"]:
                dt_i = agg[i]["delta_t"]; dt_i1 = agg[i+1]["delta_t"]
                merged = {
                    "q"      : (agg[i]["q"]*dt_i + agg[i+1]["q"]*dt_i1) / (dt_i+dt_i1),
                    "delta_t": dt_i + dt_i1,
                    "t_in"   : agg[i]["t_in"],
                    "t_end"  : agg[i+1]["t_end"],
                }
                agg[i] = merged; del agg[i+1]; changed = True; break
            i += 1
    carichi[:len(agg)] = agg
    carichi[len(agg):] = non_agg


def T_btes_fluid(carichi: list, t_now_h: float) -> float:
    """Temperatura fluido BTES via g-function (sovrapposizione temporale)."""
    if not carichi:
        return T0_btes
    Tb_val = T0_btes
    c_ext  = [{"q": 0.0, "delta_t": 0, "t_in": 0.0, "t_end": 0.0}] + list(carichi)
    for idx in range(1, len(c_ext)):
        time_p  = c_ext[idx-1]['t_end']
        Q_c     = c_ext[idx]['q']   / H_tot
        Q_p     = c_ext[idx-1]['q'] / H_tot
        delta_Q = Q_c - Q_p
        t_arg   = (t_now_h - time_p) * 3600.0 / ts
        Tb_val += delta_Q / (2.0 * math.pi * k_s) * float(gfunc_interp(t_arg))
    return Tb_val + carichi[-1]['q'] / H_tot * R_field

# =============================================================================
# METEO: OPEN-METEO API CON CACHE
# =============================================================================
_forecast_cache: dict = {
    "generationtime_ms": None,
    "data"             : None,
    "fetched_at"       : None,
}

def fetch_forecast_if_updated(lat: float, lon: float, H: int = 48) -> dict:
    """
    Restituisce previsioni orarie (dizionario di array numpy di lunghezza H).
    Chiama l'API Open-Meteo solo se il modello NWP è stato aggiornato.

    Variabili restituite:
      T_aria [°C], Q_irr [W/m²], WSpeed [m/s], RH [%], T_dew [°C], Q_diff [W/m²]

    Q_diff (radiazione diffusa) è memorizzata per future estensioni del modello
    radiativo nel loop FD (es. separazione beam/diffuse per superfici inclinate).
    """
    now = datetime.now(timezone.utc)
    ore_trascorse = (
        (now - _forecast_cache["fetched_at"]).total_seconds() / 3600
        if _forecast_cache["fetched_at"] else 999.0
    )
    if ore_trascorse < FORECAST_REFRESH_H and _forecast_cache["data"] is not None:
        return _forecast_cache["data"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"     : lat,
        "longitude"    : lon,
        "hourly"       : ("temperature_2m,shortwave_radiation,diffuse_radiation,"
                          "wind_speed_10m,relative_humidity_2m,dewpoint_2m"),
        "forecast_days": max(2, H // 24 + 1),
        "timezone"     : "Europe/Rome",
        "models"       : "best_match",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"  [Forecast] Errore API: {exc} — uso cache precedente")
        return _forecast_cache["data"]

    gen_time = payload.get("generationtime_ms")
    if gen_time == _forecast_cache["generationtime_ms"]:
        _forecast_cache["fetched_at"] = now
        print(f"  [Forecast] Modello non aggiornato (gen={gen_time:.0f}ms) — cache OK")
        return _forecast_cache["data"]

    d = payload["hourly"]
    result = {
        "T_aria" : np.array(d["temperature_2m"][:H],       dtype=float),
        "Q_irr"  : np.clip(np.array(d["shortwave_radiation"][:H], dtype=float), 0, None),
        "Q_diff" : np.clip(np.array(d["diffuse_radiation"][:H],   dtype=float), 0, None),
        "WSpeed" : np.clip(np.array(d["wind_speed_10m"][:H],      dtype=float), 0, None),
        "RH"     : np.array(d["relative_humidity_2m"][:H], dtype=float),
        "T_dew"  : np.array(d["dewpoint_2m"][:H],          dtype=float),
    }
    _forecast_cache.update({"generationtime_ms": gen_time,
                             "data": result, "fetched_at": now})
    print(f"  [Forecast] Aggiornato @ {now.strftime('%H:%M UTC')}  gen={gen_time:.0f}ms")
    return result


def meteo_da_csv(ora_offset: int, H: int) -> dict:
    """
    Estrae H ore di dati meteo dal CSV a partire da ora_offset.
    Fallback quando USE_FORECAST = False o API non raggiungibile.
    """
    i_s = ora_offset
    i_e = min(ora_offset + H, len(csv_T_aria))
    n   = i_e - i_s

    def _pad(arr):
        v = arr[i_s:i_e].copy()
        if n < H:
            v = np.concatenate([v, np.full(H - n, v[-1] if n > 0 else 0.0)])
        return v

    T = _pad(csv_T_aria)
    R = _pad(csv_RH)
    return {
        "T_aria" : T,
        "Q_irr"  : _pad(csv_Q_irr),
        "Q_diff" : _pad(csv_Q_diff),
        "WSpeed" : _pad(csv_WSpeed),
        "RH"     : R,
        "T_dew"  : T - (100.0 - R) / 5.0,   # Magnus approssimata
    }


def get_forecast(ora: int, H: int) -> dict:
    """Punto di accesso unificato: API o CSV in base a USE_FORECAST."""
    if USE_FORECAST:
        return fetch_forecast_if_updated(LAT, LON, H=H)
    return meteo_da_csv(ora, H=H)

# =============================================================================
# SCHEDULER A SOGLIE  (rimane come policy di rollout nell'MPC)
# =============================================================================
def get_mode_threshold(T_btes_C: float, T_aria_C: float, Q_irr_val: float) -> str:
    """Policy a soglie del v3 — usata nel rollout MPC."""
    solar_ok   = Q_irr_val > Q_solare_soglia_rtc
    btes_caldo = T_btes_C  > T_btes_min_scarica
    btes_ok    = T_btes_C  < T_btes_max_carica
    if T_aria_C >= T_aria_soglia_estate:
        return 'charge' if (solar_ok and btes_ok) else 'idle'
    else:
        if solar_ok:       return 'solar_dc'
        elif btes_caldo:   return 'discharge'
        else:              return 'idle'


def get_candidate_modes(T_btes_C: float, T_aria_C: float,
                        Q_irr_val: float) -> list:
    """
    Modalità fisicamente ammissibili nell'ora corrente.
    L'MPC valuterà ognuna con il rollout. 'idle' è sempre inclusa.
    """
    modes = ['idle']
    if T_aria_C >= T_aria_soglia_estate:
        if Q_irr_val > Q_solare_soglia_rtc and T_btes_C < T_btes_max_carica:
            modes.append('charge')
    else:
        if Q_irr_val > Q_solare_soglia_rtc:
            modes.append('solar_dc')
        if T_btes_C > T_btes_min_scarica:
            modes.append('discharge')
    return modes

# =============================================================================
# NUCLEO FD: SIMULAZIONE DI UNA SINGOLA ORA
# =============================================================================
def simula_ora(mode: str,
               state: dict,
               meteo_cst: dict,
               t_h_start: float,
               n_picard: int = MAX_PICARD_EXEC) -> dict:
    """
    Esegue il modello FD per una singola ora nella modalità specificata.
    Lavora ESCLUSIVAMENTE su copie dello stato: non modifica le variabili esterne.

    Struttura di 'state'
    --------------------
    Ta, Tb, Tc, Td, Tw : np.ndarray (n_z+1, buffer_size+1) — temperature [K]
    buf                : int    — indice buffer corrente
    Tw_in_K            : float  — temperatura ingresso fluido [K]
    T_btes_C           : float  — temperatura fluido BTES [°C]
    carichi            : list   — storico carichi BTES

    Struttura di 'meteo_cst'
    ------------------------
    T_aria  [°C], Q_irr [W/m²], WSpeed [m/s], RH [%]
    RH è memorizzato nell'array ma non ancora usato nelle equazioni FD.
    Sarà utilizzato quando si adotterà la formula di Brutsaert per T_sky
    (sostituirà l'attuale Swinbank che dipende solo da T_aria).

    Ritorna dict con stato aggiornato + grandezze energetiche:
      Q_rtc_W, Q_dc_W, Q_solar_W, n_iter, converged,
      Tw_in_C, Tw_out_C, Ta_C, Tb_C, Tc_C, Td_C,
      T_dc_ret_C, Tw_spatial [array n_z]
    """
    t_h_end = t_h_start + 1.0

    # --- Copia profonda dello stato (protegge le variabili esterne) ---
    Ta      = state['Ta'].copy()
    Tb      = state['Tb'].copy()
    Tc      = state['Tc'].copy()
    Td      = state['Td'].copy()
    Tw      = state['Tw'].copy()
    buf0    = int(state['buf'])
    Tw_in   = float(state['Tw_in_K'])
    T_btes  = float(state['T_btes_C'])
    carichi = copy.deepcopy(state['carichi'])

    # --- Costanti meteo per l'ora (invarianti nel loop FD interno) ---
    T_aria_C   = float(meteo_cst['T_aria'])
    Q_irr_val  = float(alpha_rad * meteo_cst['Q_irr'])
    WSpeed_val = float(meteo_cst['WSpeed'])
    RH_val     = float(meteo_cst.get('RH', 50.0))   # umidità relativa [%]

    T_air_k = T_aria_C + 273.15

    # Brutsaert (1975): emissività atmosferica in funzione di T e umidità.
    # Pressione di vapore saturo con formula di Magnus-Tetens [hPa]:
    e_sat_hPa = 6.112 * math.exp(17.67 * T_aria_C / (T_aria_C + 243.5))
    # Pressione di vapore effettiva [hPa]:
    e_a_hPa   = (RH_val / 100.0) * e_sat_hPa
    # Emissività atmosferica [-]  (Brutsaert 1975, eq. 11):
    eps_a   = 1.24 * (e_a_hPa / T_air_k) ** (1.0 / 7.0)
    eps_a   = min(eps_a, 1.0)          # limite fisico superiore
    # Temperatura efficace del cielo [K]:
    T_sky_k = eps_a ** 0.25 * T_air_k
    h_air   = 3.1 + 4.1 * WSpeed_val
    Us      = 1.0 / (ra/2.0 + (1.0/h_air if h_air > 1e-6 else 1e-6))

    # ------------------------------------------------------------------
    # Funzione interna: esegue save_every passi FD partendo da buf0.
    # Le matrici vengono modificate IN PLACE (già copie locali).
    # ------------------------------------------------------------------
    def _fd_loop(Ta_l, Tb_l, Tc_l, Td_l, Tw_l, Tw_in_l):
        buf_p   = buf0
        Q_rtc_J = 0.0
        Q_sol_J = 0.0
        buf_c   = buf_p
        for _ in range(save_every):
            buf_c = (buf_p + 1) % buffer_size
            Tw_l[0, buf_c] = Tw_in_l
            for j in range(1, n_z + 1):
                h_r = (sigma * epsilon
                       * (Ta_l[j, buf_p]**2 + T_sky_k**2)
                       * (Ta_l[j, buf_p] + T_sky_k))

                Ta_l[j, buf_c] = Ta_l[j, buf_p] + dt * (
                    Q_irr_val / Ca
                    - (2.0/(Ca*(ra+rb))) * (Ta_l[j,buf_p] - Tb_l[j,buf_p])
                    - (Us/Ca)            * (Ta_l[j,buf_p] - T_air_k)
                    - (h_r/Ca)           * (Ta_l[j,buf_p] - T_sky_k))

                Tb_l[j, buf_c] = Tb_l[j, buf_p] + dt * (
                    (2.0/(Cb*(ra+rb))) * (Ta_l[j,buf_p] - Tb_l[j,buf_p])
                    - (2.0/(Cb*(rb+rc))) * (Tb_l[j,buf_p] - Tc_l[j,buf_p])
                    - (math.pi*De/(P*Cb))*U * (Tb_l[j,buf_p] - Tw_l[j,buf_p]))

                Tc_l[j, buf_c] = Tc_l[j, buf_p] + dt * (
                    (2.0/(Cc*(rb+rc))) * (Tb_l[j,buf_p] - Tc_l[j,buf_p])
                    - (2.0/(Cc*(rc+rd))) * (Tc_l[j,buf_p] - Td_l[j,buf_p]))

                Td_l[j, buf_c] = Td_l[j, buf_p] + dt * (
                    (2.0/(Cd*(rc+rd))) * (Tc_l[j,buf_p] - Td_l[j,buf_p]))

                rho_loc, cp_loc = water_props(float(Tw_l[j, buf_p]))
                Tw_l[j, buf_c] = Tw_l[j, buf_p] + dt * (
                    (U*4*De)/(Di**2*rho_loc*cp_loc)
                    * (Tb_l[j,buf_p] - Tw_l[j,buf_p])
                    - (u_avg/Delta_z) * (Tw_l[j,buf_p] - Tw_l[j-1,buf_c]))

            Q_sol_J  += A_seg * n_z * Q_irr_val * dt
            T_out_i   = float(Tw_l[n_z, buf_c])
            T_in_i    = float(Tw_l[0,  buf_c])
            _, cp_mid = water_props((T_out_i + T_in_i) / 2.0)
            Q_rtc_J  += mass_flow_rate * cp_mid * (T_out_i - T_in_i) * dt
            buf_p = buf_c
        return buf_c, Q_rtc_J, Q_sol_J

    # ================================================================
    # BRANCH 1: CHARGE — RTC → BTES
    # ================================================================
    if mode == 'charge':
        Q_ora_prev  = 0.0
        converged   = False
        Q_rtc_W_ora = 0.0
        Q_solar_ora = 0.0
        buf_final   = buf0
        n_it        = 0
        # Stato finale (aggiornato ad ogni iterazione Picard)
        Ta_f=Ta.copy(); Tb_f=Tb.copy(); Tc_f=Tc.copy(); Td_f=Td.copy(); Tw_f=Tw.copy()
        carichi_f = copy.deepcopy(carichi)

        for it in range(n_picard):
            Ta_i=Ta.copy(); Tb_i=Tb.copy(); Tc_i=Tc.copy(); Td_i=Td.copy(); Tw_i=Tw.copy()
            buf_c, Q_rtc_J, Q_sol_J = _fd_loop(Ta_i, Tb_i, Tc_i, Td_i, Tw_i, Tw_in)
            Q_rtc_W_ora = Q_rtc_J / dt_btes
            Q_solar_ora = Q_sol_J / dt_btes

            carichi_it = copy.deepcopy(carichi)
            if carichi_it and carichi_it[-1]['t_in'] == t_h_start:
                carichi_it.pop()
            aggrega_carico_btes(carichi_it, max(Q_rtc_W_ora, 0.0), t_h_start, t_h_end)
            T_btes_new = T_btes_fluid(carichi_it, t_h_end)
            Tw_in_new  = T_btes_new + 273.15

            delta_Q = abs(Q_rtc_W_ora - Q_ora_prev)
            delta_T = abs(T_btes_new - T_btes)

            Ta_f=Ta_i; Tb_f=Tb_i; Tc_f=Tc_i; Td_f=Td_i; Tw_f=Tw_i
            buf_final = buf_c; carichi_f = carichi_it
            Tw_in = Tw_in_new; T_btes = T_btes_new; Q_ora_prev = Q_rtc_W_ora
            n_it  = it + 1

            if it > 0 and (delta_Q < tol_btes or delta_T < 0.1):
                converged = True; break

        T_dc_ret_C = T_aria_C
        Q_dc_W     = 0.0

    # ================================================================
    # BRANCH 2: SOLAR_DC — RTC → Dry Cooler
    # ================================================================
    elif mode == 'solar_dc':
        T_dc_ret_prev = T_aria_C + 3.0
        Tw_in         = T_dc_ret_prev + 273.15
        Q_ora_prev    = 0.0
        converged     = False
        Q_rtc_W_ora   = 0.0
        Q_solar_ora   = 0.0
        buf_final     = buf0
        T_dc_ret_C    = T_dc_ret_prev
        n_it          = 0
        Ta_f=Ta.copy(); Tb_f=Tb.copy(); Tc_f=Tc.copy(); Td_f=Td.copy(); Tw_f=Tw.copy()

        for it in range(n_picard):
            Ta_i=Ta.copy(); Tb_i=Tb.copy(); Tc_i=Tc.copy(); Td_i=Td.copy(); Tw_i=Tw.copy()
            buf_c, Q_rtc_J, Q_sol_J = _fd_loop(Ta_i, Tb_i, Tc_i, Td_i, Tw_i, Tw_in)
            Q_rtc_W_ora  = Q_rtc_J / dt_btes
            Q_solar_ora  = Q_sol_J / dt_btes
            T_out_rtc_C  = float(Tw_i[n_z, buf_c]) - 273.15
            T_dc_ret_new = T_dc_ritorno(T_aria_C, T_out_rtc_C)

            delta_Q = abs(Q_rtc_W_ora - Q_ora_prev)
            delta_T = abs(T_dc_ret_new - T_dc_ret_prev)

            Ta_f=Ta_i; Tb_f=Tb_i; Tc_f=Tc_i; Td_f=Td_i; Tw_f=Tw_i
            buf_final = buf_c; T_dc_ret_C = T_dc_ret_new
            Tw_in = T_dc_ret_new + 273.15; Q_ora_prev = Q_rtc_W_ora
            n_it  = it + 1

            if it > 0 and (delta_Q < tol_btes and delta_T < 0.1):
                converged = True; T_dc_ret_prev = T_dc_ret_new; break
            T_dc_ret_prev = T_dc_ret_new

        # BTES invariato in solar_dc: carico zero
        if carichi and carichi[-1]['t_in'] == t_h_start:
            carichi.pop()
        aggrega_carico_btes(carichi, 0.0, t_h_start, t_h_end)
        T_btes     = T_btes_fluid(carichi, t_h_end)
        carichi_f  = carichi
        Tw_in      = T_dc_ret_C + 273.15
        Q_dc_W     = max(Q_rtc_W_ora, 0.0)

    # ================================================================
    # BRANCH 3: DISCHARGE — BTES → Dry Cooler
    # ================================================================
    elif mode == 'discharge':
        Q_dc_richiesta = Q_dc_demand(T_aria_C, T_in_C=T_btes)
        q_discharge    = -Q_dc_richiesta
        if carichi and carichi[-1]['t_in'] == t_h_start:
            carichi.pop()
        aggrega_carico_btes(carichi, q_discharge, t_h_start, t_h_end)
        T_btes_new = T_btes_fluid(carichi, t_h_end)
        T_dc_ret_C = T_dc_ritorno(T_aria_C, T_btes_new)
        T_btes     = T_btes_new
        Tw_in      = T_dc_ret_C + 273.15
        carichi_f  = carichi
        Ta_f=Ta; Tb_f=Tb; Tc_f=Tc; Td_f=Td; Tw_f=Tw
        buf_final   = buf0
        Q_rtc_W_ora = 0.0
        Q_solar_ora = 0.0
        Q_dc_W      = Q_dc_richiesta
        converged   = True
        n_it        = 1

    # ================================================================
    # BRANCH 4: IDLE
    # ================================================================
    else:
        if carichi and carichi[-1]['t_in'] == t_h_start:
            carichi.pop()
        aggrega_carico_btes(carichi, 0.0, t_h_start, t_h_end)
        T_btes      = T_btes_fluid(carichi, t_h_end)
        carichi_f   = carichi
        Ta_f=Ta; Tb_f=Tb; Tc_f=Tc; Td_f=Td; Tw_f=Tw
        buf_final   = buf0
        Q_rtc_W_ora = 0.0
        Q_solar_ora = 0.0
        Q_dc_W      = 0.0
        T_dc_ret_C  = T_aria_C
        converged   = True
        n_it        = 1

    # --- Grandezze di output ---
    j0 = 10   # nodo di riferimento per output temperature strati
    if mode in ('charge', 'solar_dc'):
        Tw_in_C  = Tw_in - 273.15
        Tw_out_C = float(Tw_f[n_z, buf_final]) - 273.15
        Tw_spat  = Tw_f[1:n_z+1, buf_final].copy() - 273.15
        Ta_C_out = float(Ta_f[j0, buf_final]) - 273.15
        Tb_C_out = float(Tb_f[j0, buf_final]) - 273.15
        Tc_C_out = float(Tc_f[j0, buf_final]) - 273.15
        Td_C_out = float(Td_f[j0, buf_final]) - 273.15
    else:
        Tw_in_C  = float('nan')
        Tw_out_C = float('nan')
        Tw_spat  = np.full(n_z, float('nan'))
        Ta_C_out = float(Ta_f[j0, buf0]) - 273.15
        Tb_C_out = float(Tb_f[j0, buf0]) - 273.15
        Tc_C_out = float(Tc_f[j0, buf0]) - 273.15
        Td_C_out = float(Td_f[j0, buf0]) - 273.15

    return {
        # Stato aggiornato (pronto per il passo successivo)
        'Ta': Ta_f, 'Tb': Tb_f, 'Tc': Tc_f, 'Td': Td_f, 'Tw': Tw_f,
        'buf'     : buf_final,
        'Tw_in_K' : Tw_in,
        'T_btes_C': T_btes,
        'carichi' : carichi_f,
        # Grandezze energetiche
        'Q_rtc_W'   : Q_rtc_W_ora,
        'Q_dc_W'    : Q_dc_W,
        'Q_solar_W' : Q_solar_ora,
        'n_iter'    : n_it,
        'converged' : converged,
        # Output termici
        'Tw_in_C'   : Tw_in_C,
        'Tw_out_C'  : Tw_out_C,
        'Ta_C'      : Ta_C_out,
        'Tb_C'      : Tb_C_out,
        'Tc_C'      : Tc_C_out,
        'Td_C'      : Td_C_out,
        'T_dc_ret_C': T_dc_ret_C,
        'Tw_spatial': Tw_spat,
    }

# =============================================================================
# REWARD
# =============================================================================
def reward_ora(mode: str, Q_rtc_W: float, Q_dc_W: float, T_btes_C: float) -> float:
    """
    Reward per una singola ora.

    Obiettivo doppio:
      estate  → kWh caricati nel BTES   (charge)
      inverno → kWh ceduti al DC        (discharge, solar_dc)
    Penalità proporzionale se T_btes esce dai limiti operativi:
      il fattore 5.0 [kWh-eq/°C] è calibrabile.
    """
    r = 0.0
    if mode == 'charge':
        r += max(Q_rtc_W, 0.0) / 1000.0
    elif mode in ('discharge', 'solar_dc'):
        r += max(Q_dc_W, 0.0) / 1000.0

    if T_btes_C > T_btes_max_carica:
        r -= 5.0 * (T_btes_C - T_btes_max_carica)
    if T_btes_C < T_btes_min_scarica:
        r -= 5.0 * (T_btes_min_scarica - T_btes_C)
    return r

# =============================================================================
# MPC: SCELTA DELLA MODALITÀ OTTIMALE
# =============================================================================
def mpc_choose_mode(state: dict, forecast: dict,
                    t_h_start: float) -> tuple:
    """
    Valuta ogni modalità candidata con rollout a H_MPC ore.
    Restituisce (mode_ottimale, dict_rewards, risultato_esecuzione).

    Il risultato_esecuzione è il dict di simula_ora per il modo scelto:
    il main loop lo usa direttamente, evitando una seconda simulazione.

    Logica del rollout
    ------------------
    Ora 0      : simulazione completa con Picard (n_picard = MAX_PICARD_EXEC)
    Ore 1..H-1 : policy a soglie + Picard con MAX_PICARD_ROLLOUT iterazioni
                 (velocità > precisione, è solo una stima delle conseguenze future)
    """
    meteo0     = {k: float(forecast[k][0]) for k in ('T_aria','Q_irr','WSpeed','RH')}
    candidates = get_candidate_modes(state['T_btes_C'],
                                     meteo0['T_aria'], meteo0['Q_irr'])
    rewards    = {}
    ris_cache  = {}

    for mode in candidates:
        # Ora 0: simulazione con Picard completo
        ris0  = simula_ora(mode, state, meteo0, t_h_start,
                           n_picard=MAX_PICARD_EXEC)
        r_tot = reward_ora(mode, ris0['Q_rtc_W'], ris0['Q_dc_W'], ris0['T_btes_C'])
        ris_cache[mode] = ris0

        # Ore 1..H_MPC-1: rollout con policy a soglie
        state_h = {
            'Ta': ris0['Ta'], 'Tb': ris0['Tb'],
            'Tc': ris0['Tc'], 'Td': ris0['Td'], 'Tw': ris0['Tw'],
            'buf'     : ris0['buf'],
            'Tw_in_K' : ris0['Tw_in_K'],
            'T_btes_C': ris0['T_btes_C'],
            'carichi' : copy.deepcopy(ris0['carichi']),
        }

        for h in range(1, H_MPC):
            if h >= len(forecast['T_aria']):
                break
            meteo_h = {k: float(forecast[k][h])
                       for k in ('T_aria','Q_irr','WSpeed','RH')}
            mode_h  = get_mode_threshold(state_h['T_btes_C'],
                                         meteo_h['T_aria'], meteo_h['Q_irr'])
            ris_h   = simula_ora(mode_h, state_h, meteo_h,
                                 t_h_start + h,
                                 n_picard=MAX_PICARD_ROLLOUT)
            r_tot  += reward_ora(mode_h, ris_h['Q_rtc_W'],
                                 ris_h['Q_dc_W'], ris_h['T_btes_C'])
            state_h = {
                'Ta': ris_h['Ta'], 'Tb': ris_h['Tb'],
                'Tc': ris_h['Tc'], 'Td': ris_h['Td'], 'Tw': ris_h['Tw'],
                'buf'     : ris_h['buf'],
                'Tw_in_K' : ris_h['Tw_in_K'],
                'T_btes_C': ris_h['T_btes_C'],
                'carichi' : copy.deepcopy(ris_h['carichi']),
            }

        rewards[mode] = r_tot
        print(f"    [{mode:10s}]  reward {H_MPC}h = {r_tot:+.3f} kWh-eq")

    best_mode = max(rewards, key=rewards.get)
    return best_mode, rewards, ris_cache[best_mode]

# =============================================================================
# CONDIZIONI INIZIALI
# =============================================================================
T_init = 20.0 + 273.15   # [K]

state = {
    'Ta'      : np.full((n_z + 1, buffer_size + 1), T_init),
    'Tb'      : np.full((n_z + 1, buffer_size + 1), T_init),
    'Tc'      : np.full((n_z + 1, buffer_size + 1), T_init),
    'Td'      : np.full((n_z + 1, buffer_size + 1), T_init),
    'Tw'      : np.full((n_z + 1, buffer_size + 1), T_init),
    'buf'     : 0,
    'Tw_in_K' : T_init,
    'T_btes_C': T0_btes,
    'carichi' : [],
}

# =============================================================================
# OUTPUT STORAGE
# =============================================================================
out = {
    "time_h": [], "mode": [], "Tw_in_C": [], "Tw_out_C": [],
    "Q_rtc_W": [], "T_btes_C": [], "Q_solar_W": [],
    "Q_conv_W": [], "Q_rad_W": [], "n_iter": [],
    "Ta_C": [], "Tb_C": [], "Tc_C": [], "Td_C": [],
    "Q_dc_W": [], "Q_dc_demand_W": [], "T_dc_ret_C": [],
    "T_aria_C": [], "mpc_best_reward": [], "mpc_n_candidates": [],
    # Confronto con lo scheduler a soglie
    "threshold_mode"   : [],   # modalità che avrebbe scelto il v3 (soglie)
    "threshold_reward" : [],   # reward 24h del v3 (dal rollout MPC)
    "mpc_gain"         : [],   # vantaggio MPC = mpc_reward - threshold_reward
    "mpc_changed"      : [],   # True se MPC ha scelto diversamente dalle soglie
}
Tw_spatial    = []
E_btes_kWh    = 0.0
E_scarica_kWh = 0.0
E_dc_kWh      = 0.0

# =============================================================================
# MAIN LOOP
# =============================================================================
print(f"\nSimulazione RTC-BTES MPC v3 (Brutsaert): {n_ore} ore  |  H_MPC={H_MPC}h")
print(f"Sorgente meteo: {'Open-Meteo API' if USE_FORECAST else 'CSV locale'}")
print("="*60)

for ora in range(n_ore):
    t_h_start = float(ora)
    t_h_end   = float(ora + 1)

    # 1. Previsioni meteo per l'orizzonte MPC
    forecast   = get_forecast(ora, H=H_MPC + 1)
    meteo_ora  = {k: float(forecast[k][0]) for k in ('T_aria','Q_irr','WSpeed','RH')}
    T_aria_ora = meteo_ora['T_aria']
    Q_dc_ric   = Q_dc_demand(T_aria_ora, T_in_C=state['T_btes_C'])

    print(f"\n[Ora {ora+1:03d}/{n_ore}]  T_aria={T_aria_ora:.1f}°C  "
          f"Q_irr={meteo_ora['Q_irr']:.0f}W/m²  "
          f"RH={meteo_ora['RH']:.0f}%  "
          f"T_BTES={state['T_btes_C']:.2f}°C  "
          f"Q_dc_dom={Q_dc_ric:.0f}W")

    # 2. Cosa avrebbe scelto lo scheduler a soglie (per confronto)
    thr_mode = get_mode_threshold(state['T_btes_C'], T_aria_ora, meteo_ora['Q_irr'])

    # 3. MPC: valuta candidati e sceglie la modalità ottimale
    #    Restituisce anche il risultato già calcolato per il modo scelto
    mode, rewards_log, ris = mpc_choose_mode(state, forecast, t_h_start)

    thr_reward = rewards_log.get(thr_mode, rewards_log.get('idle', 0.0))
    mpc_gain   = rewards_log[mode] - thr_reward

    print(f"  → MPC sceglie: {mode.upper()}  "
          f"(reward={rewards_log[mode]:+.3f} vs "
          f"{', '.join(f'{m}={v:+.2f}' for m,v in rewards_log.items() if m != mode)})")
    if mode != thr_mode:
        print(f"  ★ MPC diverge da soglie ({thr_mode.upper()}) — guadagno: {mpc_gain:+.3f} kWh-eq")

    # 3. Aggiorna stato globale con il risultato già simulato
    state = {
        'Ta': ris['Ta'], 'Tb': ris['Tb'], 'Tc': ris['Tc'],
        'Td': ris['Td'], 'Tw': ris['Tw'],
        'buf'     : ris['buf'],
        'Tw_in_K' : ris['Tw_in_K'],
        'T_btes_C': ris['T_btes_C'],
        'carichi' : ris['carichi'],
    }

    # 4. Accumuli energetici
    if mode == 'charge':
        E_btes_kWh    += max(ris['Q_rtc_W'], 0.0) / 1000.0
    elif mode == 'discharge':
        E_scarica_kWh += ris['Q_dc_W'] / 1000.0
    if mode in ('solar_dc', 'discharge'):
        E_dc_kWh += ris['Q_dc_W'] / 1000.0

    # 5. Salva output orario
    out["time_h"].append(t_h_end)
    out["mode"].append(mode)
    out["Tw_in_C"].append(ris['Tw_in_C'])
    out["Tw_out_C"].append(ris['Tw_out_C'])
    out["Q_rtc_W"].append(ris['Q_rtc_W'])
    out["T_btes_C"].append(state['T_btes_C'])
    out["Q_solar_W"].append(ris['Q_solar_W'])
    out["Q_conv_W"].append(0.0)
    out["Q_rad_W"].append(0.0)
    out["n_iter"].append(ris['n_iter'])
    out["Ta_C"].append(ris['Ta_C'])
    out["Tb_C"].append(ris['Tb_C'])
    out["Tc_C"].append(ris['Tc_C'])
    out["Td_C"].append(ris['Td_C'])
    out["Q_dc_W"].append(ris['Q_dc_W'])
    out["Q_dc_demand_W"].append(Q_dc_ric)
    out["T_dc_ret_C"].append(ris['T_dc_ret_C'])
    out["T_aria_C"].append(T_aria_ora)
    out["mpc_best_reward"].append(rewards_log[mode])
    out["mpc_n_candidates"].append(len(rewards_log))
    out["threshold_mode"].append(thr_mode)
    out["threshold_reward"].append(thr_reward)
    out["mpc_gain"].append(mpc_gain)
    out["mpc_changed"].append(mode != thr_mode)
    Tw_spatial.append(ris['Tw_spatial'])

    # --- Scrittura live per il dashboard (ogni ora, sovrascrive il file) ---
    # Il dashboard legge questo CSV e si aggiorna automaticamente ogni 60s.
    # Scrittura leggera: solo le colonne scalari, senza i profili spaziali.
    _live_cols = ["time_h","mode","T_btes_C","T_aria_C","Tw_in_C","Tw_out_C",
                  "Q_rtc_W","Q_dc_W","Q_solar_W","Q_dc_demand_W","T_dc_ret_C",
                  "Ta_C","Tb_C","Tc_C","Td_C","n_iter",
                  "mpc_best_reward","threshold_mode","threshold_reward",
                  "mpc_gain","mpc_changed","mpc_n_candidates"]
    pd.DataFrame({c: out[c] for c in _live_cols}).to_csv(
        os.path.join(output_dir, "live_results.csv"), index=False
    )

# =============================================================================
# SALVATAGGIO RISULTATI
# =============================================================================
df_out = pd.DataFrame(out)
df_out.to_excel(
    os.path.join(output_dir, f"RTC_BTES_MPC_v1_{SIM_TAG}.xlsx"),
    index=False
)

t_elapsed = time.perf_counter() - t_start
print("\n" + "="*60)
print("BILANCIO ENERGETICO FINALE")
print("="*60)
print(f"Periodo simulazione:                   {fmt_data(sim_start_dt)} → {fmt_data(sim_end_dt)}")
print(f"Durata simulazione:                    {sim_giorni:.0f} giorni ({n_ore_sim} ore)")
print(f"Tempo di calcolo:                      {t_elapsed/60:.1f} minuti")
print(f"Energia caricata nel BTES (estate):    {E_btes_kWh:.2f} kWh")
print(f"Energia scaricata dal BTES (inverno):  {E_scarica_kWh:.2f} kWh")
print(f"Energia totale al Dry Cooler:          {E_dc_kWh:.2f} kWh")
print(f"Temperatura max BTES:                  {max(out['T_btes_C']):.2f} °C")
print(f"Temperatura min BTES:                  {min(out['T_btes_C']):.2f} °C")
print(f"Temperatura media BTES:                {np.mean(out['T_btes_C']):.2f} °C")
modes_arr = np.array(out["mode"])
for m in ['charge', 'solar_dc', 'discharge', 'idle']:
    print(f"  Ore in modalità '{m}': {int(np.sum(modes_arr == m))}")

# =============================================================================
# PLOT
# =============================================================================
t_h   = np.array(out["time_h"]) / 24
colors_mode = {
    'charge'   : 'gold',
    'solar_dc' : 'skyblue',
    'discharge': 'tomato',
    'idle'     : 'lightgray'
}

# --- Plot 1: Overview ---
fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=True)

ax = axes[0]
for m, col in colors_mode.items():
    mask = modes_arr == m
    ax.fill_between(t_h, -5, 55, where=mask, alpha=0.12, color=col)
ax.plot(t_h, out["Tw_out_C"],  label="T uscita RTC",   color="tomato")
ax.plot(t_h, out["Tw_in_C"],   label="T ingresso RTC", color="steelblue")
ax.plot(t_h, out["T_btes_C"],  label="T fluido BTES",  color="darkorange", linestyle="--")
ax.plot(t_h, out["T_dc_ret_C"],label="T ritorno DC",   color="purple",     linestyle=":")
ax.plot(t_h, out["T_aria_C"],  label="T aria esterna", color="green",      alpha=0.6)
ax.axhline(T0_btes, color="k", linestyle="--", alpha=0.3,
           label=f"T0 terreno ({T0_btes}°C)")
ax.set_ylabel("Temperatura [°C]")
ax.set_title("Temperature sistema RTC–BTES–Dry Cooler  [MPC v3 — Brutsaert]")
ax.legend(loc="best", ncol=3, fontsize=11); ax.grid(True)

ax = axes[1]
ax.plot(t_h, np.array(out["Q_rtc_W"])/1000,       label="Q utile RTC [kW]",       color="forestgreen")
ax.plot(t_h, np.array(out["Q_dc_W"])/1000,        label="Q al Dry Cooler [kW]",   color="tomato")
ax.plot(t_h, np.array(out["Q_dc_demand_W"])/1000, label="Q domanda DC [kW]",      color="tomato", linestyle="--", alpha=0.6)
ax.plot(t_h, np.array(out["Q_solar_W"])/1000,     label="Q solare incidente [kW]",color="goldenrod", alpha=0.7)
ax.axhline(0, color='k', linewidth=0.8)
ax.set_ylabel("Potenza [kW]")
ax.set_title("Bilancio energetico")
ax.legend(loc="best", fontsize=11); ax.grid(True)

ax = axes[2]
ax.plot(t_h, out["T_btes_C"], color="darkorange", label="T fluido BTES")
ax.axhline(T0_btes,            color="k",         linestyle="--",
           label=f"T terreno indisturbato ({T0_btes}°C)")
ax.axhline(T_btes_min_scarica, color="steelblue", linestyle=":",
           label=f"Soglia min scarica ({T_btes_min_scarica}°C)")
ax.axhline(T_btes_max_carica,  color="firebrick", linestyle=":",
           label=f"Soglia max carica ({T_btes_max_carica}°C)")
ax.set_ylabel("T fluido BTES [°C]")
ax.set_title("Evoluzione temperatura BTES")
ax.legend(loc="best", fontsize=11); ax.grid(True)

ax = axes[3]
mode_num  = {'charge': 3, 'solar_dc': 2, 'discharge': 1, 'idle': 0}
mode_vals = [mode_num[m] for m in out["mode"]]
ax.bar(t_h, mode_vals, width=1.0/24, align='edge',
       color=[colors_mode[m] for m in out["mode"]], alpha=0.85)
ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['idle','discharge','solar_dc','charge'])
ax.set_ylabel("Modalità"); ax.set_xlabel("Tempo [giorni]")
ax.set_title("Scheduler MPC — modalità operative")
patches = [mpatches.Patch(color=col, label=m) for m, col in colors_mode.items()]
ax.legend(handles=patches, loc="best", ncol=4, fontsize=11)
ax.grid(True, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_overview_{SIM_TAG}.png"), dpi=300)
plt.close()

# --- Plot 2: Temperature strati RTC ---
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(t_h, out["Ta_C"], label="Ta (asfalto)",    color="gray")
ax.plot(t_h, out["Tb_C"], label="Tb (cls tubi)",   color="orange")
ax.plot(t_h, out["Tc_C"], label="Tc (cls inf.)",   color="brown")
ax.plot(t_h, out["Td_C"], label="Td (sottofondo)", color="purple")
ax.set_ylabel("Temperatura [°C]"); ax.set_xlabel("Tempo [giorni]")
ax.set_title("Temperature strati RTC  [MPC v3 — Brutsaert]")
ax.legend(loc="best"); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_strati_{SIM_TAG}.png"), dpi=300)
plt.close()

# --- Plot 3: Convergenza Picard + reward MPC ---
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

ax = axes[0]
n_iter_arr = np.array(out["n_iter"], dtype=float)
bar_colors = [colors_mode[m] for m in out["mode"]]
ax.bar(t_h, n_iter_arr, width=1.0/24, align='edge', color=bar_colors, alpha=0.85)
ax.axhline(MAX_PICARD_EXEC, color='firebrick', linestyle='--',
           label=f'max_iter={MAX_PICARD_EXEC}')
ax.set_ylabel("N° iterazioni"); ax.set_title("Convergenza Picard per ora")
patches = [mpatches.Patch(color=col, label=m) for m, col in colors_mode.items()]
ax.legend(handles=patches + [mpatches.Patch(color='firebrick',
          label=f'max_iter={MAX_PICARD_EXEC}')], ncol=3, fontsize=11)
ax.grid(True, axis='y')

ax = axes[1]
ax.plot(t_h, out["mpc_best_reward"], color="darkgreen", linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.8)
ax.fill_between(t_h, 0, out["mpc_best_reward"],
                where=np.array(out["mpc_best_reward"]) >= 0,
                alpha=0.3, color='green', label='reward positivo')
ax.fill_between(t_h, 0, out["mpc_best_reward"],
                where=np.array(out["mpc_best_reward"]) < 0,
                alpha=0.3, color='red', label='reward negativo (penalità)')
ax.set_ylabel("Reward MPC [kWh-eq]")
ax.set_title(f"Reward cumulativo MPC  (orizzonte {H_MPC}h)")
ax.legend(fontsize=11); ax.grid(True)

ax = axes[2]
ax.bar(t_h, out["mpc_n_candidates"], width=1.0/24, align='edge',
       color=[colors_mode[m] for m in out["mode"]], alpha=0.85)
ax.set_ylabel("N° candidati valutati"); ax.set_xlabel("Tempo [giorni]")
ax.set_title("Modalità candidate valutate per ora")
ax.set_yticks([1, 2, 3]); ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_convergenza_{SIM_TAG}.png"), dpi=300)
plt.close()

# --- Plot 4: Profilo spaziale Tw(z) ---
z_axis        = np.arange(1, n_z + 1) * Delta_z
Tw_spatial_arr= np.array(Tw_spatial)
Q_rtc_arr     = np.array(out["Q_rtc_W"])
T_btes_arr    = np.array(out["T_btes_C"])

ora_Q_max = int(np.nanargmax(Q_rtc_arr))
ora_T_max = int(np.argmax(T_btes_arr))
ora_T_min = int(np.argmin(T_btes_arr))
ora_meta  = n_ore // 2
idle_idx  = np.where(modes_arr == 'idle')[0]
ora_idle  = int(idle_idx[len(idle_idx)//2]) if len(idle_idx) > 0 else None

ore_sel = {
    f"Q_rtc max ({Q_rtc_arr[ora_Q_max]/1000:.1f} kW, ora {ora_Q_max+1})":
        (ora_Q_max, 'tomato', '-'),
    f"T_BTES max ({T_btes_arr[ora_T_max]:.1f}°C, ora {ora_T_max+1})":
        (ora_T_max, 'darkorange', '--'),
    f"T_BTES min ({T_btes_arr[ora_T_min]:.1f}°C, ora {ora_T_min+1})":
        (ora_T_min, 'steelblue', '--'),
    f"Metà simulazione (ora {ora_meta+1})":
        (ora_meta, 'forestgreen', ':'),
}
if ora_idle is not None:
    ore_sel[f"Idle — riferimento (ora {ora_idle+1})"] = (ora_idle, 'gray', ':')

fig, ax = plt.subplots(figsize=(14, 6))
for label, (ora_p, col, ls) in ore_sel.items():
    if not np.all(np.isnan(Tw_spatial_arr[ora_p])):
        ax.plot(z_axis, Tw_spatial_arr[ora_p], label=label, color=col, linestyle=ls)
ax.set_xlabel("Posizione lungo il collettore z [m]")
ax.set_ylabel("Temperatura fluido Tw [°C]")
ax.set_title("Profilo spaziale Tw(z) a ore selezionate  [MPC v1]")
ax.legend(loc="best", fontsize=11); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_profilo_{SIM_TAG}.png"), dpi=300)
plt.close()

# --- Plot 5: Bilancio energetico cumulativo ---
E_btes_cum   = np.cumsum([max(q,0)/1000 if m=='charge'    else 0
                           for q,m in zip(out["Q_rtc_W"], out["mode"])])
E_dc_cum     = np.cumsum([q/1000 if m in ('solar_dc','discharge') else 0
                           for q,m in zip(out["Q_dc_W"],  out["mode"])])
E_scarica_cum= np.cumsum([q/1000 if m=='discharge' else 0
                           for q,m in zip(out["Q_dc_W"],  out["mode"])])

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(t_h, E_btes_cum,    label="Energia cumulativa BTES (carica) [kWh]",   color="darkorange")
ax.plot(t_h, E_dc_cum,      label="Energia cumulativa Dry Cooler [kWh]",      color="tomato")
ax.plot(t_h, E_scarica_cum, label="Energia cumulativa BTES (scarica) [kWh]",  color="steelblue", linestyle="--")
ax.set_ylabel("Energia [kWh]"); ax.set_xlabel("Tempo [giorni]")
ax.set_title("Bilancio energetico cumulativo  [MPC v3 — Brutsaert]")
ax.legend(loc="best", fontsize=11); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_energia_{SIM_TAG}.png"), dpi=300)
plt.close()

# --- Plot 6: Confronto MPC vs Scheduler a Soglie ---
mpc_reward_arr = np.array(out["mpc_best_reward"])
thr_reward_arr = np.array(out["threshold_reward"])
mpc_gain_arr   = np.array(out["mpc_gain"])
mpc_changed    = np.array(out["mpc_changed"])
thr_modes_arr  = np.array(out["threshold_mode"])
cum_gain       = np.cumsum(mpc_gain_arr)

n_changed      = int(mpc_changed.sum())
n_tot          = len(mpc_changed)
gain_tot       = float(mpc_gain_arr.sum())
gain_pos       = float(mpc_gain_arr[mpc_gain_arr > 0].sum())
gain_neg       = float(mpc_gain_arr[mpc_gain_arr < 0].sum())

fig, axes = plt.subplots(4, 1, figsize=(18, 22), sharex=True)

# Panel 1: reward orario MPC vs soglie
ax = axes[0]
ax.plot(t_h, mpc_reward_arr, color='darkgreen',  linewidth=1.2, label='Reward MPC v3 — Brutsaert (rollout)')
ax.plot(t_h, thr_reward_arr, color='firebrick',  linewidth=1.2,
        linestyle='--', alpha=0.8, label='Reward soglie v3 (rollout)')
# Evidenzia le ore in cui MPC ha scelto diversamente
if mpc_changed.any():
    ax.scatter(t_h[mpc_changed], mpc_reward_arr[mpc_changed],
               color='gold', s=30, zorder=5, label=f'MPC ≠ soglie ({n_changed} ore)')
ax.axhline(0, color='k', linewidth=0.7)
ax.set_ylabel("Reward [kWh-eq / 24h]")
ax.set_title(f"Confronto MPC vs Scheduler a Soglie  —  "
             f"ore con scelta diversa: {n_changed}/{n_tot} "
             f"({100*n_changed/n_tot:.1f}%)")
ax.legend(loc='best', fontsize=11)
ax.grid(True)

# Panel 2: guadagno orario MPC − soglie
ax = axes[1]
ax.bar(t_h[mpc_gain_arr >= 0], mpc_gain_arr[mpc_gain_arr >= 0],
       width=1.0/24, align='edge', color='steelblue', alpha=0.85,
       label='Guadagno MPC > 0')
ax.bar(t_h[mpc_gain_arr < 0], mpc_gain_arr[mpc_gain_arr < 0],
       width=1.0/24, align='edge', color='tomato', alpha=0.85,
       label='Guadagno MPC < 0')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_ylabel("Guadagno orario [kWh-eq]")
ax.set_title(f"Guadagno orario MPC − Soglie  "
             f"(positivo={gain_pos:.2f}, negativo={gain_neg:.2f}, netto={gain_tot:.2f} kWh-eq)")
ax.legend(loc='best', fontsize=11)
ax.grid(True)

# Panel 3: guadagno cumulativo
ax = axes[2]
ax.plot(t_h, cum_gain, color='darkblue', linewidth=2)
ax.fill_between(t_h, 0, cum_gain,
                where=cum_gain >= 0, alpha=0.25, color='steelblue',
                label='Guadagno cumulativo positivo')
ax.fill_between(t_h, 0, cum_gain,
                where=cum_gain < 0,  alpha=0.25, color='tomato',
                label='Guadagno cumulativo negativo')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_ylabel("Guadagno cumulativo [kWh-eq]")
ax.set_title(f"Guadagno cumulativo MPC vs Soglie  "
             f"(valore finale: {cum_gain[-1]:+.2f} kWh-eq)")
ax.legend(loc='best', fontsize=11)
ax.grid(True)

# Panel 4: mappa oraria delle decisioni divergenti
ax = axes[3]
# Mostra la modalità MPC (barre) e segna con X quelle diverse dalle soglie
mode_num  = {'charge': 3, 'solar_dc': 2, 'discharge': 1, 'idle': 0}
thr_num   = {'charge': 3, 'solar_dc': 2, 'discharge': 1, 'idle': 0}
mode_vals = np.array([mode_num[m] for m in out["mode"]], dtype=float)
thr_vals  = np.array([thr_num[m]  for m in out["threshold_mode"]], dtype=float)

ax.bar(t_h, mode_vals, width=1.0/24, align='edge',
       color=[colors_mode[m] for m in out["mode"]], alpha=0.75, label='Modalità MPC')
# Puntini neri per le ore in cui soglie avrebbero scelto diversamente
if mpc_changed.any():
    ax.scatter(t_h[mpc_changed] + 0.5/24, thr_vals[mpc_changed],
               color='black', s=20, zorder=6, marker='x',
               label=f'Soglie avrebbero scelto diversamente')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['idle', 'discharge', 'solar_dc', 'charge'])
ax.set_ylabel("Modalità")
ax.set_xlabel("Tempo [giorni]")
ax.set_title("Modalità MPC (barre) vs Soglie (×) — le × segnano le divergenze")
patches = [mpatches.Patch(color=col, label=m) for m, col in colors_mode.items()]
ax.legend(handles=patches + [
    mpatches.Patch(color='black', label=f'Divergenza MPC/soglie ({n_changed} ore)')
], loc='best', ncol=3, fontsize=10)
ax.grid(True, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"MPC_v1_confronto_soglie_{SIM_TAG}.png"), dpi=300)
plt.close()

# Stampa riepilogo confronto
print("\n--- CONFRONTO MPC vs SCHEDULER A SOGLIE ---")
print(f"Ore con scelta identica:   {n_tot - n_changed:4d} / {n_tot}  ({100*(n_tot-n_changed)/n_tot:.1f}%)")
print(f"Ore con scelta diversa:    {n_changed:4d} / {n_tot}  ({100*n_changed/n_tot:.1f}%)")
print(f"Guadagno reward netto MPC: {gain_tot:+.2f} kWh-eq  "
      f"(+{gain_pos:.2f} / {gain_neg:.2f})")
print(f"Guadagno medio per ora:    {gain_tot/n_tot:+.4f} kWh-eq/h")
# Dettaglio per tipo di divergenza
print("\nDettaglio divergenze MPC → Soglie:")
for m_mpc in ['charge', 'solar_dc', 'discharge', 'idle']:
    for m_thr in ['charge', 'solar_dc', 'discharge', 'idle']:
        if m_mpc == m_thr:
            continue
        mask = (modes_arr == m_mpc) & (thr_modes_arr == m_thr)
        cnt  = int(mask.sum())
        if cnt > 0:
            g = float(mpc_gain_arr[mask].sum())
            print(f"  MPC={m_mpc:10s} vs Soglie={m_thr:10s} : {cnt:3d} ore, "
                  f"guadagno totale {g:+.3f} kWh-eq")

print(f"\nOutput salvato in: {output_dir}")
print(f"Tempo totale: {(time.perf_counter()-t_start)/60:.1f} min")
