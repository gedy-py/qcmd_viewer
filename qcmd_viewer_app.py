import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from scipy.signal import savgol_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors

# ══════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════

# Palette qualitative pour les fichiers multiples
FILE_PALETTE = [
    "#2253A2", "#A71B11", "#2E8B22", "#8B5A22",
    "#6A22A7", "#22A79A", "#C47A1E", "#225B8B",
    "#A72268", "#5A8B22",
]

# Palette qualitative pour les étapes (graphe ΔD vs Δf)
STEP_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ══════════════════════════════════════════════════
#  Fonctions utilitaires
# ══════════════════════════════════════════════════

def seconds_to_hhmm(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"

def seconds_to_hhmmss(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def hhmm_to_seconds(clock):
    """Accepte hh:mm ou hh:mm:ss. Retourne None si vide, lève ValueError sinon."""
    if not clock or not str(clock).strip():
        return None
    parts = str(clock).strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid format '{clock}' — expected hh:mm or hh:mm:ss")
    if len(parts) == 2:
        return parts[0] * 3600 + parts[1] * 60
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Invalid format '{clock}' — expected hh:mm or hh:mm:ss")

def rgb01_to_str(t):
    """Tuple RGB 0-1 → chaîne plotly 'rgb(r,g,b)'."""
    return f"rgb({int(t[0]*255)},{int(t[1]*255)},{int(t[2]*255)})"

def make_shades(base_hex, n):
    """n teintes d'une couleur de base (clair → foncé), au format plotly."""
    if n <= 0:
        return []
    rgb = np.array(pcolors.hex_to_rgb(base_hex)) / 255.0
    if n == 1:
        return [rgb01_to_str(rgb)]
    return [rgb01_to_str(np.ones(3) * (1 - t) + rgb * t)
            for t in np.linspace(0.45, 1.0, n)]

def default_scale(name, n):
    """Échantillonne une colorscale plotly (ex: 'Blues') en n couleurs."""
    if n <= 0:
        return []
    vals = np.linspace(0.5, 1.0, n) if n > 1 else [0.85]
    return pcolors.sample_colorscale(name, list(vals))

def time_ticks(t0, t1):
    """Graduations adaptatives de l'axe temps (valeurs en secondes, labels hh:mm)."""
    span = max(t1 - t0, 1)
    if   span <= 5 * 60:    step = 60
    elif span <= 20 * 60:   step = 5 * 60
    elif span <= 60 * 60:   step = 10 * 60
    elif span <= 3 * 3600:  step = 30 * 60
    elif span <= 12 * 3600: step = 3600
    else:                   step = 2 * 3600
    start = (int(t0) // step) * step
    vals = list(range(start, int(t1) + step, step))
    return vals, [seconds_to_hhmm(v) for v in vals]

def value_at(df, col, t):
    """Valeur de col au point de données le plus proche du temps t."""
    if col not in df.columns or len(df) == 0:
        return np.nan
    idx = (df["Time [s]"] - t).abs().idxmin()
    return float(df.loc[idx, col])

# ══════════════════════════════════════════════════
#  Chargement & détection
# ══════════════════════════════════════════════════

@st.cache_data
def load_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Lecture/nettoyage d'un fichier QCM-D. Caché par contenu (pas de relecture inutile)."""
    buf = io.BytesIO(file_bytes)
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(buf, sep="\t", skiprows=1, engine="python",
                         decimal=",", encoding="utf-8")
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(buf)
    else:
        raise ValueError("Unsupported format (CSV or XLSX expected).")
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next((c for c in df.columns
                     if "time" in c.lower() or "temps" in c.lower()), None)
    if time_col is None:
        raise KeyError(f"No time column detected. Columns: {df.columns.tolist()}")
    df = df.rename(columns={time_col: "Time [s]"})
    df["Time [s]"] = pd.to_numeric(df["Time [s]"], errors="coerce")
    return df.dropna(subset=["Time [s]"]).reset_index(drop=True)

def detect_harmonics(df):
    fh = sorted([int(c[1:].split()[0]) for c in df.columns
                 if c.startswith("f") and "[Hz]" in c])
    dh = sorted([int(c[1:].split()[0]) for c in df.columns
                 if c.startswith("D") and "[ppm]" in c])
    return fh, dh

def file_metadata(df):
    t = df["Time [s]"]
    dt = t.diff().median()
    fh, dh = detect_harmonics(df)
    return {
        "Duration":      seconds_to_hhmmss(t.max() - t.min()),
        "Data points":   f"{len(df):,}",
        "Sampling rate": f"~{1/dt:.2f} Hz" if dt and dt > 0 else "N/A",
        "Δf harmonics":  ", ".join(map(str, fh)) or "—",
        "ΔD harmonics":  ", ".join(map(str, dh)) or "—",
    }

# ══════════════════════════════════════════════════
#  Traitement (normalisation + lissage)
# ══════════════════════════════════════════════════

def process(df, freq_sel, diss_sel, normalize=False, smooth=False, win=21, poly=1):
    """Applique normalisation Δf/n puis lissage Savitzky-Golay sur les colonnes utilisées."""
    d = df.copy()
    warns = []

    if normalize:
        for n in freq_sel:
            col = f"f{n} [Hz]"
            if col in d.columns and n != 0:
                d[col] = d[col] / n

    n_pts = len(d)
    if smooth and n_pts >= 3:
        eff = min(win, n_pts if n_pts % 2 == 1 else n_pts - 1)
        eff = max(eff, poly + 1)
        if eff % 2 == 0:
            eff -= 1
        eff = max(eff, 3)
        if eff != win:
            warns.append(f"Smoothing window adjusted from {win} to {eff} ({n_pts} pts).")
        for n in freq_sel:
            col = f"f{n} [Hz]"
            if col in d.columns:
                d[col] = savgol_filter(d[col].values, eff, poly)
        for n in diss_sel:
            col = f"D{n} [ppm]"
            if col in d.columns:
                d[col] = savgol_filter(d[col].values, eff, poly)

    return d, warns

# ══════════════════════════════════════════════════
#  Construction des figures Plotly
# ══════════════════════════════════════════════════

def build_timeseries(plots, freq_sel, diss_sel, steps, normalize,
                     freq_lim=None, diss_lim=None, show_legend=True, title=""):
    multi = len(plots) > 1
    has_f, has_d = bool(freq_sel), bool(diss_sel)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for fi, p in enumerate(plots):
        df = p["df"]
        # Choix des couleurs
        if not multi:
            if p["custom_colors"]:
                sf = make_shades(p["f_color"], len(freq_sel))
                sd = make_shades(p["d_color"], len(diss_sel))
            else:
                sf = default_scale("Blues", len(freq_sel))
                sd = default_scale("Reds",  len(diss_sel))
        else:
            sf = make_shades(p["color"], len(freq_sel))
            sd = make_shades(p["color"], len(diss_sel))

        if has_f:
            for i, n in enumerate(freq_sel):
                col = f"f{n} [Hz]"
                if col not in df.columns:
                    continue
                name = f"{'Δf/n' if normalize else 'Δf'}{n}" + (f" — {p['label']}" if multi else "")
                fig.add_trace(go.Scatter(
                    x=df["Time [s]"], y=df[col], name=name, mode="lines",
                    line=dict(color=sf[i], width=1.6),
                    hovertemplate="%{customdata}<br>%{y:.3f} Hz<extra></extra>",
                    customdata=[seconds_to_hhmmss(t) for t in df["Time [s]"]],
                ), secondary_y=False)

        if has_d:
            for i, n in enumerate(diss_sel):
                col = f"D{n} [ppm]"
                if col not in df.columns:
                    continue
                name = f"ΔD{n}" + (f" — {p['label']}" if multi else "")
                fig.add_trace(go.Scatter(
                    x=df["Time [s]"], y=df[col], name=name, mode="lines",
                    line=dict(color=sd[i], width=1.6, dash="dash" if multi else "solid"),
                    opacity=0.85,
                    hovertemplate="%{customdata}<br>%{y:.4f} ppm<extra></extra>",
                    customdata=[seconds_to_hhmmss(t) for t in df["Time [s]"]],
                ), secondary_y=True)

    # Lignes verticales des marqueurs + noms d'étapes
    boundaries = sorted({s["start"] for s in steps} | {s["stop"] for s in steps})
    for b in boundaries:
        fig.add_vline(x=b, line_dash="dot", line_width=1, line_color="gray")
    for s in steps:
        fig.add_annotation(x=(s["start"] + s["stop"]) / 2, y=1.0, yref="paper",
                           text=s["name"], showarrow=False,
                           font=dict(size=11, color="black"), yanchor="bottom")

    # Axe temps adaptatif
    all_t = [t for p in plots for t in p["df"]["Time [s]"].tolist()]
    if all_t:
        vals, text = time_ticks(min(all_t), max(all_t))
        fig.update_xaxes(tickvals=vals, ticktext=text, title_text="Time (hh:mm)")

    if has_f:
        fig.update_yaxes(title_text="Δf/n [Hz]" if normalize else "Frequency shift [Hz]",
                         secondary_y=False, color=None if multi else "#2253A2",
                         range=list(freq_lim) if freq_lim else None)
    if has_d:
        fig.update_yaxes(title_text="Dissipation shift [ppm]",
                         secondary_y=True, color=None if multi else "#A71B11",
                         range=list(diss_lim) if diss_lim else None)

    fig.update_layout(
        title=title, showlegend=show_legend, height=520,
        margin=dict(t=50, b=40, l=10, r=10),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode="zoom", hovermode="closest",
    )
    return fig

def build_ddf(plots, harmonic, steps, normalize, title=""):
    multi = len(plots) > 1
    fig = go.Figure()
    fig.add_hline(y=0, line_width=0.8, line_color="lightgray")
    fig.add_vline(x=0, line_width=0.8, line_color="lightgray")

    fc, dc = f"f{harmonic} [Hz]", f"D{harmonic} [ppm]"

    for fi, p in enumerate(plots):
        df = p["df"]
        if fc not in df.columns or dc not in df.columns:
            continue
        prefix = f"{p['label']} — " if multi else ""

        if not steps:
            fig.add_trace(go.Scatter(
                x=df[fc], y=df[dc], mode="lines", name=f"{prefix}n={harmonic}",
                line=dict(color=FILE_PALETTE[fi % len(FILE_PALETTE)] if multi else "#2253A2", width=1.6),
            ))
        else:
            # Trajectoire colorée par étape
            for si, s in enumerate(steps):
                seg = df[(df["Time [s]"] >= s["start"]) & (df["Time [s]"] <= s["stop"])]
                if len(seg) == 0:
                    continue
                color = STEP_PALETTE[si % len(STEP_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=seg[fc], y=seg[dc], mode="lines",
                    name=f"{prefix}{s['name']}",
                    line=dict(color=color, width=1.8),
                ))
            # Gros points aux transitions
            for b in sorted({s["start"] for s in steps} | {s["stop"] for s in steps}):
                fig.add_trace(go.Scatter(
                    x=[value_at(df, fc, b)], y=[value_at(df, dc, b)],
                    mode="markers", showlegend=False,
                    marker=dict(size=11, color="black", symbol="circle-open", line=dict(width=2)),
                    hovertemplate=f"transition @ {seconds_to_hhmmss(b)}<extra></extra>",
                ))

    fig.update_xaxes(title_text="Δf/n [Hz]" if normalize else "Δf [Hz]")
    fig.update_yaxes(title_text="ΔD [ppm]")
    fig.update_layout(title=title, height=520, margin=dict(t=50, b=40, l=10, r=10),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
    return fig

# ══════════════════════════════════════════════════
#  Tableau des étapes (étapes en colonnes, propriétés en lignes)
# ══════════════════════════════════════════════════

def step_table(plots, steps, freq_sel, diss_sel, normalize):
    multi = len(plots) > 1
    if not steps:
        return None

    # Ordre des lignes
    row_order = ["Name", "Start", "End", "Duration"]
    for p in plots:
        lab = f" ({p['label']})" if multi else ""
        for n in freq_sel:
            if f"f{n} [Hz]" in p["df"].columns:
                row_order.append(f"Δ{'f/n' if normalize else 'f'}{n} [Hz]{lab}")
        for n in diss_sel:
            if f"D{n} [ppm]" in p["df"].columns:
                row_order.append(f"ΔD{n} [ppm]{lab}")

    data = {}
    for i, s in enumerate(steps):
        col = f"Step {i+1}"
        d = {
            "Name":     s["name"] or f"Step {i+1}",
            "Start":    seconds_to_hhmmss(s["start"]),
            "End":      seconds_to_hhmmss(s["stop"]),
            "Duration": seconds_to_hhmmss(s["stop"] - s["start"]),
        }
        for p in plots:
            df = p["df"]
            lab = f" ({p['label']})" if multi else ""
            for n in freq_sel:
                fcol = f"f{n} [Hz]"
                if fcol in df.columns:
                    v = value_at(df, fcol, s["stop"]) - value_at(df, fcol, s["start"])
                    d[f"Δ{'f/n' if normalize else 'f'}{n} [Hz]{lab}"] = round(v, 4)
            for n in diss_sel:
                dcol = f"D{n} [ppm]"
                if dcol in df.columns:
                    v = value_at(df, dcol, s["stop"]) - value_at(df, dcol, s["start"])
                    d[f"ΔD{n} [ppm]{lab}"] = round(v, 6)
        data[col] = d

    return pd.DataFrame(data).reindex(row_order)

# ══════════════════════════════════════════════════
#  Helpers session_state pour les marqueurs
# ══════════════════════════════════════════════════

def add_marker(t):
    st.session_state.marker_counter += 1
    idx = st.session_state.marker_counter
    st.session_state.markers.append(
        {"id": idx, "time": float(t), "name": f"Step {len(st.session_state.markers)+1}"}
    )

def sorted_steps():
    ms = sorted(st.session_state.markers, key=lambda m: m["time"])
    steps = []
    for i in range(len(ms) - 1):
        steps.append({"name": ms[i]["name"], "start": ms[i]["time"], "stop": ms[i+1]["time"]})
    return steps

def plotly_png_bytes(fig, fmt="png", scale=2):
    """Export serveur via kaleido. Retourne None si indisponible."""
    try:
        return fig.to_image(format=fmt, scale=scale)
    except Exception:
        return None

# ══════════════════════════════════════════════════
#  Interface Streamlit
# ══════════════════════════════════════════════════

st.set_page_config(page_title="QCM-D Viewer", layout="wide")
st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# Session state
st.session_state.setdefault("markers", [])
st.session_state.setdefault("marker_counter", 0)
st.session_state.setdefault("file_settings", {})

st.title("QCM-D Viewer")
st.markdown(
    '<p style="margin-top:-10px; color:gray; font-size:0.9em; font-style:italic;">'
    '📂 <a href="https://github.com/gedy-py/qcmd_viewer" target="_blank">github.com/gedy-py/qcmd_viewer</a></p>',
    unsafe_allow_html=True
)

# ── Sidebar : upload + options globales ───────────
st.sidebar.header("📂 Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV/Excel file(s)", type=["csv", "xlsx"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ Upload one or more QCM-D files in the sidebar to get started.")
    st.stop()

raw_dfs = {}
for uf in uploaded_files:
    try:
        raw_dfs[uf.name] = load_cached(uf.getvalue(), uf.name)
    except Exception as e:
        st.sidebar.error(f"**{uf.name}**: {e}")
if not raw_dfs:
    st.stop()

all_fh = sorted({n for df in raw_dfs.values() for n in detect_harmonics(df)[0]})
all_dh = sorted({n for df in raw_dfs.values() for n in detect_harmonics(df)[1]})
is_multi = len(raw_dfs) > 1
default_order = [7, 5, 3, 9, 11, 13, 1]

st.sidebar.header("🎵 Overtones")
with st.sidebar.expander("Frequency shift (Δf)", expanded=True):
    df_def = next((h for h in default_order if h in all_fh), None)
    freq_sel = []
    cc = st.columns(3)
    for i, n in enumerate(all_fh):
        if cc[i % 3].checkbox(f"f{n}", value=(n == df_def), key=f"f{n}"):
            freq_sel.append(n)

with st.sidebar.expander("Dissipation shift (ΔD)", expanded=True):
    dd_def = next((h for h in default_order if h in all_dh), None)
    diss_sel = []
    cc = st.columns(3)
    for i, n in enumerate(all_dh):
        if cc[i % 3].checkbox(f"D{n}", value=(n == dd_def), key=f"D{n}"):
            diss_sel.append(n)

st.sidebar.header("🔬 Processing")
normalize = st.sidebar.checkbox("Normalize Δf/n", value=False,
                                help="Divides each frequency shift by its harmonic number n.")
smooth = st.sidebar.checkbox("Smoothing (Savitzky-Golay)", value=False)
win_len = st.sidebar.slider("Window length", 3, 101, 21, step=2, disabled=not smooth)
poly = st.sidebar.slider("Polyorder", 1, 5, 1, disabled=not smooth)

if not freq_sel and not diss_sel:
    st.warning("⚠️ Select at least one overtone (Δf or ΔD) in the sidebar.")
    st.stop()

# Initialisation des settings par fichier
for idx, fname in enumerate(raw_dfs):
    if fname not in st.session_state.file_settings:
        st.session_state.file_settings[fname] = {
            "label": fname.rsplit(".", 1)[0],
            "color": FILE_PALETTE[idx % len(FILE_PALETTE)],
            "f_color": "#2253A2", "d_color": "#A71B11",
            "custom_colors": False, "enabled": True,
        }

# ── Traitement des fichiers actifs ────────────────
plots, warns = [], []
for idx, (fname, df_raw) in enumerate(raw_dfs.items()):
    fs = st.session_state.file_settings[fname]
    if not fs["enabled"]:
        continue
    d, w = process(df_raw, freq_sel, diss_sel, normalize, smooth, win_len, poly)
    warns += [f"**{fname}**: {x}" for x in w]
    plots.append({**fs, "df": d, "df_raw": df_raw, "fname": fname})

for w in warns:
    st.warning(w)
if not plots:
    st.warning("No file enabled. Enable at least one in the Data tab.")

# ══════════════════════════════════════════════════
#  ONGLETS
# ══════════════════════════════════════════════════

tab_data, tab_ts, tab_ddf = st.tabs(["📂 Data", "📈 Time Series", "🔄 ΔD vs Δf"])

# ─── Onglet Data ──────────────────────────────────
with tab_data:
    st.subheader("Files & appearance")
    for idx, fname in enumerate(raw_dfs):
        fs = st.session_state.file_settings[fname]
        with st.expander(f"{'✅' if fs['enabled'] else '⬜'} {fname}", expanded=(len(raw_dfs) == 1)):
            ca, cb = st.columns([1, 2])
            fs["enabled"] = ca.checkbox("Include", value=fs["enabled"], key=f"en_{fname}")
            fs["label"] = cb.text_input("Label", value=fs["label"], key=f"lbl_{fname}")

            meta = file_metadata(raw_dfs[fname])
            mc = st.columns(len(meta))
            for (k, v), c in zip(meta.items(), mc):
                c.metric(k, v)

            st.divider()
            if is_multi:
                fs["custom_colors"] = st.checkbox("Custom color", value=fs["custom_colors"], key=f"cc_{fname}")
                if fs["custom_colors"]:
                    fs["color"] = st.color_picker("File color", value=fs["color"], key=f"cp_{fname}")
                else:
                    fs["color"] = FILE_PALETTE[idx % len(FILE_PALETTE)]
                    st.caption(f"Auto color: {fs['color']}")
            else:
                fs["custom_colors"] = st.checkbox("Custom colors", value=fs["custom_colors"], key=f"cc_{fname}")
                if fs["custom_colors"]:
                    c1, c2 = st.columns(2)
                    fs["f_color"] = c1.color_picker("Δf", value=fs["f_color"], key=f"fc_{fname}")
                    fs["d_color"] = c2.color_picker("ΔD", value=fs["d_color"], key=f"dc_{fname}")
                else:
                    st.caption("Auto: Blues for Δf, Reds for ΔD")

# ─── Onglet Time Series ───────────────────────────
with tab_ts:
    if not plots:
        st.stop()

    steps = sorted_steps()

    opt1, opt2, opt3 = st.columns([2, 2, 1])
    flim_in = opt1.text_input("Δf axis limits", "", placeholder="min,max")
    dlim_in = opt2.text_input("ΔD axis limits", "", placeholder="min,max")
    show_leg = opt3.checkbox("Legend", value=True)

    def _lim(v):
        try:
            p = v.split(",")
            return (float(p[0]), float(p[1])) if len(p) == 2 else None
        except Exception:
            return None
    freq_lim, diss_lim = _lim(flim_in), _lim(dlim_in)

    fig_ts = build_timeseries(plots, freq_sel, diss_sel, steps, normalize,
                              freq_lim, diss_lim, show_leg)

    event = st.plotly_chart(fig_ts, use_container_width=True,
                            on_select="rerun", selection_mode="points", key="ts_chart")

    # Récupération du clic
    clicked_x = None
    try:
        pts = event["selection"]["points"]
        if pts:
            clicked_x = pts[0]["x"]
    except Exception:
        clicked_x = None

    st.markdown("##### Markers")
    cc1, cc2 = st.columns([3, 1])
    if clicked_x is not None:
        cc1.success(f"Selected on chart: **{seconds_to_hhmmss(clicked_x)}**")
        if cc2.button("➕ Add marker", use_container_width=True):
            add_marker(clicked_x)
            st.rerun()
    else:
        cc1.caption("Click a point on a curve to pick a time, then **Add marker**. Or add manually below.")

    mc1, mc2, mc3 = st.columns([3, 1, 1])
    man_t = mc1.text_input("Manual time (hh:mm or hh:mm:ss)", "", key="man_t", label_visibility="collapsed",
                           placeholder="hh:mm:ss")
    if mc2.button("➕ Add manual", use_container_width=True):
        try:
            t = hhmm_to_seconds(man_t)
            if t is not None:
                add_marker(t)
                st.rerun()
        except ValueError as e:
            st.warning(str(e))
    if mc3.button("🗑️ Clear all", use_container_width=True):
        st.session_state.markers = []
        st.rerun()

    # Liste éditable des marqueurs (indépendants)
    if st.session_state.markers:
        st.caption("Each marker is editable independently. Steps are the intervals between "
                   "chronologically-sorted markers (N markers → N-1 steps).")
        for m in list(st.session_state.markers):
            e1, e2, e3 = st.columns([2, 2, 1])
            new_name = e1.text_input("Name", value=m["name"], key=f"mn_{m['id']}")
            new_time = e2.text_input("Time", value=seconds_to_hhmmss(m["time"]), key=f"mt_{m['id']}")
            m["name"] = new_name
            try:
                tt = hhmm_to_seconds(new_time)
                if tt is not None:
                    m["time"] = float(tt)
            except ValueError:
                e2.warning("hh:mm[:ss]")
            if e3.button("🗑️", key=f"md_{m['id']}"):
                st.session_state.markers = [x for x in st.session_state.markers if x["id"] != m["id"]]
                st.rerun()

    # Tableau des étapes
    steps = sorted_steps()
    if steps:
        st.markdown("##### Step table (Δ)")
        tbl = step_table(plots, steps, freq_sel, diss_sel, normalize)
        st.dataframe(tbl, use_container_width=True)
    else:
        st.info("Add at least two markers to define a step.")

    # Exports
    st.markdown("##### Export")
    ec1, ec2, ec3 = st.columns(3)
    fmt = ec1.selectbox("Figure format", ["png", "svg", "pdf"], key="ts_fmt")
    img = plotly_png_bytes(fig_ts, fmt)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if img:
        ec2.download_button(f"💾 Figure (.{fmt})", data=img,
                            file_name=f"qcmd_timeseries_{ts}.{fmt}",
                            mime=f"image/{fmt}" if fmt != "pdf" else "application/pdf",
                            use_container_width=True)
    else:
        ec2.caption("Server export unavailable — use the 📷 camera icon in the chart toolbar.")
    if steps:
        csv = step_table(plots, steps, freq_sel, diss_sel, normalize).to_csv().encode("utf-8")
        ec3.download_button("📋 Step table (CSV)", data=csv,
                            file_name=f"qcmd_steps_{ts}.csv", mime="text/csv",
                            use_container_width=True)

# ─── Onglet ΔD vs Δf ──────────────────────────────
with tab_ddf:
    if not plots:
        st.stop()
    common_n = [n for n in freq_sel if n in diss_sel]
    if not common_n:
        st.info("ℹ️ Select at least one harmonic present in **both** Δf and ΔD (sidebar) to plot ΔD vs Δf.")
    else:
        harmonic = st.selectbox("Harmonic", common_n,
                                format_func=lambda n: f"n = {n}", key="ddf_n")
        steps = sorted_steps()
        fig_ddf = build_ddf(plots, harmonic, steps, normalize,
                            title=f"ΔD vs Δf — n={harmonic}")
        st.plotly_chart(fig_ddf, use_container_width=True, key="ddf_chart")
        if steps:
            st.caption("Line color = step (see legend). ○ = transition between steps.")
        else:
            st.caption("Add markers in the Time Series tab to color the trajectory by step.")

        ec1, ec2 = st.columns(2)
        fmt2 = ec1.selectbox("Figure format", ["png", "svg", "pdf"], key="ddf_fmt")
        img2 = plotly_png_bytes(fig_ddf, fmt2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if img2:
            ec2.download_button(f"💾 Figure (.{fmt2})", data=img2,
                                file_name=f"qcmd_ddf_n{harmonic}_{ts}.{fmt2}",
                                mime=f"image/{fmt2}" if fmt2 != "pdf" else "application/pdf",
                                use_container_width=True)
        else:
            ec2.caption("Server export unavailable — use the 📷 camera icon in the chart toolbar.")
