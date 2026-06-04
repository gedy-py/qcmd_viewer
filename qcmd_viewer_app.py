import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
from datetime import timedelta, datetime

# ══════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════

# Palette qualitative pour les fichiers multiples
FILE_PALETTE = [
    "#2253A2", "#A71B11", "#2E8B22", "#8B5A22",
    "#6A22A7", "#22A79A", "#C47A1E", "#225B8B",
    "#A72268", "#5A8B22",
]

# ══════════════════════════════════════════════════
#  Fonctions utilitaires
# ══════════════════════════════════════════════════

def seconds_to_hhmm(s):
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"

def hhmm_to_seconds(hhmm):
    """Retourne None si vide, lève ValueError si format invalide."""
    if not hhmm or not hhmm.strip():
        return None
    p = hhmm.strip().split(":")
    if len(p) != 2:
        raise ValueError(f"Invalid format '{hhmm}' — expected hh:mm")
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60
    except ValueError:
        raise ValueError(f"Invalid format '{hhmm}' — expected hh:mm")

def make_shades(base_hex, n):
    """Génère n teintes d'une couleur, du plus clair au plus foncé."""
    if n <= 0:
        return []
    rgb = np.array(mcolors.to_rgb(base_hex))
    if n == 1:
        return [tuple(rgb)]
    return [tuple(np.ones(3) * (1 - t) + rgb * t)
            for t in np.linspace(0.45, 1.0, n)]

def get_x_axis_params(t_range_s):
    """Retourne (locator, formatter) adapté à la durée visible — axe X adaptatif."""
    if t_range_s <= 5 * 60:
        return mdates.MinuteLocator(interval=1),  mdates.DateFormatter('%H:%M')
    elif t_range_s <= 20 * 60:
        return mdates.MinuteLocator(interval=5),  mdates.DateFormatter('%H:%M')
    elif t_range_s <= 60 * 60:
        return mdates.MinuteLocator(interval=10), mdates.DateFormatter('%H:%M')
    elif t_range_s <= 3 * 3600:
        return mdates.MinuteLocator(interval=30), mdates.DateFormatter('%H:%M')
    elif t_range_s <= 12 * 3600:
        return mdates.HourLocator(interval=1),    mdates.DateFormatter('%H:%M')
    else:
        return mdates.HourLocator(interval=2),    mdates.DateFormatter('%Hh')

def parse_time(val, label):
    """Helper UI : parse hh:mm → secondes, affiche un warning si invalide."""
    if not val:
        return None
    try:
        return hhmm_to_seconds(val)
    except ValueError as e:
        st.warning(f"{label}: {e}")
        return None

def parse_limits(val, label):
    """Helper UI : parse 'min,max' → tuple, affiche un warning si invalide."""
    if not val:
        return None
    try:
        p = val.split(",")
        if len(p) != 2:
            raise ValueError("expected two comma-separated values")
        return tuple(map(float, p))
    except ValueError as e:
        st.warning(f"{label}: {e}")
        return None

# ══════════════════════════════════════════════════
#  Chargement des données
# ══════════════════════════════════════════════════

@st.cache_data
def load_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Lecture et nettoyage d'un fichier QCM-D.
    Mis en cache par contenu de fichier : un re-upload du même fichier ne le relit pas.
    """
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
    time_col = next(
        (c for c in df.columns if "time" in c.lower() or "temps" in c.lower()),
        None
    )
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

def file_metadata(df, filename):
    """Retourne un dict lisible des métadonnées du fichier."""
    t = df["Time [s]"]
    dur = t.max() - t.min()
    dt = t.diff().median()
    fh, dh = detect_harmonics(df)
    return {
        "Duration":      seconds_to_hhmm(dur),
        "Data points":   f"{len(df):,}",
        "Sampling rate": f"~{1/dt:.2f} Hz" if dt and dt > 0 else "N/A",
        "Δf harmonics":  ", ".join(map(str, fh)) or "—",
        "ΔD harmonics":  ", ".join(map(str, dh)) or "—",
    }

# ══════════════════════════════════════════════════
#  Pipeline de traitement
# ══════════════════════════════════════════════════

def process(df, freq_sel, diss_sel,
            t_min=None, t_max=None,
            bl_start=None, bl_end=None,
            normalize=False,
            smooth=False, win=21, poly=1):
    """
    Pipeline dans l'ordre :
      1. Filtrage temporel [t_min, t_max]
      2. Correction de baseline (référence calculée sur df brut)
      3. Normalisation Δf/n
      4. Lissage Savitzky-Golay
    Retourne (df_processed, effective_window, [warnings]).
    """
    d = df.copy()
    warns = []

    # 1. Filtrage temporel
    if t_min is not None:
        d = d[d["Time [s]"] >= t_min]
    if t_max is not None:
        d = d[d["Time [s]"] <= t_max]
    d = d.reset_index(drop=True)

    # 2. Baseline (calculée sur le df ORIGINAL avant filtrage)
    if bl_start is not None and bl_end is not None:
        mask = (df["Time [s]"] >= bl_start) & (df["Time [s]"] <= bl_end)
        ref = df[mask]
        if len(ref) == 0:
            warns.append("Baseline window contains no data points — correction skipped.")
        else:
            for n in freq_sel:
                col = f"f{n} [Hz]"
                if col in d.columns:
                    d[col] = d[col] - ref[col].mean()
            for n in diss_sel:
                col = f"D{n} [ppm]"
                if col in d.columns:
                    d[col] = d[col] - ref[col].mean()

    # 3. Normalisation Δf/n
    if normalize:
        for n in freq_sel:
            col = f"f{n} [Hz]"
            if col in d.columns and n != 0:
                d[col] = d[col] / n

    # 4. Lissage Savitzky-Golay (avec ajustement automatique de la fenêtre)
    n_pts = len(d)
    eff_win = win
    if smooth and n_pts >= 3:
        eff_win = min(win, n_pts if n_pts % 2 == 1 else n_pts - 1)
        eff_win = max(eff_win, poly + 1)
        if eff_win % 2 == 0:
            eff_win -= 1
        eff_win = max(eff_win, 3)
        if eff_win != win:
            warns.append(
                f"Smoothing window adjusted from {win} to {eff_win} ({n_pts} pts in range)."
            )
        for n in freq_sel:
            col = f"f{n} [Hz]"
            if col in d.columns:
                d[col] = savgol_filter(d[col].values, eff_win, poly)
        for n in diss_sel:
            col = f"D{n} [ppm]"
            if col in d.columns:
                d[col] = savgol_filter(d[col].values, eff_win, poly)

    d["Time_dt"] = [datetime(1900, 1, 1) + timedelta(seconds=float(s))
                    for s in d["Time [s]"]]
    return d, eff_win, warns

# ══════════════════════════════════════════════════
#  Graphe : time series
# ══════════════════════════════════════════════════

def plot_timeseries(plots, freq_sel, diss_sel,
                    title="QCM-D", legend_right=False,
                    figsize=(10, 6), dpi=120,
                    freq_lim=None, diss_lim=None,
                    steps=None, z_text=None,
                    normalize=False):
    """
    plots : liste de dicts {df, f_color, d_color, label, custom_colors}
    Mode fichier unique  → Blues/Reds classiques (ou couleurs custom)
    Mode multi-fichiers  → une couleur de base par fichier, ΔD en tirets
    """
    if steps is None:
        steps = []
    multi = len(plots) > 1
    has_f = bool(freq_sel)
    has_d = bool(diss_sel)
    linestyles = ['-', '--', '-.', ':']

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax1.set_xlabel("Time (hh:mm)")
    ax2 = ax1.twinx() if (has_f and has_d) else ax1

    for fi, p in enumerate(plots):
        df   = p['df']
        ls   = linestyles[fi % 4]
        lbl  = p['label']

        # Choix des couleurs selon le mode
        if not multi:
            # Fichier unique : Blues/Reds sauf si couleurs custom
            if p.get('custom_colors'):
                sf = make_shades(p['f_color'], len(freq_sel))
                sd = make_shades(p['d_color'], len(diss_sel))
            else:
                sf = [cm.Blues(v) for v in np.linspace(0.5, 1.0, max(len(freq_sel), 1))]
                sd = [cm.Reds(v)  for v in np.linspace(0.5, 1.0, max(len(diss_sel), 1))]
        else:
            # Multi-fichiers : shades de la couleur du fichier
            sf = make_shades(p['color'], len(freq_sel))
            sd = make_shades(p['color'], len(diss_sel))

        if has_f:
            for i, n in enumerate(freq_sel):
                col = f"f{n} [Hz]"
                if col not in df.columns:
                    continue
                name = f"{'Δf/n' if normalize else 'Δf'}{n}"
                if multi:
                    name = f"{lbl} — {name}"
                ax1.plot(df["Time_dt"], df[col], label=name,
                         color=sf[i], linestyle=ls, linewidth=1.2)

        if has_d:
            for i, n in enumerate(diss_sel):
                col = f"D{n} [ppm]"
                if col not in df.columns:
                    continue
                name = f"ΔD{n}"
                if multi:
                    name = f"{lbl} — {name}"
                # En mode multi : tirets pour distinguer visuellement Δf de ΔD
                dls = '--' if multi else ls
                ax2.plot(df["Time_dt"], df[col], label=name,
                         color=sd[i], linestyle=dls, linewidth=1.2, alpha=0.75)

    # Styles des axes
    if has_f:
        ax1.set_ylabel("Δf/n [Hz]" if normalize else "Frequency shift [Hz]",
                       color="black" if multi else "#2253A2")
        if not multi:
            ax1.tick_params(axis='y', labelcolor='#2253A2')
        if freq_lim:
            ax1.set_ylim(freq_lim)

    if has_d:
        ax2.set_ylabel("Dissipation shift [ppm]",
                       color="black" if multi else "#A71B11")
        if not multi:
            ax2.tick_params(axis='y', labelcolor='#A71B11')
        if diss_lim:
            ax2.set_ylim(diss_lim)

    # Axe X adaptatif
    all_t = [s for p in plots for s in p['df']["Time [s]"].tolist()]
    if all_t:
        loc, fmt = get_x_axis_params(max(all_t) - min(all_t))
        ax1.xaxis.set_major_locator(loc)
        ax1.xaxis.set_major_formatter(fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Légende
    lines, lbls = ax1.get_legend_handles_labels()
    if has_f and has_d and ax2 is not ax1:
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2; lbls += lb2
    if lines:
        if legend_right:
            ax1.legend(lines, lbls, loc='center left',
                       bbox_to_anchor=(1.15, 0.5), fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.82, 1])
        else:
            ax1.legend(lines, lbls, loc='best', fontsize='small')
            plt.tight_layout()
    else:
        plt.tight_layout()

    # Steps : position Z par défaut = max de l'axe gauche (après tight_layout)
    if z_text is None:
        z_text = ax1.get_ylim()[1]
    for s in steps:
        # [FIX 5] Coordonnée X reconstruite depuis l'epoch datetime(1900,1,1)
        sd = datetime(1900, 1, 1) + timedelta(seconds=s["start"])
        ed = datetime(1900, 1, 1) + timedelta(seconds=s["stop"])
        tx = datetime(1900, 1, 1) + timedelta(seconds=(s["start"] + s["stop"]) / 2)
        ax1.axvline(sd, linestyle=':', linewidth=0.75, color='black')
        ax1.axvline(ed, linestyle=':', linewidth=0.75, color='black')
        ax1.text(tx, z_text, s["text"],
                 ha='center', va='top', fontsize='small', color='black')

    plt.title(title)
    return fig

# ══════════════════════════════════════════════════
#  Graphe : ΔD vs Δf
# ══════════════════════════════════════════════════

def plot_dd_vs_df(plots, freq_sel, diss_sel,
                  title="ΔD vs Δf", figsize=(7, 6), dpi=120, normalize=False):
    """
    Graphe paramétrique (couleur = progression temporelle via gradient sur la ligne).
    Seuls les harmoniques présents à la fois dans freq_sel et diss_sel sont tracés.
    """
    common_n = [n for n in freq_sel if n in diss_sel]
    if not common_n:
        return None

    multi = len(plots) > 1
    linestyles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlabel("Δf/n [Hz]" if normalize else "Δf [Hz]")
    ax.set_ylabel("ΔD [ppm]")
    ax.axhline(0, color='lightgray', linewidth=0.8, zorder=0)
    ax.axvline(0, color='lightgray', linewidth=0.8, zorder=0)

    start_legend_added = end_legend_added = False
    for fi, p in enumerate(plots):
        df = p['df']
        shades = make_shades(p['color'] if multi else "#2253A2", len(common_n))
        ls = linestyles[fi % 4]
        for i, n in enumerate(common_n):
            fc, dc = f"f{n} [Hz]", f"D{n} [ppm]"
            if fc not in df.columns or dc not in df.columns:
                continue
            x, y = df[fc].values, df[dc].values
            name = f"n={n}" + (f" — {p['label']}" if multi else "")
            ax.plot(x, y, color=shades[i], linestyle=ls, linewidth=1.2, label=name)
            # Marqueurs début (○) et fin (■) : ajoutés une seule fois dans la légende
            slbl = "start" if not start_legend_added else ""
            elbl = "end"   if not end_legend_added   else ""
            ax.plot(x[0],  y[0],  'o', color=shades[i], markersize=5, label=slbl)
            ax.plot(x[-1], y[-1], 's', color=shades[i], markersize=5, label=elbl)
            start_legend_added = end_legend_added = True

    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l]
    if filtered:
        ax.legend(*zip(*filtered), fontsize='small')
    plt.title(title)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════
#  Analyse Δ
# ══════════════════════════════════════════════════

def compute_delta(raw_dfs_list, labels, freq_sel, diss_sel,
                  t1, t2, half_win, normalize,
                  bl_start=None, bl_end=None):
    """
    Calcule la valeur moyenne dans [Ti - half_win, Ti + half_win] pour T1 et T2,
    sur les données BRUTES re-traitées (baseline + normalisation, sans filtrage temporel).
    Retourne une liste de dicts (lignes du tableau).
    """
    rows = []
    for df_raw, label in zip(raw_dfs_list, labels):
        # Re-traitement sans filtrage temporel pour avoir accès à T1/T2 hors fenêtre
        d, _, _ = process(df_raw, freq_sel, diss_sel,
                          t_min=None, t_max=None,
                          bl_start=bl_start, bl_end=bl_end,
                          normalize=normalize,
                          smooth=False)
        m1 = (d["Time [s]"] >= t1 - half_win) & (d["Time [s]"] <= t1 + half_win)
        m2 = (d["Time [s]"] >= t2 - half_win) & (d["Time [s]"] <= t2 + half_win)
        d1, d2 = d[m1], d[m2]

        for n in freq_sel:
            fc  = f"f{n} [Hz]"
            dcc = f"D{n} [ppm]"
            if fc not in d.columns:
                continue

            f1 = d1[fc].mean() if len(d1) else np.nan
            f2 = d2[fc].mean() if len(d2) else np.nan
            v1 = d1[dcc].mean() if (dcc in d.columns and len(d1)) else np.nan
            v2 = d2[dcc].mean() if (dcc in d.columns and len(d2)) else np.nan

            def fmt(v, decimals=4):
                return f"{v:.{decimals}f}" if not np.isnan(v) else "—"

            fl = "Δf/n [Hz]" if normalize else "Δf [Hz]"
            rows.append({
                "File":              label,
                "n":                 n,
                f"{fl} @ T1":        fmt(f1),
                f"{fl} @ T2":        fmt(f2),
                f"Δ({fl})":          fmt(f2 - f1) if not (np.isnan(f1) or np.isnan(f2)) else "—",
                "ΔD [ppm] @ T1":     fmt(v1, 6),
                "ΔD [ppm] @ T2":     fmt(v2, 6),
                "Δ(ΔD) [ppm]":       fmt(v2 - v1, 6) if not (np.isnan(v1) or np.isnan(v2)) else "—",
            })
    return rows

# ══════════════════════════════════════════════════
#  Interface Streamlit
# ══════════════════════════════════════════════════

st.set_page_config(page_title="QCM-D Viewer", layout="wide")
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── Initialisation session state ─────────────────
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0
if "file_settings" not in st.session_state:
    st.session_state.file_settings = {}

# rc = suffixe pour tous les widgets d'analyse.
# Changer rc (via Reset) force la recréation de ces widgets avec leurs valeurs par défaut.
rc = st.session_state.reset_count

# ── Titre ─────────────────────────────────────────
st.title("QCM-D Viewer")
st.markdown(
    '<p style="margin-top:-10px; color:gray; font-size:0.9em; font-style:italic;">'
    '📂 GitHub & sample data : '
    '<a href="https://github.com/gedy-py/qcmd_viewer" target="_blank">'
    'github.com/gedy-py/qcmd_viewer</a></p>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════

st.sidebar.header("📂 Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV/Excel file(s)", type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ Upload one or more QCM-D files to get started.")
    st.stop()

# Chargement (avec cache : pas de relecture si le fichier n'a pas changé)
raw_dfs = {}
for uf in uploaded_files:
    try:
        raw_dfs[uf.name] = load_cached(uf.getvalue(), uf.name)
    except Exception as e:
        st.sidebar.error(f"**{uf.name}**: {e}")

if not raw_dfs:
    st.stop()

# Union de tous les harmoniques détectés dans l'ensemble des fichiers
all_fh = sorted({n for df in raw_dfs.values() for n in detect_harmonics(df)[0]})
all_dh = sorted({n for df in raw_dfs.values() for n in detect_harmonics(df)[1]})
is_multi = len(raw_dfs) > 1

# ── Paramètres par fichier (label, couleur, activé) ──
st.sidebar.markdown("**Files loaded:**")
for idx, fname in enumerate(raw_dfs):
    # Initialisation des settings par défaut lors du premier chargement
    if fname not in st.session_state.file_settings:
        st.session_state.file_settings[fname] = {
            "label":         fname.rsplit(".", 1)[0],
            "color":         FILE_PALETTE[idx % len(FILE_PALETTE)],
            "f_color":       "#2253A2",
            "d_color":       "#A71B11",
            "custom_colors": False,
            "enabled":       True,
        }
    fs = st.session_state.file_settings[fname]

    with st.sidebar.expander(f"{'✅' if fs['enabled'] else '⬜'} {fname}", expanded=False):
        fs["enabled"] = st.checkbox("Include in plot", value=fs["enabled"],
                                     key=f"en_{fname}")
        fs["label"] = st.text_input("Display label", value=fs["label"],
                                     key=f"lbl_{fname}")

        # Métadonnées
        for k, v in file_metadata(raw_dfs[fname], fname).items():
            st.caption(f"**{k}:** {v}")

        st.divider()
        # Options de couleur : deux modes selon contexte
        if is_multi:
            fs["custom_colors"] = st.checkbox("Custom color", value=fs["custom_colors"],
                                               key=f"cc_{fname}")
            if fs["custom_colors"]:
                fs["color"] = st.color_picker("File color", value=fs["color"],
                                               key=f"cp_{fname}")
            else:
                fs["color"] = FILE_PALETTE[idx % len(FILE_PALETTE)]
                st.caption(f"Auto color: {fs['color']}")
        else:
            # Fichier unique : deux color pickers (Δf / ΔD)
            fs["custom_colors"] = st.checkbox("Custom colors", value=fs["custom_colors"],
                                               key=f"cc_{fname}")
            if fs["custom_colors"]:
                c1, c2 = st.columns(2)
                fs["f_color"] = c1.color_picker("Δf", value=fs["f_color"], key=f"fc_{fname}")
                fs["d_color"] = c2.color_picker("ΔD", value=fs["d_color"], key=f"dc_{fname}")
            else:
                st.caption("Auto: Blues for Δf, Reds for ΔD")

# Fichiers actifs
active = {fn: raw_dfs[fn] for fn in raw_dfs
          if st.session_state.file_settings.get(fn, {}).get("enabled", True)}
if not active:
    st.warning("No file is enabled. Enable at least one in the sidebar.")
    st.stop()

# ── Sélection des harmoniques ─────────────────────
st.sidebar.header("🎵 Overtones")
default_order = [7, 5, 3, 9, 11, 13, 1]

with st.sidebar.expander("Frequency shift (Δf)", expanded=False):
    default_f = next((h for h in default_order if h in all_fh), None)
    freq_sel = []
    cols = st.columns(3)
    for i, n in enumerate(all_fh):
        if cols[i % 3].checkbox(f"f{n}", value=(n == default_f), key=f"f{n}_{rc}"):
            freq_sel.append(n)

with st.sidebar.expander("Dissipation shift (ΔD)", expanded=False):
    default_d = next((h for h in default_order if h in all_dh), None)
    diss_sel = []
    cols = st.columns(3)
    for i, n in enumerate(all_dh):
        if cols[i % 3].checkbox(f"D{n}", value=(n == default_d), key=f"D{n}_{rc}"):
            diss_sel.append(n)

# ── Options ───────────────────────────────────────
st.sidebar.header("⚙️ Options")
graph_title = st.sidebar.text_input("Plot title", value="QCM-D", key=f"title_{rc}")

with st.sidebar.expander("⏱ Time window"):
    t_min_in = st.text_input("Start time", "", placeholder="hh:mm", key=f"tmin_{rc}")
    t_max_in = st.text_input("End time",   "", placeholder="hh:mm", key=f"tmax_{rc}")

with st.sidebar.expander("🔬 Processing"):
    normalize = st.checkbox(
        "Normalize Δf/n", value=False, key=f"norm_{rc}",
        help="Divides each frequency shift by its harmonic number n (Sauerbrey analysis)."
    )
    do_baseline = st.checkbox(
        "Baseline correction", value=False, key=f"bl_{rc}",
        help="Subtracts the mean value of each channel over a reference time window."
    )
    bl_start_in = bl_end_in = ""
    if do_baseline:
        st.caption("Reference window (applied on raw data):")
        c1, c2 = st.columns(2)
        bl_start_in = c1.text_input("Start", "", placeholder="hh:mm", key=f"bls_{rc}")
        bl_end_in   = c2.text_input("End",   "", placeholder="hh:mm", key=f"ble_{rc}")

with st.sidebar.expander("📐 Axes & Legend"):
    freq_lim_in = st.text_input("Δf limits", "", placeholder="min,max", key=f"flim_{rc}")
    diss_lim_in = st.text_input("ΔD limits", "", placeholder="min,max", key=f"dlim_{rc}")
    legend_right = st.checkbox("Legend outside plot", value=False, key=f"leg_{rc}")

with st.sidebar.expander("🧪 Experimental steps"):
    add_steps = st.checkbox("Add steps", key=f"addstep_{rc}")
    steps  = []
    z_text = None
    if add_steps:
        n_steps = st.number_input("Number of steps", 1, 10, 1, key=f"nstep_{rc}")
        for i in range(int(n_steps)):
            with st.expander(f"Step {i+1}"):
                stxt = st.text_input("Name", "", key=f"stxt_{i}_{rc}")
                c1, c2 = st.columns(2)
                ss = c1.text_input("Start", "", placeholder="hh:mm", key=f"ss_{i}_{rc}")
                se = c2.text_input("Stop",  "", placeholder="hh:mm", key=f"se_{i}_{rc}")
                try:    ss_s = hhmm_to_seconds(ss) or 0
                except: ss_s = 0
                try:    se_s = hhmm_to_seconds(se) or ss_s + 1
                except: se_s = ss_s + 1
                steps.append({"text": stxt, "start": ss_s, "stop": se_s})

        z_in = st.text_input("Label Y position", "", placeholder="default: top of left axis",
                              key=f"zin_{rc}")
        if z_in:
            try:
                z_text = float(z_in)
            except ValueError:
                st.warning("Y position — numeric value expected.")

        # Avertissement sur l'axe de référence de z_text
        if freq_sel and diss_sel:
            st.info("ℹ️ Y position is in **Hz** (left axis: Δf). Switching to ΔD-only changes unit to ppm.")
        elif diss_sel and not freq_sel:
            st.info("ℹ️ Y position is in **ppm** (left axis: ΔD).")

with st.sidebar.expander("〰️ Smoothing"):
    smooth  = st.checkbox("Savitzky-Golay", value=False, key=f"sm_{rc}")
    win_len = st.slider("Window length", 3, 101, 21, step=2, key=f"wl_{rc}")
    poly    = st.slider("Polyorder", 1, 5, 1, key=f"po_{rc}")

with st.sidebar.expander("🖼 Figure size"):
    fsz_in = st.text_input("Width, height (inches)", "", placeholder="10,6",  key=f"fsz_{rc}")
    dpi_in = st.text_input("Resolution (dpi)",       "", placeholder="120",   key=f"dpi_{rc}")

st.sidebar.divider()
if st.sidebar.button("🔄 Reset analysis settings", key="reset_btn",
                     help="Resets all options to default. File labels and colors are preserved."):
    st.session_state.reset_count += 1
    st.rerun()

# ── Parsing des entrées ───────────────────────────
t_min    = parse_time(t_min_in,    "Start time")
t_max    = parse_time(t_max_in,    "End time")
bl_start = parse_time(bl_start_in, "Baseline start") if do_baseline else None
bl_end   = parse_time(bl_end_in,   "Baseline end")   if do_baseline else None
freq_lim = parse_limits(freq_lim_in, "Δf limits")
diss_lim = parse_limits(diss_lim_in, "ΔD limits")

try:
    figsize = tuple(map(float, fsz_in.split(","))) if fsz_in else (10, 6)
    if len(figsize) != 2:
        raise ValueError()
except Exception:
    st.warning("Figure size: expected width,height (e.g. 10,6)")
    figsize = (10, 6)

try:
    dpi = int(dpi_in) if dpi_in else 120
except ValueError:
    st.warning("DPI: integer expected.")
    dpi = 120

if not freq_sel and not diss_sel:
    st.warning("⚠️ Select at least one overtone (Δf or ΔD) to display.")
    st.stop()

# ── Traitement de chaque fichier actif ───────────
plots = []
all_warnings = []

for fname, df_raw in active.items():
    fs = st.session_state.file_settings[fname]
    df_proc, _, warns = process(
        df_raw, freq_sel, diss_sel,
        t_min=t_min, t_max=t_max,
        bl_start=bl_start, bl_end=bl_end,
        normalize=normalize,
        smooth=smooth, win=win_len, poly=poly
    )
    for w in warns:
        all_warnings.append(f"**{fname}**: {w}")
    plots.append({
        "df":            df_proc,
        "df_raw":        df_raw,
        "color":         fs["color"],
        "f_color":       fs["f_color"],
        "d_color":       fs["d_color"],
        "custom_colors": fs["custom_colors"],
        "label":         fs["label"],
        "fname":         fname,
    })

for w in all_warnings:
    st.warning(w)

# ══════════════════════════════════════════════════
#  ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time series",
    "🔄 ΔD vs Δf",
    "📊 Δ Analysis",
    "💾 Export",
])

# ── Onglet 1 : Time series ────────────────────────
with tab1:
    fig_ts = plot_timeseries(
        plots, freq_sel, diss_sel,
        title=graph_title,
        legend_right=legend_right,
        figsize=figsize, dpi=dpi,
        freq_lim=freq_lim, diss_lim=diss_lim,
        steps=steps, z_text=z_text,
        normalize=normalize,
    )
    st.pyplot(fig_ts)

# ── Onglet 2 : ΔD vs Δf ──────────────────────────
with tab2:
    common_n = [n for n in freq_sel if n in diss_sel]
    if not common_n:
        st.info("ℹ️ Select at least one harmonic present in **both** Δf and ΔD to display this plot.")
    else:
        fig_ddf = plot_dd_vs_df(
            plots, freq_sel, diss_sel,
            title=f"ΔD vs Δf — {graph_title}",
            figsize=(min(figsize[0], 8), figsize[1]),
            dpi=dpi,
            normalize=normalize,
        )
        if fig_ddf:
            st.pyplot(fig_ddf)
            st.caption("**○** = start of measurement  |  **■** = end of measurement")

# ── Onglet 3 : Analyse Δ ─────────────────────────
with tab3:
    st.subheader("Δ Analysis between two time points")
    st.caption(
        "The mean value is computed in a window **[T ± W]** around each time point. "
        "Baseline correction and normalization (if enabled) are applied."
    )

    ca, cb, cc = st.columns(3)
    dt1_in   = ca.text_input("T1", "", placeholder="hh:mm", key=f"dt1_{rc}")
    dt2_in   = cb.text_input("T2", "", placeholder="hh:mm", key=f"dt2_{rc}")
    half_win = cc.number_input("Window ± W (s)", min_value=1, max_value=600,
                                value=30, key=f"hw_{rc}")

    delta_t1 = parse_time(dt1_in, "T1")
    delta_t2 = parse_time(dt2_in, "T2")

    if delta_t1 is not None and delta_t2 is not None:
        if delta_t1 >= delta_t2:
            st.warning("T1 must be strictly before T2.")
        else:
            rows = compute_delta(
                [p['df_raw'] for p in plots],
                [p['label']  for p in plots],
                freq_sel, diss_sel,
                delta_t1, delta_t2, half_win,
                normalize=normalize,
                bl_start=bl_start, bl_end=bl_end,
            )
            if rows:
                df_delta = pd.DataFrame(rows)
                st.dataframe(df_delta, use_container_width=True, hide_index=True)
                csv_delta = df_delta.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download delta table (CSV)", data=csv_delta,
                    file_name="delta_analysis.csv", mime="text/csv"
                )
            else:
                st.warning("No data found around T1 or T2 — check time values and window size.")
    else:
        st.info("Enter T1 and T2 to compute Δ values.")

# ── Onglet 4 : Export ─────────────────────────────
with tab4:
    col_fig, col_csv = st.columns(2)

    with col_fig:
        st.subheader("Export figure")
        exp_fmt = st.selectbox("Format", ["PNG", "PDF", "SVG", "JPG", "EPS"])
        buf = io.BytesIO()
        fig_ts.savefig(buf, format=exp_fmt.lower(), dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            f"💾 Download ({exp_fmt})", data=buf,
            file_name=f"{graph_title or 'qcmd'}_{ts}.{exp_fmt.lower()}",
            mime=f"image/{'jpeg' if exp_fmt == 'JPG' else exp_fmt.lower()}"
        )

    with col_csv:
        st.subheader("Export processed data")
        for p in plots:
            df_exp = p['df'].drop(columns=["Time_dt"], errors="ignore")
            csv_b = df_exp.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ {p['label']} (CSV)", data=csv_b,
                file_name=f"{p['label']}_processed.csv",
                mime="text/csv", key=f"csv_{p['fname']}"
            )
