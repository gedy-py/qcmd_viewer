import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from datetime import timedelta, datetime

# -------------------------
# Fonctions utilitaires
# -------------------------

def seconds_to_hhmm(seconds):
    td = timedelta(seconds=seconds)
    h = td.seconds // 3600
    m = (td.seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"

def hhmm_to_seconds(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 3600 + m * 60

def load_qcmd_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep="\t", skiprows=1, engine="python", decimal=",", encoding="utf-8")
    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format non supporté (CSV ou XLSX attendu).")
    
    df.columns = [str(c).strip() for c in df.columns]

    # Détection colonne temps
    time_col = None
    for c in df.columns:
        if "time" in c.lower() or "temps" in c.lower() or c.lower().startswith("t"):
            time_col = c
            break
    if time_col is None:
        raise KeyError(f"Aucune colonne temps détectée. Colonnes : {df.columns.tolist()}")
    
    df = df.rename(columns={time_col: "Time [s]"})
    return df

def detect_harmonics(df):
    freq_cols = [c for c in df.columns if c.startswith("f") and "[Hz]" in c]
    diss_cols = [c for c in df.columns if c.startswith("D") and "[ppm]" in c]

    freq_harm = sorted([int(c[1:].split()[0]) for c in freq_cols])
    diss_harm = sorted([int(c[1:].split()[0]) for c in diss_cols])
    return freq_harm, diss_harm

def plot_qcmd(df, freq_selection, diss_selection, smooth=False, window_length=21, polyorder=1,
              t_min=None, t_max=None, freq_limits=None, diss_limits=None,
              title="QCM-D", legend_right=False, figsize=(10,6), dpi=120, steps=0):

    df_plot = df.copy()
    
    # Filtrage temporel
    if t_min is not None:
        df_plot = df_plot[df_plot["Time [s]"] >= t_min]
    if t_max is not None:
        df_plot = df_plot[df_plot["Time [s]"] <= t_max]

    # Conversion Time[s] -> datetime pour mdates
    df_plot["Time_dt"] = [datetime(1900,1,1) + timedelta(seconds=s) for s in df_plot["Time [s]"]]

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax1.set_xlabel("Time (hh:mm)")

    has_freq = len(freq_selection) > 0
    has_diss = len(diss_selection) > 0

    if has_freq:
        colors_freq = cm.Blues(np.linspace(0.5,1.0,len(freq_selection)))
        for i, n in enumerate(freq_selection):
            y = df_plot[f"f{n} [Hz]"].values
            if smooth: y = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            ax1.plot(df_plot["Time_dt"], y, label=f"Δf{n}", color=colors_freq[i])
        ax1.set_ylabel("Frequency shift [Hz]", color="#2253A2")
        if freq_limits: ax1.set_ylim(freq_limits)
        ax1.tick_params(axis='y', labelcolor='#2253A2')

    if has_diss:
        ax2 = ax1.twinx() if has_freq else ax1
        colors_diss = cm.Reds(np.linspace(0.5,1.0,len(diss_selection)))
        for i, n in enumerate(diss_selection):
            y = df_plot[f"D{n} [ppm]"].values
            if smooth: y = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            ax2.plot(df_plot["Time_dt"], y, label=f"ΔD{n}", color=colors_diss[i], alpha=0.65)
        ax2.set_ylabel("Dissipation shift [ppm]", color="#A71B11")
        if diss_limits: ax2.set_ylim(diss_limits)
        ax2.tick_params(axis='y', labelcolor='#A71B11')

    # Formatter l'axe x
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]))
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # Légende combinée
    if has_freq or has_diss:
        lines, labels = ax1.get_legend_handles_labels()
        if has_freq and has_diss:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2

        if legend_right:
            ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.075,0.5))
            plt.tight_layout(rect=[0,0,1,1])
        else:
            ax1.legend(lines, labels, loc='best')
            plt.tight_layout()

    for step in steps:
        start_dt = datetime(1900,1,1) + timedelta(seconds=step["start"])
        stop_dt = datetime(1900,1,1) + timedelta(seconds=step["stop"])

        ax1.axvline(start_dt, linestyle=':', linewidth=0.75, color='black')
        ax1.axvline(stop_dt, linestyle=':', linewidth=0.75, color='black')
        
        text = step["text"]
        ax1.text(
            df_plot["Time_dt"].iloc[0] + timedelta(seconds=(step["start"] + step["stop"]) / 2),
            text_position_z_value, text,
            color='black', ha='center', fontsize='small'
        )
    plt.title(title)
    return fig

# -------------------------
# Streamlit interface
# -------------------------

st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

st.title("QCM-D Viewer")

uploaded_file = st.file_uploader(
    "Upload a CSV/Excel file",
    type=["csv","xlsx"]
)

if uploaded_file:
    try:
        df = load_qcmd_data(uploaded_file)
    except Exception as e:
        st.error(f"Error with the file '{uploaded_file.name}': {e}")
        st.stop()

    freq_h, diss_h = detect_harmonics(df)

    st.sidebar.header("Overtone selection")

    default_order_f = [7, 5, 3, 9, 11, 13, 1]
    default_harm_f = next((h for h in default_order_f if h in freq_h), None)
    
    with st.sidebar.expander("Frequency shift (Δf)", expanded=False):
        freq_selection = []
        cols_freq = st.columns(3)
        for i, n in enumerate(freq_h):
            default_checked_f = (n == default_harm_f)
            col = cols_freq[i % 3]
            if col.checkbox(f"f{n}", value=default_checked_f, key=f"f{n}"):
                freq_selection.append(n)

    default_order_d = [7, 5, 3, 9, 11, 13, 1]
    default_harm_d = next((h for h in default_order_d if h in diss_h), None)

    with st.sidebar.expander("Dissipation shift (ΔD)", expanded=False):
        diss_selection = []
        cols_diss = st.columns(3)
        for i, n in enumerate(diss_h):
            default_checked_d = (n == default_harm_d)
            col = cols_diss[i % 3]
            if col.checkbox(f"D{n}", value=default_checked_d, key=f"D{n}"):
                diss_selection.append(n)

    st.sidebar.header("Options")

    graph_title = st.sidebar.text_input("**Plot title**", value=uploaded_file.name, placeholder="No title")

    with st.sidebar.expander("Axes and legend options"):
        t_min_input = st.text_input("Start time", "", placeholder="hh:mm")
        t_max_input = st.text_input("End time", "", placeholder="hh:mm")
        freq_limits_input = st.text_input("Δf axis limits", "", placeholder="min,max")
        diss_limits_input = st.text_input("ΔD axis limits", "", placeholder="min,max")
        legend_right = st.checkbox("Display legend out of the plot", value=False)

    with st.sidebar.expander("Experimental steps visualisation"):
        add_step = st.checkbox("Add experimental step(s)")
        steps = []
        t_max_seconds = df["Time [s]"].max()
        if add_step:
            num_steps = st.number_input("Number of steps", 1, 10, 1)

            for i in range(int(num_steps)):
                with st.expander(f"Step {i+1}"):
                    text = st.text_input(f"Step {i+1} name", "", placeholder=f"Step {i+1}", key=f"text_{i}")
                    start_hhmm = st.text_input(f"Start time", "", placeholder= "00:00", key=f"start_{i}")
                    stop_hhmm  = st.text_input(f"Stop time", "", placeholder= "00:05", key=f"stop_{i}")

                    # Conversion en secondes
                    try:
                        start_sec = hhmm_to_seconds(start_hhmm)
                    except:
                        start_sec = 0
                    try:
                        stop_sec = hhmm_to_seconds(stop_hhmm)
                    except:
                        stop_sec = start_sec + 1  # éviter un stop avant start

                    steps.append({
                        "text": text,
                        "start": start_sec,
                        "stop": stop_sec
                    })
            text_position_z = st.text_input("Step names position (relative to the left axis)", "", placeholder="by default : 0")
            text_position_z_value = float(text_position_z) if text_position_z else 0

    with st.sidebar.expander("Smoothing"):
        smooth = st.checkbox("Enable smoothing (Savitzky-Golay)", value=False)
        window_length = st.slider("Window length", 3, 101, 21, step=2)
        polyorder = st.slider("Polyorder", 1, 5, 1)

    with st.sidebar.expander("Size and resolution"):
        figsize_input = st.text_input("Size (width,height)", "", placeholder="by default : 10,6")
        dpi_input = st.text_input("Resolution (dpi)","", placeholder="by default : 120")

    # Convertir les entrées
    try:
        t_min = hhmm_to_seconds(t_min_input)
    except:
        t_min = None
    try:
        t_max = hhmm_to_seconds(t_max_input) if t_max_input else None
    except:
        t_max = None
    try:
        freq_limits = tuple(map(float, freq_limits_input.split(","))) if freq_limits_input else None
    except:
        freq_limits = None
    try:
        diss_limits = tuple(map(float, diss_limits_input.split(","))) if diss_limits_input else None
    except:
        diss_limits = None
    try:
        figsize = tuple(map(float, figsize_input.split(","))) if figsize_input else (10,6)
    except:
        figsize = (10,6)
    try:
        dpi = int(dpi_input)
    except:
        dpi = 120

    if not freq_selection and not diss_selection:
        st.warning("Select at least one overtone for Δf or ΔD")
    else:
        fig = plot_qcmd(
            df,
            freq_selection=freq_selection,
            diss_selection=diss_selection,
            smooth=smooth,
            window_length=window_length,
            polyorder=polyorder,
            t_min=t_min,
            t_max=t_max,
            freq_limits=freq_limits,
            diss_limits=diss_limits,
            title=graph_title,
            legend_right=legend_right,
            figsize=figsize,
            dpi=dpi,
            steps=steps
        )
        st.pyplot(fig)

    st.sidebar.header("Export figure")

    export_format = st.sidebar.selectbox(
        "Select export format",
        options=["PNG", "PDF", "SVG", "JPG", "EPS"],
        index=0
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = graph_title if graph_title else f"export_QCMD_Viewer_{timestamp}"
    buf = io.BytesIO()
    fig.savefig(buf, format=export_format.lower(), dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    st.sidebar.download_button(
        label=f"💾 Download figure ({export_format})",
        data=buf,
        file_name=f"{default_filename}.{export_format.lower()}",
        mime=f"image/{'jpeg' if export_format == 'JPG' else export_format.lower()}"
    )