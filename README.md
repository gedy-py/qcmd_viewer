QCM-D Viewer – Interactive Visualization of Quartz Crystal Microbalance Data

This project provides a user-friendly Streamlit application for visualizing and analyzing Quartz Crystal Microbalance with Dissipation monitoring (QCM-D) data. Designed for researchers and lab technicians, the QCM-D Viewer simplifies the exploration of frequency (Δf) and dissipation (ΔD) shifts across multiple harmonics, allowing detailed inspection of experimental results in real time.

Key features:

Multi-harmonic selection: Easily choose which overtones to display for both frequency and dissipation signals.

Customizable plots: Adjust axes limits, time ranges, figure size, resolution, and smoothing options (Savitzky-Golay filter).

Experimental step annotation: Add labeled vertical lines to highlight specific time intervals in the measurement, with configurable text positioning.

Flexible data input: Supports CSV and Excel files with automatic detection of time and measurement columns.

Export functionality: Save plots in common formats (PNG, PDF, SVG, JPG, EPS) with customizable filenames and timestamps for version tracking.

The application provides an intuitive interface through Streamlit’s sidebar, enabling users with minimal technical background to manipulate data visualization parameters interactively. By combining Python’s scientific libraries (Pandas, NumPy, Matplotlib, SciPy) with Streamlit’s web interface, the QCM-D Viewer allows immediate exploration of QCM-D datasets without the need for coding knowledge.

This tool is ideal for rapid inspection of experimental results, troubleshooting, and preparing figures for reports or publications. The project can be deployed via Streamlit Cloud for easy sharing, enabling collaborators to access the app directly through a web browser without installing Python or any dependencies.

Repository contents:

qcmd_app.py – Main Streamlit application script

requirements.txt – List of required Python packages

Example datasets (optional)

QCM-D Viewer aims to streamline QCM-D data analysis and visualization, making it accessible, interactive, and reproducible for research teams.
