# VER Analysis Program

This repository contains a modular Python program for VER (Visually Evoked Response) analysis. It replays a raw EEG text file at 250 Hz to simulate live acquisition, performs trigger-locked epoch averaging, computes wavelet scalograms, and generates a final summary report.

## Installation

### Raspberry Pi 4 (Bookworm)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-pyqt6 python3-scipy python3-numpy python3-matplotlib
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Desktop Python

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python ver_main.py
```

At startup, choose a `.txt` raw data file (for example `RAW_files_combined.txt`).

## Module Overview

- `ver_config.py`: all tunable parameters (acquisition, file columns, filter, epochs, wavelet).
- `ver_acquisition.py`: file-based simulator yielding one sample row at a time at 250 Hz.
- `ver_filter.py`: Butterworth bandpass filter with causal and zero-phase modes.
- `ver_scope.py`: rising-edge trigger detection, epoch extraction, and running/session averages.
- `ver_wavelet.py`: CWT/scalogram computation (`pywt.cwt`) for averaged epochs.
- `ver_display.py`: PyQtGraph live display panels.
- `ver_report.py`: final Matplotlib summary figure export (PNG).
- `ver_main.py`: GUI entry point, controls, menu actions, and acquisition thread orchestration.

## Changing Filter Settings

Use the GUI filter controls (Low cut, High cut, **Apply Filter**) to redesign the filter on the fly without resetting accumulated sessions.

## Data File Formats

Use the **File format** dropdown in the GUI to switch between:

- **SD-card** (5-column format, trigger in column 0, EEG in column 2)
- **LabChart** (2-column format, interval trigger in column 0, EEG in column 1)

Format definitions are centralized in `FILE_FORMATS` inside `ver_config.py`.

## Fast Replay Mode

Use the **Fast mode** checkbox in the Controls panel to run replay faster than real-time.
Sampling rate remains at 250 Hz for all calculations.

## Output Report

The final report is saved under `Reports/<input-file-stem>/` next to the selected input file and includes:

- minute-by-minute VER averages arranged sequentially across a wide top panel,
- individual minute wavelet scalograms shown sequentially in one wide panel,
- per-minute wavelet peak statistics (peak frequency, latency, and power),
- any final partial minute that reaches at least 50% of the required flashes.

Files are saved as:

- `<input-file-stem>.png`
- `<input-file-stem>.pdf`
