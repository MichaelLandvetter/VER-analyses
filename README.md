# VER Analysis Program

This repository contains a modular Python program for VER (Visually Evoked Response) analysis. It replays a raw EEG text file at 250 Hz, reads live Waveshare ADS1256 data on a Raspberry Pi, or streams from any USB-serial microcontroller (e.g. Raspberry Pi Pico, Arduino, Teensy), performs trigger-locked epoch averaging, computes wavelet scalograms, and generates a final summary report.

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
For live mode, switch **Source** to **Waveshare Live (CH0/CH1 @ 250 Hz)**.

## Module Overview

- `ver_config.py`: all tunable parameters (acquisition, file columns, filter, epochs, wavelet).
- `ver_acquisition.py`: file replay, Waveshare ADS1256 live acquisition, and USB serial microcontroller sources.
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

Use the **Source** dropdown to select file replay, Waveshare live mode, or USB Serial mode.
For file replay, use the speed dropdown to run replay faster than real-time.
Sampling rate remains at 250 Hz for all calculations.

## USB Serial Live Mode (microcontroller)

Connect any USB microcontroller (Raspberry Pi Pico, Arduino, Teensy, etc.) that samples the EEG and trigger signals and streams them to the PC over USB-CDC serial.

### Firmware output protocol

The microcontroller must print **one ASCII line per sample** at 115 200 baud (configurable in `SERIAL_CONFIG`):

```
<trigger>,<eeg_volts>
```

- `trigger` — integer: `1` on the sample where a flash/stimulus occurred, `0` otherwise.
- `eeg_volts` — floating-point voltage already scaled to volts by the ADC driver on the microcontroller.

Example lines at 250 Hz:

```
0,0.1234
0,-0.0503
1,0.0021
0,-0.0318
```

### Running

1. Flash your microcontroller firmware and connect it via USB.
2. Select **USB Serial (microcontroller)** from the **Source** dropdown.
3. The port combo is populated automatically; click **⟳** to refresh if the device was connected after startup.
4. Select the correct port (e.g. `COM3` on Windows, `/dev/ttyACM0` on Linux) and press **Start**.

Default baud rate and timeout are in `SERIAL_CONFIG` inside `ver_config.py`.

## Waveshare Live Mode

- Default channels: CH0 = EEG, CH1 = trigger.
- Trigger uses Schmitt-style hysteresis for robust edge detection in noisy environments:
  - `HARDWARE_CONFIG["trigger_high_threshold"]` (arm high)
  - `HARDWARE_CONFIG["trigger_low_threshold"]` (re-arm low)
  - `HARDWARE_CONFIG["trigger_min_interval_ms"]` (minimum interval between accepted trigger pulses)
- `HARDWARE_CONFIG["trigger_threshold"]` is retained for compatibility.
- The ADS1256 data-rate default is set to 500 SPS in `HARDWARE_CONFIG`, which provides an effective 250 Hz output per channel when alternating CH0/CH1 reads.
- SPI speed is configured in `Waveshare/config.py` (`SPI.max_speed_hz = 1000000`).

## Output Report

The final report is saved under `Reports/<input-file-stem>/` next to the selected input file and includes:

- minute-by-minute VER averages arranged sequentially across a wide top panel,
- individual minute wavelet scalograms shown sequentially in one wide panel,
- per-minute wavelet peak statistics (peak frequency, latency, and power),
- any final partial minute that reaches at least 50% of the required flashes.

Files are saved as:

- `<input-file-stem>.png`
- `<input-file-stem>.pdf`
