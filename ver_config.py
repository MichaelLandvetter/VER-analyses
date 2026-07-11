"""Configuration for the VER analysis application."""

ACQ_CONFIG = {
    "sample_rate": 250,
    "simulate_realtime": True,
    "source_mode": "File",
}

FILE_FORMATS = {
    "SD-card": {
        "delimiter": "\t",
        "trigger_column": 0,
        "eeg_column": 2,
        "skip_header": 0,
        "trigger_mode": "threshold",
        "trigger_threshold": 0.5,
    },
    "LabChart": {
        "delimiter": "\t",
        "trigger_column": 0,
        "eeg_column": 1,
        "skip_header": 0,
        "trigger_mode": "interval",
        "trigger_threshold": 0.1,
    },
}

FILE_CONFIG = dict(FILE_FORMATS["SD-card"])  # active config, updated at runtime when user switches format

FILTER_CONFIG = {
    "lowcut_hz": 12.0,
    "highcut_hz": 32.0,
    "order": 4,
    "sample_rate": ACQ_CONFIG["sample_rate"],
}

EPOCH_CONFIG = {
    "pre_stim_ms": 100,  # fish VER uses -100 ms pre-stimulus window
    "post_stim_ms": 400,
    "flashes_per_session": 120,  # flashes collected per minute
    "num_sessions": 10,          # kept as key name for compatibility; displayed as minutes
}

# --- Peak detection ---
# SNR_THRESHOLD: minimum signal-to-noise ratio for a peak to be classified as a VER.
# BASELINE_START_MS / BASELINE_END_MS: time window (ms, relative to flash) used to calculate baseline noise.
# Current default: -100 to 0 ms (pre-flash). Alternative used by some groups: +250 to +450 ms (post-flash).
SNR_THRESHOLD = 2.0
BASELINE_START_MS = -100
BASELINE_END_MS = 0

SERIAL_CONFIG = {
    # USB serial port the microcontroller is connected to.
    # Windows: "COM3", "COM4", … — Linux/macOS: "/dev/ttyUSB0", "/dev/ttyACM0", …
    "port": "COM4",
    # Baud rate must match the firmware setting on the microcontroller.
    "baud_rate": 921600,
    # readline() timeout in seconds.  If no byte arrives within this window
    # the read returns an empty bytes object and the loop retries.
    "timeout": 2.0,
    # Hysteresis thresholds (normalized 0..1) used after trigger auto-normalization.
    "trigger_high_threshold": 0.7,
    "trigger_low_threshold": 0.3,
}

WAVELET_CONFIG = {
    # --- SHAPE SETTINGS (The Morlet Wavelet Parameters) ---
    "bandwidth": 1,      # Time resolution (Lower = skinnier on X-axis, Higher = wider, 1.5 default)
    "center_freq": 1,    # Frequency focus (Lower = pushes energy down, Higher = pushes up, 2.0 default)

    # --- RESOLUTION SETTINGS (The Canvas) ---
    "num_freqs": 320,      # Smoothness (Higher = smoother gradient, Lower = blocky bands, default 160)
    "freq_min": 1.0,       # Bottom of the Y-axis (Hz), default 2.0
    "freq_max": 100.0,      # Top of the Y-axis (Hz), default 60.0
}

DISPLAY_CONFIG = {
    "scroll_seconds": 10,
    "max_epoch_overlays": 120,
    # Maximum frames per second for the scroll panel redraws.
    # Caps setData calls so fast replay (e.g. 10×) does not overwhelm the Qt paint
    # pipeline on low-power hardware such as a Raspberry Pi 4B.
    "scroll_max_fps": 30,
}
