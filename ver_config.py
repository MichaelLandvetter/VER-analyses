"""Configuration for the VER analysis application."""

ACQ_CONFIG = {
    "sample_rate": 250,
    "simulate_realtime": True,
}

FILE_CONFIG = {
    "delimiter": "\t",
    "trigger_column": 0,
    "eeg_column": 2,
    "skip_header": 0,
    "trigger_mode": "threshold",
    "trigger_threshold": 0.5,
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

WAVELET_CONFIG = {
    "wavelet": "cmor1.5-1.0",
    "freq_min": 5.0,
    "freq_max": 50.0,
    "num_freqs": 50,
}

DISPLAY_CONFIG = {
    "scroll_seconds": 10,
    "max_epoch_overlays": 120,
}
