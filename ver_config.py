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
}

FILTER_CONFIG = {
    "lowcut_hz": 13.0,
    "highcut_hz": 30.0,
    "order": 4,
    "sample_rate": ACQ_CONFIG["sample_rate"],
}

EPOCH_CONFIG = {
    "pre_stim_ms": 100,
    "post_stim_ms": 400,
    "flashes_per_session": 120,
    "num_sessions": 10,
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
