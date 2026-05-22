"""Data acquisition module for replaying file data as a live stream."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ver_config import ACQ_CONFIG, FILE_CONFIG


class FileAcquisitionSimulator:
    """Replay a raw text file sample-by-sample."""

    def __init__(self, file_path: str, sample_rate: Optional[float] = None, simulate_realtime: Optional[bool] = None):
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate if sample_rate is not None else ACQ_CONFIG["sample_rate"]
        self.simulate_realtime = (
            simulate_realtime if simulate_realtime is not None else ACQ_CONFIG["simulate_realtime"]
        )

    def stream_samples(self) -> Generator[np.ndarray, None, None]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        data = np.loadtxt(
            str(self.file_path),
            delimiter=FILE_CONFIG["delimiter"],
            skiprows=FILE_CONFIG["skip_header"],
            dtype=float,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        sleep_time = 1.0 / float(self.sample_rate)
        for row in data:
            yield row
            if self.simulate_realtime:
                time.sleep(sleep_time)
