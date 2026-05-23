"""Trigger detection, epoch extraction, and running/session averaging."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ver_config import ACQ_CONFIG, EPOCH_CONFIG


class VERScopeProcessor:
    """Manage ring buffer, rising-edge trigger detection, and averaging."""

    def __init__(self, bandpass_filter, epoch_config: Optional[dict] = None):
        self.bandpass_filter = bandpass_filter
        self.config = dict(EPOCH_CONFIG)
        if epoch_config:
            self.config.update(epoch_config)

        self.sample_rate = ACQ_CONFIG["sample_rate"]
        self.pre_samples = int(self.config["pre_stim_ms"] * self.sample_rate / 1000)
        self.post_samples = int(self.config["post_stim_ms"] * self.sample_rate / 1000)
        self.epoch_samples = self.pre_samples + self.post_samples

        self.epoch_time_ms = (np.arange(self.epoch_samples) / self.sample_rate * 1000.0) - self.config["pre_stim_ms"]
        self.reset()

    def reset(self) -> None:
        self.pre_buffer = deque(maxlen=self.pre_samples)
        self.prev_trigger = 0.0
        self.pending_epochs: List[Dict[str, object]] = []
        self.session_epochs: List[np.ndarray] = []
        self.session_averages: List[np.ndarray] = []
        self.flash_count = 0
        self.session_index = 0
        self.running_average = None

    def _active_session_number(self) -> int:
        return min(self.config["num_sessions"], self.session_index + 1)

    def _finalize_current_session(self) -> dict:
        session_average = self.running_average.copy()
        completed_session_number = self.session_index + 1
        self.session_averages.append(session_average)
        self.session_index += 1
        self.session_epochs = []
        self.running_average = None
        self.flash_count = 0
        return {
            "session_average": session_average,
            "session_number": completed_session_number,
        }

    def save_partial_session(self, min_flashes: Optional[int] = None) -> Optional[dict]:
        required_flashes = self.config["flashes_per_session"]
        threshold = required_flashes // 2 if min_flashes is None else int(min_flashes)
        if self.running_average is None or self.flash_count < threshold:
            return None

        partial_flash_count = self.flash_count
        partial_session = self._finalize_current_session()
        partial_session["flash_count"] = partial_flash_count
        return partial_session

    def process_sample(self, trigger_value: float, eeg_sample: float) -> dict:
        result = {
            "trigger_detected": False,
            "epoch_complete": False,
            "session_complete": False,
            "completed_epoch": None,
            "running_average": self.running_average,
            "completed_session_average": None,
            "flash_count": self.flash_count,
            "session_index": self.session_index,
            "session_number": self._active_session_number(),
            "completed_session_number": None,
        }

        for pending in list(self.pending_epochs):
            pending["samples"].append(float(eeg_sample))
            pending["remaining"] -= 1
            if pending["remaining"] <= 0:
                epoch = np.asarray(pending["samples"][: self.epoch_samples], dtype=float)
                baseline = float(np.mean(epoch[: self.pre_samples])) if self.pre_samples > 0 else float(np.mean(epoch))
                filtered_epoch = self.bandpass_filter.apply_zero_phase(epoch, baseline_mean=baseline)
                self.session_epochs.append(filtered_epoch)
                self.running_average = np.mean(np.vstack(self.session_epochs), axis=0)
                self.flash_count += 1

                result.update(
                    {
                        "epoch_complete": True,
                        "completed_epoch": filtered_epoch,
                        "running_average": self.running_average,
                        "flash_count": self.flash_count,
                    }
                )

                self.pending_epochs.remove(pending)

                if self.flash_count >= self.config["flashes_per_session"]:
                    completed_session = self._finalize_current_session()
                    result.update(
                        {
                            "session_complete": True,
                            "completed_session_average": completed_session["session_average"],
                            "session_index": self.session_index,
                            "completed_session_number": completed_session["session_number"],
                        }
                    )

        rising_edge = float(trigger_value) > 0 and self.prev_trigger <= 0
        if rising_edge:
            pre = list(self.pre_buffer)
            if len(pre) == self.pre_samples:
                self.pending_epochs.append(
                    {
                        "samples": pre + [float(eeg_sample)],
                        "remaining": max(0, self.post_samples - 1),
                    }
                )
                result["trigger_detected"] = True

        self.pre_buffer.append(float(eeg_sample))
        self.prev_trigger = float(trigger_value)
        return result

    def has_completed_all_sessions(self) -> bool:
        return self.session_index >= self.config["num_sessions"]
