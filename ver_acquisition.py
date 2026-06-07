"""Data acquisition module for file replay and USB serial microcontrollers."""

from __future__ import annotations

import time
import struct
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ver_config import ACQ_CONFIG, FILE_CONFIG, SERIAL_CONFIG


class FileAcquisitionSimulator:
    """Replay a raw text file sample-by-sample."""

    def __init__(self, file_path: str, sample_rate: Optional[float] = None, speed_factor: Optional[float] = 1.0):
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate if sample_rate is not None else ACQ_CONFIG["sample_rate"]
        # speed_factor=1.0 → real-time, 10.0 → 10× faster, None → maximum speed (no sleep)
        self.speed_factor = speed_factor

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

        base_sleep = 1.0 / float(self.sample_rate)
        trigger_column = int(FILE_CONFIG["trigger_column"])
        eeg_column = int(FILE_CONFIG["eeg_column"])
        trigger_mode = str(FILE_CONFIG.get("trigger_mode", "threshold"))
        trigger_threshold = float(FILE_CONFIG.get("trigger_threshold", 0.5))

        for row in data:
            trigger_value = float(row[trigger_column])
            if trigger_mode == "threshold":
                trigger = trigger_value > trigger_threshold
            elif trigger_mode == "interval":
                trigger = trigger_value > trigger_threshold
            else:
                raise ValueError(f"Unsupported trigger mode: {trigger_mode}")

            eeg = float(row[eeg_column])
            yield np.asarray([1.0 if trigger else 0.0, eeg], dtype=float)
            if self.speed_factor is not None and self.speed_factor > 0:
                time.sleep(base_sleep / self.speed_factor)


class SerialAcquisitionSource:
    """Read live EEG/trigger data from a microcontroller over USB serial.

    Expects framed binary packets (little-endian)::

        [0xA5, 0x5A][trigger:uint16][eeg:float32][0x01]

    This packet is 9 bytes total and keeps trigger/EEG aligned per sample.

    Malformed data is skipped so occasional transmission errors do not crash
    the acquisition loop.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        sample_rate: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self.port = str(port if port is not None else SERIAL_CONFIG["port"])
        self.baud_rate = int(baud_rate if baud_rate is not None else SERIAL_CONFIG["baud_rate"])
        self.sample_rate = float(sample_rate if sample_rate is not None else ACQ_CONFIG["sample_rate"])
        self.timeout = float(timeout if timeout is not None else SERIAL_CONFIG.get("timeout", 2.0))
        self._serial = None
        self._buffer = bytearray()
        self._binary_header = b"\xA5\x5A"
        self._binary_footer = 0x01
        self._binary_packet_size = 9
        self._serial_trigger_high = False
        self._serial_trigger_floor = 0.0
        self._serial_trigger_ceil = 1.0
        self._serial_trigger_high_threshold = float(SERIAL_CONFIG.get("trigger_high_threshold", 0.7))
        self._serial_trigger_low_threshold = float(SERIAL_CONFIG.get("trigger_low_threshold", 0.3))

    def _decode_serial_trigger(self, trigger_state: int) -> float:
        raw_level = max(0.0, float(trigger_state))
        self._serial_trigger_floor = min(self._serial_trigger_floor, raw_level)
        self._serial_trigger_ceil = max(self._serial_trigger_ceil, raw_level)
        span = self._serial_trigger_ceil - self._serial_trigger_floor
        if span > 0:
            normalized_level = (raw_level - self._serial_trigger_floor) / span
        else:
            normalized_level = 0.0

        if self._serial_trigger_high:
            if normalized_level <= self._serial_trigger_low_threshold:
                self._serial_trigger_high = False
        elif normalized_level >= self._serial_trigger_high_threshold:
            self._serial_trigger_high = True

        return 1.0 if self._serial_trigger_high else 0.0

    def _open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial"
            ) from exc
        self._serial = serial.Serial(self.port, baudrate=self.baud_rate, timeout=self.timeout)

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _try_parse_binary_sample(self) -> Optional[np.ndarray]:
        header_index = self._buffer.find(self._binary_header)
        if header_index < 0:
            if len(self._buffer) > 1:
                del self._buffer[:-1]
            return None
        if header_index > 0:
            del self._buffer[:header_index]
        if len(self._buffer) < self._binary_packet_size:
            return None

        packet = bytes(self._buffer[: self._binary_packet_size])
        if packet[-1] != self._binary_footer:
            del self._buffer[0]
            return None

        try:
            _, trigger_state, eeg, _ = struct.unpack("<2sHf1s", packet)
        except struct.error:
            del self._buffer[0]
            return None

        del self._buffer[: self._binary_packet_size]
        trigger_level = self._decode_serial_trigger(trigger_state)
        return np.asarray([trigger_level, float(eeg)], dtype=float)

    def stream_samples(self) -> Generator[np.ndarray, None, None]:
        """Yield ``[trigger, eeg_volts]`` arrays read from the serial port.

        Runs until the acquisition is stopped externally (the
        :class:`AcquisitionWorker` sets ``_running = False`` and the
        generator is garbage-collected, which closes the port via
        ``finally``).
        """
        self._open()
        try:
            while True:
                chunk = self._serial.read(self._serial.in_waiting or 1)
                if not chunk:
                    continue
                self._buffer.extend(chunk)
                if len(self._buffer) > 8192:
                    del self._buffer[:-1024]

                while True:
                    sample = self._try_parse_binary_sample()
                    if sample is not None:
                        yield sample
                        continue
                    break
        finally:
            self.close()
