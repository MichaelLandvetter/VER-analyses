"""Data acquisition module for file replay, live Waveshare streaming, and USB serial microcontrollers."""

from __future__ import annotations

import time
import importlib
import struct
import sys
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ver_config import ACQ_CONFIG, FILE_CONFIG, HARDWARE_CONFIG, SERIAL_CONFIG


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


class WaveshareAcquisitionSource:
    """Read live EEG/trigger data from a Waveshare ADS1256 (channels 0/1)."""

    def __init__(
        self,
        sample_rate: Optional[float] = None,
        waveshare_dir: Optional[str] = None,
        eeg_channel: Optional[int] = None,
        trigger_channel: Optional[int] = None,
        trigger_threshold: Optional[float] = None,
        trigger_high_threshold: Optional[float] = None,
        trigger_low_threshold: Optional[float] = None,
        trigger_min_interval_ms: Optional[float] = None,
        adc_gain: Optional[str] = None,
        adc_rate: Optional[str] = None,
        voltage_ref: Optional[float] = None,
    ):
        self.sample_rate = float(sample_rate if sample_rate is not None else ACQ_CONFIG["sample_rate"])
        self.waveshare_dir = Path(waveshare_dir if waveshare_dir is not None else HARDWARE_CONFIG["waveshare_dir"])
        self.eeg_channel = int(eeg_channel if eeg_channel is not None else HARDWARE_CONFIG["eeg_channel"])
        self.trigger_channel = int(trigger_channel if trigger_channel is not None else HARDWARE_CONFIG["trigger_channel"])
        self.trigger_threshold = float(
            trigger_threshold if trigger_threshold is not None else HARDWARE_CONFIG["trigger_threshold"]
        )
        self.trigger_high_threshold = float(
            trigger_high_threshold
            if trigger_high_threshold is not None
            else HARDWARE_CONFIG.get("trigger_high_threshold", self.trigger_threshold)
        )
        self.trigger_low_threshold = float(
            trigger_low_threshold
            if trigger_low_threshold is not None
            else HARDWARE_CONFIG.get("trigger_low_threshold", self.trigger_threshold * 0.5)
        )
        self.trigger_min_interval_s = max(
            0.0,
            float(
                trigger_min_interval_ms
                if trigger_min_interval_ms is not None
                else HARDWARE_CONFIG.get("trigger_min_interval_ms", 0.0)
            )
            / 1000.0,
        )
        self.adc_gain = str(adc_gain if adc_gain is not None else HARDWARE_CONFIG["adc_gain"])
        self.adc_rate = str(adc_rate if adc_rate is not None else HARDWARE_CONFIG["adc_rate"])
        self.voltage_ref = float(voltage_ref if voltage_ref is not None else HARDWARE_CONFIG["voltage_ref"])
        self._adc = None
        self._waveshare_config = None
        self._trigger_high = False
        self._last_trigger_time = None

    def _load_waveshare_modules(self) -> tuple[object, object]:
        waveshare_path = str(self.waveshare_dir.resolve())
        if waveshare_path not in sys.path:
            sys.path.insert(0, waveshare_path)
        try:
            ads1256_module = importlib.import_module("ADS1256")
            config_module = importlib.import_module("config")
        except Exception as exc:
            raise RuntimeError(
                "Waveshare ADS1256 modules could not be imported from "
                f"{waveshare_path}. Ensure ADS1256.py and config.py exist and required "
                "dependencies (spidev, lgpio) are installed."
            ) from exc
        return ads1256_module, config_module

    def _open(self) -> None:
        if self._adc is not None:
            return
        ads1256_module, config_module = self._load_waveshare_modules()
        self._waveshare_config = config_module
        self._adc = ads1256_module.ADS1256()
        if self._adc.ADS1256_init() != 0:
            raise RuntimeError("Failed to initialize Waveshare ADS1256")

        if self.adc_gain not in ads1256_module.ADS1256_GAIN_E:
            raise ValueError(
                f"Unsupported ADS1256 gain: {self.adc_gain}. "
                f"Supported values: {sorted(ads1256_module.ADS1256_GAIN_E.keys())}"
            )
        if self.adc_rate not in ads1256_module.ADS1256_DRATE_E:
            raise ValueError(
                f"Unsupported ADS1256 data rate: {self.adc_rate}. "
                f"Supported values: {sorted(ads1256_module.ADS1256_DRATE_E.keys())}"
            )
        if self.trigger_low_threshold >= self.trigger_high_threshold:
            raise ValueError(
                "Invalid trigger thresholds: trigger_low_threshold must be less than trigger_high_threshold."
            )

        gain = ads1256_module.ADS1256_GAIN_E[self.adc_gain]
        drate = ads1256_module.ADS1256_DRATE_E[self.adc_rate]
        self._adc.ADS1256_ConfigADC(gain, drate)

    def close(self) -> None:
        if self._waveshare_config is not None:
            try:
                self._waveshare_config.module_exit()
            except Exception:
                pass
        self._adc = None
        self._waveshare_config = None

    def _raw_to_voltage(self, raw_value: int) -> float:
        raw = int(raw_value) & 0xFFFFFF
        if raw & 0x800000:
            raw -= 1 << 24
        return float(raw) * self.voltage_ref / 0x7FFFFF

    def stream_samples(self) -> Generator[np.ndarray, None, None]:
        self._open()
        sample_interval = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        next_sample_time = time.perf_counter()
        try:
            while True:
                # Method name is from vendor API ("Channal").
                eeg_raw = self._adc.ADS1256_GetChannalValue(self.eeg_channel)
                trigger_raw = self._adc.ADS1256_GetChannalValue(self.trigger_channel)
                eeg = self._raw_to_voltage(eeg_raw)
                trigger_value = self._raw_to_voltage(trigger_raw)
                prev_trigger_high = self._trigger_high
                if self._trigger_high:
                    if trigger_value <= self.trigger_low_threshold:
                        self._trigger_high = False
                elif trigger_value >= self.trigger_high_threshold:
                    self._trigger_high = True

                rising_edge = self._trigger_high and not prev_trigger_high
                trigger = 0.0
                if rising_edge:
                    now = time.perf_counter()
                    if (
                        self._last_trigger_time is None
                        or self.trigger_min_interval_s <= 0
                        or now - self._last_trigger_time >= self.trigger_min_interval_s
                    ):
                        trigger = 1.0
                        self._last_trigger_time = now
                yield np.asarray([trigger, eeg], dtype=float)
                if sample_interval > 0:
                    next_sample_time += sample_interval
                    sleep_for = next_sample_time - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    elif sleep_for < -sample_interval:
                        next_sample_time = time.perf_counter()
        finally:
            self.close()


class SerialAcquisitionSource:
    """Read live EEG/trigger data from a microcontroller over USB serial.

    Supports two serial protocols:

    1) ASCII CSV (legacy)::

        <trigger>,<eeg_volts>\\n

       ``trigger`` is an integer: ``1`` on the sample where a flash/stimulus
       occurred, ``0`` otherwise. ``eeg_volts`` is a floating-point voltage.

    2) Framed binary packet (little-endian)::

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
        return np.asarray([1.0 if trigger_state else 0.0, float(eeg)], dtype=float)

    def _try_parse_ascii_sample(self) -> Optional[np.ndarray]:
        newline_index = self._buffer.find(b"\n")
        if newline_index < 0:
            return None
        line = bytes(self._buffer[:newline_index])
        del self._buffer[: newline_index + 1]
        if not line:
            return None
        try:
            text = line.decode("ascii", errors="ignore").strip()
            parts = text.split(",")
            if len(parts) < 2:
                return None
            trigger = float(parts[0])
            eeg = float(parts[1])
            return np.asarray([1.0 if trigger else 0.0, eeg], dtype=float)
        except (ValueError, IndexError):
            return None

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

                    if self._binary_header not in self._buffer:
                        sample = self._try_parse_ascii_sample()
                        if sample is not None:
                            yield sample
                            continue
                    break
        finally:
            self.close()
