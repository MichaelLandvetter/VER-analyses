"""Tests for the raw USB live-session export format.

Validates that:
- ``SerialAcquisitionSource`` writes a ``#VER_LIVE_USB`` marker as the first
  line of the raw log and stores the *normalised* trigger level (not the
  hysteresis-decoded level which can be constant 1.0).
- ``auto_detect_file_format`` in ver_main returns ``"Live-USB"`` for files
  that start with the marker.
- ``FILE_FORMATS["Live-USB"]`` contains the correct column and threshold
  settings so that ``FileAcquisitionSimulator`` can replay the raw log and
  produce proper rising-edge events (epochs).
"""

from __future__ import annotations

import io
import struct
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_usb_packet(trigger_state: int, eeg: float) -> bytes:
    """Build a valid binary packet matching SerialAcquisitionSource protocol."""
    header = b"\xA5\x5A"
    footer = b"\x01"
    return header + struct.pack("<Hf", trigger_state, eeg) + footer


# ---------------------------------------------------------------------------
# ver_config – Live-USB format present and sensible
# ---------------------------------------------------------------------------

def test_live_usb_format_in_file_formats():
    from ver_config import FILE_FORMATS
    assert "Live-USB" in FILE_FORMATS, "'Live-USB' must be a named format in FILE_FORMATS"


def test_live_usb_format_columns():
    from ver_config import FILE_FORMATS
    fmt = FILE_FORMATS["Live-USB"]
    assert fmt["trigger_column"] == 0
    assert fmt["eeg_column"] == 1
    assert fmt["skip_header"] == 1


def test_live_usb_format_threshold_detects_normalised_level():
    """threshold=0.5 correctly splits 0.0 (off) from 1.0 (on)."""
    from ver_config import FILE_FORMATS
    fmt = FILE_FORMATS["Live-USB"]
    threshold = fmt["trigger_threshold"]
    assert threshold == 0.5, f"Expected trigger_threshold=0.5; got {threshold}"
    assert 0.0 < threshold < 1.0, "threshold must lie strictly between 0 and 1"


# ---------------------------------------------------------------------------
# auto_detect_file_format – Live-USB marker detection
# ---------------------------------------------------------------------------

def _compile_auto_detect():
    """Extract and compile ``auto_detect_file_format`` from ver_main.py without
    importing the full module (which would require a running Qt display)."""
    import ast as _ast
    import logging
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "auto_detect_file_format":
            module = _ast.Module(body=[node], type_ignores=[])
            _ast.fix_missing_locations(module)
            ns = {"log": logging.getLogger("test_auto_detect")}
            exec(compile(module, filename="ver_main.py", mode="exec"), ns)
            return ns["auto_detect_file_format"]
    raise AssertionError("auto_detect_file_format not found in ver_main.py")


def test_auto_detect_returns_live_usb_for_marker_file(tmp_path):
    """A file whose first line is '#VER_LIVE_USB' must be detected as 'Live-USB'."""
    fn = _compile_auto_detect()
    raw = tmp_path / "Test_1_raw.txt"
    raw.write_text("#VER_LIVE_USB\n0.0\t1.5\n1.0\t2.3\n0.0\t1.8\n", encoding="utf-8")
    assert fn(str(raw)) == "Live-USB"


def test_auto_detect_still_returns_labchart_for_plain_2col_file(tmp_path):
    """Existing 2-column LabChart files (no marker) must still be detected as 'LabChart'."""
    fn = _compile_auto_detect()
    raw = tmp_path / "labchart.txt"
    raw.write_text("0.0\t1.5\n1.0\t2.3\n0.0\t1.8\n", encoding="utf-8")
    assert fn(str(raw)) == "LabChart"


# ---------------------------------------------------------------------------
# SerialAcquisitionSource – raw log format
# ---------------------------------------------------------------------------

def _run_source_with_packets(packets: list[bytes], tmp_path: Path):
    """Feed binary packets into SerialAcquisitionSource via a mock serial port
    and return (samples_yielded, raw_log_lines)."""
    from ver_acquisition import SerialAcquisitionSource

    source = SerialAcquisitionSource(port="MOCK", baud_rate=921600)

    # Inject a fake serial object and pre-fill the buffer
    combined = b"".join(packets)

    class _FakeSerial:
        def __init__(self, data: bytes):
            self._buf = bytearray(data)
            self.in_waiting = len(self._buf)

        def read(self, n: int) -> bytes:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
            self.in_waiting = len(self._buf)
            return chunk

        def close(self):
            pass

    source._serial = _FakeSerial(combined)

    # Redirect the log file to a StringIO so we can inspect it
    log_buf = io.StringIO()
    source._raw_log_file = log_buf
    log_buf.write("#VER_LIVE_USB\n")  # mimic _open() header

    samples = []
    for sample in source.stream_samples():
        samples.append(sample.copy())
        # Stop after consuming all pre-loaded data
        if source._serial.in_waiting == 0 and not source._buffer:
            break

    lines = [ln for ln in log_buf.getvalue().splitlines() if ln and not ln.startswith("#")]
    return samples, lines


def test_serial_source_writes_ver_live_usb_header(tmp_path):
    """_open() must write '#VER_LIVE_USB' as the first line of the raw log."""
    from unittest.mock import patch, MagicMock

    import serial as _serial_mod  # noqa: F401 (needed so import inside _open works)
    from ver_acquisition import SerialAcquisitionSource

    source = SerialAcquisitionSource(port="MOCK")

    buf = io.StringIO()

    mock_serial = MagicMock()

    with patch("serial.Serial", return_value=mock_serial), \
         patch("builtins.open", return_value=buf):
        # Patch Path so filename is predictable
        with patch("ver_acquisition.Path", side_effect=lambda x: Path(tmp_path / x)):
            try:
                source._open()
            except Exception:
                pass  # we only care about what was written before any error

    written = buf.getvalue()
    assert written.startswith("#VER_LIVE_USB\n"), (
        f"Expected '#VER_LIVE_USB' as first line; got: {written!r}"
    )


def test_serial_source_normalised_trigger_written_not_hysteresis(tmp_path):
    """When the MCU sends trigger_state=HIGH for every packet the raw log
    must NOT be constant 1.0.  The pre-hysteresis normalised level is written
    instead, so the initial sample (floor=0, ceil=1) at trigger_state=0
    writes 0.0 and at trigger_state=65535 writes 1.0."""
    from ver_acquisition import SerialAcquisitionSource

    source = SerialAcquisitionSource(port="MOCK")
    # Feed _decode_serial_trigger with alternating low/high states
    source._decode_serial_trigger(0)
    assert source._last_normalized_trigger == 0.0

    source._decode_serial_trigger(65535)
    assert source._last_normalized_trigger == 1.0

    source._decode_serial_trigger(0)
    assert source._last_normalized_trigger == 0.0


# ---------------------------------------------------------------------------
# End-to-end: raw log → FileAcquisitionSimulator → rising-edge epochs
# ---------------------------------------------------------------------------

def test_live_usb_raw_log_reloaded_yields_rising_edges(tmp_path):
    """A synthetic Live-USB raw file with alternating trigger values must
    produce multiple rising-edge detections when replayed by
    FileAcquisitionSimulator using the Live-USB format."""
    from ver_acquisition import FileAcquisitionSimulator
    from ver_config import FILE_FORMATS

    # Build a raw log: 5 ON pulses of 3 samples each, separated by 5 OFF samples
    lines = ["#VER_LIVE_USB\n"]
    eeg_val = 1.0
    for _ in range(5):
        for _ in range(5):   # OFF samples
            lines.append(f"0.0\t{eeg_val:.1f}\n")
            eeg_val += 0.1
        for _ in range(3):   # ON samples
            lines.append(f"1.0\t{eeg_val:.1f}\n")
            eeg_val += 0.1

    raw = tmp_path / "Test_live.txt"
    raw.write_text("".join(lines), encoding="utf-8")

    sim = FileAcquisitionSimulator(
        str(raw),
        sample_rate=250,
        speed_factor=None,
        file_config=dict(FILE_FORMATS["Live-USB"]),
    )

    rising_edges = 0
    prev = False
    for sample in sim.stream_samples():
        current = bool(sample[0] > FILE_FORMATS["Live-USB"]["trigger_threshold"])
        if current and not prev:
            rising_edges += 1
        prev = current

    assert rising_edges == 5, (
        f"Expected 5 rising edges from 5 ON pulses; got {rising_edges}"
    )
