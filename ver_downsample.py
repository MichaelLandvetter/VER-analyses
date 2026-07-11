"""Module for downsampling LabChart data files."""

import numpy as np
from pathlib import Path
from scipy import signal  # Assuming you used scipy for the anti-alias filter!

def downsample_labchart_file(input_filepath: str) -> str:
    """
    Downsample a LabChart .txt file from 1000 Hz to 250 Hz using anti-alias decimation.
    Saves the result as <original_name>_250_Hz.txt in the same directory.
    Returns the output file path.
    """
    from scipy.signal import decimate

    input_path = Path(input_filepath)
    output_path = input_path.parent / f"{input_path.stem}_250_Hz{input_path.suffix}"
    source_rate_hz = 1000
    target_rate_hz = 250
    decimation_factor = source_rate_hz // target_rate_hz

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_rows = []
    header_lines = []
    parsing_started = False
    for line in lines:
        parts = line.strip().split("\t")
        try:
            numeric_parts = [float(p) for p in parts if p.strip()]
            data_rows.append(numeric_parts)
            parsing_started = True
        except ValueError:
            if not parsing_started:
                header_lines.append(line)

    if not data_rows:
        raise ValueError("No numeric data rows found in file.")

    array = np.array(data_rows)

    decimated_cols = []
    for col_idx in range(array.shape[1]):
        col = array[:, col_idx]
        try:
            # Primary attempt: FIR filter (highest quality, linear phase)
            dec = decimate(col, q=decimation_factor, ftype="fir", zero_phase=True)
        except Exception:
            try:
                # Secondary attempt: IIR filter (handles shorter/awkward arrays better)
                dec = decimate(col, q=decimation_factor, ftype="iir", zero_phase=True)
            except Exception as e:
                # If both fail, STOP. Do not slice the array, as it will cause severe aliasing.
                raise RuntimeError(
                    f"Anti-alias decimation completely failed on column {col_idx + 1}. "
                    f"Data might be too short or corrupted. Error: {e}"
                )
        decimated_cols.append(dec)

    decimated = np.column_stack(decimated_cols)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
        for row in decimated:
            f.write("\t".join(f"{v:.6g}" for v in row) + "\n")

    return str(output_path)

