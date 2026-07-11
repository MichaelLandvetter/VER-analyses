"""Module for collecting Human-in-the-Loop validation data for Machine Learning."""

import csv
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QComboBox, QPushButton, QMessageBox, QLabel, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from ver_classifier import evaluate_ver_peak
from ver_settings import SettingsManager

class HumanValidationDialog(QDialog):
    def __init__(self, block_data, png_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Machine Learning - Human Validation")
        self.resize(1200, 750) 
        self.block_data = block_data
        self.human_overrides = []

        main_layout = QVBoxLayout(self)

        # --- 1. TOP: The Report Image ---
        if png_path and Path(png_path).exists():
            img_label = QLabel()
            pixmap = QPixmap(str(png_path))
            
            pixmap = pixmap.scaled(1150, 350, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            scroll_area = QScrollArea()
            scroll_area.setWidget(img_label)
            scroll_area.setWidgetResizable(True)
            scroll_area.setMaximumHeight(370) 
            main_layout.addWidget(scroll_area)
        else:
            lbl = QLabel("Report Image not found.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(lbl)

        # --- 2. MIDDLE: Instructions ---
        lbl = QLabel("<b>Human-in-the-Loop Validation</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
                     "1. Review the generated report above. &nbsp;&nbsp;|&nbsp;&nbsp; "
                     "2. Compare it with the computer's predictions below. &nbsp;&nbsp;|&nbsp;&nbsp; "
                     "3. Correct any mistakes. &nbsp;&nbsp;|&nbsp;&nbsp; "
                     "4. Click Save to append to your ML dataset.")
        lbl.setWordWrap(True)
        main_layout.addWidget(lbl)

        # --- 3. BOTTOM: The Validation Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(10) 
        # REORDERED COLUMNS: Computer Label -> Computer Reason -> Human Validation
        self.table.setHorizontalHeaderLabels([
            "Block", "Peak Power", "Scale (Hz)", "P1 Latency", "P2 Latency", 
            "P3 Latency", "SNR-2", "Computer Label", "Computer Reason", "Human Validation"
        ])
        self.table.setRowCount(len(block_data))
        self.combos = []

        for row_idx, data in enumerate(block_data):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(data['block'])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(f"{data['power']:.2e}"))
            self.table.setItem(row_idx, 2, QTableWidgetItem(f"{data['scale']:.1f}"))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{data['p1_lat']:.1f}"))
            self.table.setItem(row_idx, 4, QTableWidgetItem(f"{data['p2_lat']:.1f}"))
            self.table.setItem(row_idx, 5, QTableWidgetItem(f"{data['p3_lat']:.1f}")) 
            self.table.setItem(row_idx, 6, QTableWidgetItem(f"{data['snr']:.2f}"))
            
            comp_lbl = "VER" if data['computer_label'] else "No VER"
            self.table.setItem(row_idx, 7, QTableWidgetItem(comp_lbl))
            
            # MOVED REASON to Column 8
            self.table.setItem(row_idx, 8, QTableWidgetItem(data['reason']))

            # MOVED COMBOBOX to Column 9
            combo = QComboBox()
            combo.addItems(["VER", "No VER"])
            combo.setCurrentText(comp_lbl)
            self.table.setCellWidget(row_idx, 9, combo)
            self.combos.append(combo)

        self.table.resizeColumnsToContents()
        
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents() 
        
        # Stretches the new Combobox column nicely to the edge
        self.table.horizontalHeader().setStretchLastSection(True) 
        main_layout.addWidget(self.table)

        # --- 4. BOTTOM: Save Button ---
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save Validations to CSV and create the Report files")
        save_btn.setStyleSheet("font-weight: bold; background-color: #4472C4; color: white; padding: 10px;")
        save_btn.clicked.connect(self.save_data)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        
        main_layout.addLayout(btn_layout)

    def save_data(self):
        csv_path = Path.cwd() / "training_data.csv"
        file_exists = csv_path.exists()
        self.human_overrides = [] 

        try:
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    # CSV Header remains exactly the same to preserve old data
                    writer.writerow(["Block", "Power", "Scale_Hz", "P1_Latency", "P2_Latency", 
                                     "P3_Latency", "SNR-2", "Computer_Label", "Human_Label", "Reason"])
                
                for i, data in enumerate(self.block_data):
                    is_ver = self.combos[i].currentText() == "VER"
                    self.human_overrides.append(is_ver)
                    
                    human_label = 1 if is_ver else 0
                    comp_label = 1 if data['computer_label'] else 0
                    
                    writer.writerow([
                        data['block'], data['power'], data['scale'],
                        data['p1_lat'], data['p2_lat'], data['p3_lat'], data['snr'],
                        comp_label, human_label, data['reason']
                    ])
            
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save training data: {e}")


def launch_ml_logger(session_wavelets, session_wavelet_freqs, epoch_time_ms, session_ver_peaks, labels, png_path=None, parent=None):
    """Helper function to extract features and launch the UI."""
    manager = SettingsManager()
    cfg = manager.load_settings().get("CLASSIFIER_CONFIG", {})
    snr_threshold = cfg.get("snr_threshold", 2.0)

    block_data = []
    
    for idx, wavelet in enumerate(session_wavelets):
        peak_idx = np.unravel_index(np.argmax(wavelet), wavelet.shape)
        peak_scale = float(session_wavelet_freqs[peak_idx[0]])
        peak_power = float(wavelet[peak_idx])

        ver_peaks = session_ver_peaks[idx] if session_ver_peaks and idx < len(session_ver_peaks) else None
        
        p1_lat = 0.0; p2_lat = 0.0; p3_lat = 0.0; p2_snr = 0.0
        if ver_peaks is not None:
            p1_d, p2_d, p3_d = ver_peaks.get('Peak-1',{}), ver_peaks.get('Peak-2',{}), ver_peaks.get('Peak-3',{})
            p1_lat = float(p1_d['latency_ms']) if p1_d.get('found') else 0.0
            p2_lat = float(p2_d['latency_ms']) if p2_d.get('found') else 0.0
            p3_lat = float(p3_d['latency_ms']) if p3_d.get('found') else 0.0
            p2_snr = float(p2_d['snr']) if p2_d.get('found') else 0.0
            
        # Look for this line inside launch_ml_logger:
        is_ver, failure_details = evaluate_ver_peak(peak_scale, peak_power, p1_lat if p1_lat else None, p2_lat if p2_lat else None, p3_lat if p3_lat else None, p2_snr)
        
        # --- NEW REASON LOGIC ---
        if is_ver:
            reason = "Passed"
        else:
            failed_tests = [test_name for test_name, passed in failure_details.items() if not passed]
            reason = "Failed:\n" + "\n".join(failed_tests)
        # ------------------------

        block_data.append({
            'block': labels[idx] if idx < len(labels) else f"Block {idx+1}",
            'power': peak_power, 'scale': peak_scale,
            'p1_lat': p1_lat, 'p2_lat': p2_lat, 'p3_lat': p3_lat, 'snr': p2_snr,
            'computer_label': is_ver, 'reason': reason
        })

    dialog = HumanValidationDialog(block_data, png_path, parent)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.human_overrides
    return None