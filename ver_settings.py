import json
from pathlib import Path
import ver_config

SETTINGS_FILE = "user_settings.json"

class SettingsManager:
    def __init__(self, filepath=SETTINGS_FILE):
        self.filepath = Path(filepath)
        self.settings = self.load_settings()
        self.apply_to_live_config() # Apply instantly on startup

    def get_default_settings(self):
        """Grabs the hardcoded lab defaults from ver_config.py"""
        return {
            "EPOCH_CONFIG": dict(ver_config.EPOCH_CONFIG),
            "WAVELET_CONFIG": dict(ver_config.WAVELET_CONFIG),
            "FILTER_CONFIG": dict(ver_config.FILTER_CONFIG)
        }

    def load_settings(self):
        """Loads from JSON, or creates it if it doesn't exist."""
        if not self.filepath.exists():
            default_settings = self.get_default_settings()
            self.save_settings(default_settings)
            return default_settings

        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {self.filepath}: {e}. Using defaults.")
            return self.get_default_settings()

    def save_settings(self, new_settings):
        """Writes the dictionary to the JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(new_settings, f, indent=4)
        self.settings = new_settings
        self.apply_to_live_config()

    def apply_to_live_config(self):
        """Injects the loaded settings directly into the live memory variables."""
        if "EPOCH_CONFIG" in self.settings:
            ver_config.EPOCH_CONFIG.update(self.settings["EPOCH_CONFIG"])
        if "WAVELET_CONFIG" in self.settings:
            ver_config.WAVELET_CONFIG.update(self.settings["WAVELET_CONFIG"])
        if "FILTER_CONFIG" in self.settings:
            ver_config.FILTER_CONFIG.update(self.settings["FILTER_CONFIG"])