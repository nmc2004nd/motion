import yaml
import os

class ConfigParser:
    def __init__(self, config_path="config/pipeline_config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get(self, key_path, default=None):
        """
        Retrieve a configuration value using dot notation (e.g. 'tracking.pyrlk.win_size')
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key):
        return self.config[key]

# Provide a global instance that can be initialized once and accessed everywhere if needed, 
# or individual modules can instantiate it.
def load_config(config_path="config/pipeline_config.yaml"):
    return ConfigParser(config_path)
