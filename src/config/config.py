import yaml
from typing import Dict, Any

class ConfigLoader:
    """
    Responsible for loading and managing configuration files
    """

    def __init__(self, base_config_path: str = "config/base.yaml"):
        self.base_config = self._load_yaml(base_config_path)
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get merged configuration (base + enviroment specific)
        """
        return self.base_config
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        """
        for key, value in updates.items():
            if isinstance(value, dict):
                self.base_config[key].update(value)
            else:
                self.base_config[key] = value
