import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_project_root():
    # src/utils/config.py -> src/utils -> src -> crypto_advanced
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def load_config(config_path=None):
    """
    Loads the YAML configuration file.
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs/settings.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_model_config(config_path=None):
    """
    Loads the Model YAML configuration file.
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs/model.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
