from pathlib import Path
import logging.config
import json


root_directory = Path(__file__).resolve().parents[3]
logging_configuration_file = root_directory / "config.json"
log_dir = root_directory / "logs"
log_dir.mkdir(parents=True, exist_ok=True)


def configure_logging():
  
  try:
    with open(file=logging_configuration_file) as f:
      log_config = json.load(fp=f)
      
    logging.config.dictConfig(config=log_config)
    
  except FileNotFoundError:
    raise FileNotFoundError(f"Logging configuration file not found: {logging_configuration_file}")
  
  except PermissionError:
    raise PermissionError(f"Permission denied to access logging configuration file: {logging_configuration_file}")
  
  except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON in logging configuration file: {logging_configuration_file}")
  
  except Exception as e:
    raise ValueError(f"Error while loading logging configuration file: {e}")