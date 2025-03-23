# utils/data_loader.py
import json
import logging

logger = logging.getLogger(__name__)

def load_data_from_file(file_path):
    """Load job applications from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def load_data_from_json_string(json_string):
    """Parse JSON string into job applications data."""
    try:
        data = json.loads(json_string)
        # Convert single object to list if needed
        if not isinstance(data, list):
            data = [data]
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return None