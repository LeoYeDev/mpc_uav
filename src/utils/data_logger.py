import numpy as np
import pickle
import os
from collections import defaultdict
from typing import Dict, Any, Optional

class DataLogger:
    """
    A simple logger to record experiment data (scalars or arrays) over time steps.
    """
    def __init__(self):
        self.data: Dict[str, list] = defaultdict(list)

    def log(self, key: str, value: Any) -> None:
        """
        Log a value for a specific key.
        
        Args:
            key (str): The name of the data field (e.g., "rmse", "position").
            value (Any): The value to append.
        """
        self.data[key].append(value)

    def get(self, key: str) -> list:
        """
        Get the list of logged values for a key.
        """
        return self.data.get(key, [])

    def reset(self) -> None:
        """
        Clear all logged data.
        """
        self.data.clear()

    def save_pickle(self, filename: str) -> None:
        """
        Save the entire data dictionary to a pickle file.
        
        Args:
            filename (str): The path to the file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Convert lists to numpy arrays where applicable (optimization)
        export_data = {}
        for k, v in self.data.items():
            try:
                export_data[k] = np.array(v)
            except Exception:
                export_data[k] = v
                
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
        print(f"Data saved to {filename}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the data as a dictionary, converting lists to numpy arrays.
        """
        export_data = {}
        for k, v in self.data.items():
            try:
                export_data[k] = np.array(v)
            except Exception:
                export_data[k] = v
        return export_data
