"""
Enhanced Data Loader Agent - Fully Robust & Orchestrator-Compatible
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union
from base_agent import BaseAgent


class DataLoaderAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None

    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    # ------------------------------------------------------------
    # ✅ DYNAMIC CAPABILITY DISCOVERY
    # ------------------------------------------------------------
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "load_data",
                "description": "Load dataset from CSV or Excel file",
                "parameters": ["file", "file_type"],
                "examples": [
                    "load the dataset",
                    "read the data file",
                    "import the CSV"
                ]
            },
            {
                "function_name": "get_preview",
                "description": "Show first N rows of dataset",
                "parameters": ["n"],
                "examples": [
                    "show me the first 5 rows",
                    "display first 10 rows",
                    "preview the dataset"
                ]
            },
            {
                "function_name": "get_shape",
                "description": "Get number of rows and columns",
                "parameters": [],
                "examples": [
                    "how many rows and columns",
                    "dataset size",
                    "show dimensions"
                ]
            },
            {
                "function_name": "get_columns",
                "description": "Get list of column names",
                "parameters": [],
                "examples": [
                    "list all columns",
                    "what are the column names",
                    "show data fields"
                ]
            }
        ]

    # ------------------------------------------------------------
    # ✅ SAFE DATA LOADING
    # ------------------------------------------------------------
    def load_data(self, file, file_type: str = "csv") -> Union[pd.DataFrame, str]:
        """Load dataset safely from file path or Streamlit-like uploader."""
        try:
            if file is None:
                return "No file provided."

            # Auto-detect file type if needed
            if file_type == "auto":
                name = getattr(file, "name", "")
                if name.endswith(".csv"):
                    file_type = "csv"
                elif name.endswith(".xls") or name.endswith(".xlsx"):
                    file_type = "excel"
                else:
                    return f"Unsupported file type for: {name}"

            if file_type.lower() in ["csv"]:
                self.data = pd.read_csv(file)

            elif file_type.lower() in ["xls", "xlsx", "excel"]:
                self.data = pd.read_excel(file)

            else:
                return f"Unsupported file type '{file_type}'. Use csv or excel."

            return self.data

        except Exception as e:
            return f"Error loading data: {e}"

    # ------------------------------------------------------------
    # ✅ PREVIEW
    # ------------------------------------------------------------
    def get_preview(self, n: int = 5) -> Union[pd.DataFrame, str]:
        if not self.has_data():
            return "No dataset loaded yet. Load a dataset first."
        return self.data.head(max(1, int(n)))

    # ------------------------------------------------------------
    # ✅ SHAPE
    # ------------------------------------------------------------
    def get_shape(self) -> Dict[str, int]:
        if not self.has_data():
            return {"rows": 0, "columns": 0}
        r, c = self.data.shape
        return {"rows": int(r), "columns": int(c)}

    # ------------------------------------------------------------
    # ✅ COLUMN LIST
    # ------------------------------------------------------------
    def get_columns(self) -> Union[List[str], str]:
        if not self.has_data():
            return "No dataset loaded yet."
        return list(self.data.columns)

    # ------------------------------------------------------------
    # ✅ SAFE EXECUTION DISPATCHER
    # ------------------------------------------------------------
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        """Required by orchestrator: safely call functions."""
        if not hasattr(self, function_name):
            return f"Method '{function_name}' not found in DataLoaderAgent."

        try:
            method = getattr(self, function_name)
            if callable(method):
                return method(**kwargs)
            return method
        except Exception as e:
            return f"❌ Error executing {function_name}: {e}"
