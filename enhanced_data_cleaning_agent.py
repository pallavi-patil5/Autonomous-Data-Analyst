"""
Enhanced Data Cleaning Agent - Fully Robust & Orchestrator-Compatible
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from base_agent import BaseAgent


class DataCleaningAgent(BaseAgent):

    def __init__(self, data: Optional[pd.DataFrame]):
        super().__init__()
        self.data: pd.DataFrame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    # ------------------------------------------------------------------
    # ✅ UTILITY HELPERS
    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    def set_data(self, df: pd.DataFrame):
        self.data = df.copy()

    def get_preview(self, n: int = 10):
        if not self.has_data():
            return "No dataset loaded yet. Cannot preview data."
        return self.data.head(max(1, int(n)))

    def get_shape(self):
        if not self.has_data():
            return {"rows": 0, "columns": 0}
        r, c = self.data.shape
        return {"rows": int(r), "columns": int(c)}

    def get_row_count(self):
        if not self.has_data():
            return 0
        return int(self.data.shape[0])

    # ------------------------------------------------------------------
    # ✅ CAPABILITY DISCOVERY
    # ------------------------------------------------------------------
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "handle_missing",
                "description": "Handle missing values using mean, median, mode, drop, forward-fill, backward-fill.",
                "parameters": ["strategy"],
                "examples": ["fill missing values with mean", "handle missing data"],
            },
            {
                "function_name": "handle_outliers",
                "description": "Remove or handle outliers using z-score or IQR",
                "parameters": ["method", "threshold"],
                "examples": ["remove outliers", "clean outliers using IQR"],
            },
            {
                "function_name": "standardize_columns",
                "description": "Standardize column names to lowercase with underscores",
                "parameters": [],
                "examples": ["clean column names", "standardize column names"],
            },
            {
                "function_name": "remove_duplicates",
                "description": "Remove duplicate rows from dataset",
                "parameters": [],
                "examples": ["drop duplicates", "remove duplicate rows"],
            },
            {
                "function_name": "convert_types",
                "description": "Convert column data types (auto or target type)",
                "parameters": ["column", "target_type"],
                "examples": ["convert age to integer", "fix data types"],
            },
            # Universal utility functions (for your orchestrator)
            {
                "function_name": "get_preview",
                "description": "Show first N rows",
                "parameters": ["n"],
                "examples": ["preview dataset"],
            },
            {
                "function_name": "get_shape",
                "description": "Dataset shape",
                "parameters": [],
                "examples": ["how many rows and columns"],
            },
            {
                "function_name": "get_row_count",
                "description": "Dataset row count",
                "parameters": [],
                "examples": ["how many rows"],
            },
        ]

    # ------------------------------------------------------------------
    # ✅ MISSING VALUE HANDLING
    # ------------------------------------------------------------------
    def handle_missing(self, strategy: str = "mean"):
        if not self.has_data():
            return "No dataset loaded. Cannot handle missing values."

        strategy = strategy.lower()

        strategies = ["mean", "median", "mode", "drop", "ffill", "bfill"]
        if strategy not in strategies:
            return f"Invalid strategy '{strategy}'. Use: {strategies}"

        df = self.data.copy()

        try:
            for col in df.columns:
                missing = df[col].isna().sum()
                if missing == 0:
                    continue

                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())

                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())

                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

                elif strategy == "drop":
                    df = df.dropna()
                    break

                elif strategy == "ffill":
                    df[col] = df[col].ffill()

                elif strategy == "bfill":
                    df[col] = df[col].bfill()

            self.data = df
            return df

        except Exception as e:
            return f"Error handling missing values: {e}"

    # ------------------------------------------------------------------
    # ✅ OUTLIER HANDLING
    # ------------------------------------------------------------------
    def handle_outliers(self, method: str = "zscore", threshold: float = 3):
        if not self.has_data():
            return "No dataset loaded. Cannot handle outliers."

        df = self.data.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            return "No numeric columns found for outlier processing."

        try:
            if method == "zscore":
                from scipy.stats import zscore

                z_scores = np.abs(df[num_cols].apply(zscore))
                df = df[(z_scores < float(threshold)).all(axis=1)]

            elif method == "iqr":
                for col in num_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower) & (df[col] <= upper)]

            else:
                return f"Invalid method '{method}'. Use 'zscore' or 'iqr'."

            self.data = df
            return df

        except Exception as e:
            return f"Error handling outliers: {e}"

    # ------------------------------------------------------------------
    # ✅ STANDARDIZE COLUMN NAMES
    # ------------------------------------------------------------------
    def standardize_columns(self):
        if not self.has_data():
            return "No dataset loaded. Cannot standardize columns."

        try:
            self.data.columns = [
                str(col).strip().lower().replace(" ", "_") for col in self.data.columns
            ]
            return self.data
        except Exception as e:
            return f"Error standardizing columns: {e}"

    # ------------------------------------------------------------------
    # ✅ REMOVE DUPLICATES
    # ------------------------------------------------------------------
    def remove_duplicates(self):
        if not self.has_data():
            return "No dataset loaded. Cannot remove duplicates."

        try:
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            after = len(self.data)
            return {
                "duplicates_removed": before - after,
                "rows_remaining": after,
                "data": self.data,
            }
        except Exception as e:
            return f"Error removing duplicates: {e}"

    # ------------------------------------------------------------------
    # ✅ TYPE CONVERSION
    # ------------------------------------------------------------------
    def convert_types(self, column: Optional[str] = None, target_type: Optional[str] = None):
        if not self.has_data():
            return "No dataset loaded. Cannot convert types."

        df = self.data.copy()

        try:
            if column and target_type:
                if column not in df.columns:
                    return f"Column '{column}' not found."

                df[column] = df[column].astype(target_type)
                self.data = df
                return df

            # Auto infer types
            df = df.convert_dtypes()
            self.data = df
            return df

        except Exception as e:
            return f"Error converting types: {e}"

    # ------------------------------------------------------------------
    # ✅ SAFE EXECUTION DISPATCHER (required by orchestrator)
    # ------------------------------------------------------------------
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        if not hasattr(self, function_name):
            return f"Method '{function_name}' not found in DataCleaningAgent."

        try:
            method = getattr(self, function_name)
            if callable(method):
                return method(**kwargs)
            return method
        except Exception as e:
            return f"❌ Error executing {function_name}: {e}"
