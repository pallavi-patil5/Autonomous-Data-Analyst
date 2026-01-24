"""
Enhanced EDA Agent - Self-Describing (Robust)
- Adds execute_capability() safe dispatcher
- Adds get_preview / get_shape / get_row_count
- All methods return data/figures or friendly messages; no hard failures
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from base_agent import BaseAgent

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


class EDAAgent(BaseAgent):
    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__()
        self.data: pd.DataFrame = (
            data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        )

    # ---- dataset utils ----
    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    def set_data(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    def get_preview(self, n: int = 10) -> Union[pd.DataFrame, str]:
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df) with a non-empty DataFrame."
        return self.data.head(max(1, int(n)))

    def get_shape(self) -> Dict[str, int]:
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return {"rows": 0, "columns": 0}
        r, c = self.data.shape
        return {"rows": int(r), "columns": int(c)}

    def get_row_count(self) -> int:
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return 0
        return int(self.data.shape[0])

    # ---- capabilities ----
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "summary_statistics",
                "description": "Descriptive statistics for numeric & categorical",
                "parameters": [],
                "examples": ["summary stats", "describe dataset"],
            },
            {
                "function_name": "get_data_info",
                "description": "Shape, columns, dtypes, missing, memory",
                "parameters": [],
                "examples": ["dataset info"],
            },
            {
                "function_name": "get_columns",
                "description": "List all column names",
                "parameters": [],
                "examples": ["what columns do we have"],
            },
            {
                "function_name": "calculate_column_stats",
                "description": "Compute mean/median/mode/std/min/max for a column",
                "parameters": ["column_name", "metrics"],
                "examples": ["stats for age"],
            },
            {
                "function_name": "missing_values",
                "description": "Count & percent missing per column",
                "parameters": [],
                "examples": ["show missing values"],
            },
            {
                "function_name": "correlation_heatmap",
                "description": "Correlation heatmap for numeric features",
                "parameters": [],
                "examples": ["plot correlation matrix"],
            },
            {
                "function_name": "get_pairwise_correlation",
                "description": "Pearson correlation between two numeric columns",
                "parameters": ["col1", "col2"],
                "examples": ["corr age and salary"],
            },
            {
                "function_name": "value_counts",
                "description": "Frequency distribution of a column",
                "parameters": ["column_name"],
                "examples": ["value counts for gender"],
            },
            {
                "function_name": "get_unique_values",
                "description": "Unique values in a column",
                "parameters": ["column_name"],
                "examples": ["unique countries"],
            },
            {
                "function_name": "get_column_data_types",
                "description": "Column -> dtype mapping",
                "parameters": [],
                "examples": ["what are the data types"],
            },
            {
                "function_name": "get_dataset_overview",
                "description": "Overview with shape, dtypes, missing, memory, preview",
                "parameters": [],
                "examples": ["overview"],
            },
            {
                "function_name": "get_preview",
                "description": "First N rows (default 10)",
                "parameters": ["n"],
                "examples": ["first 10 rows", "head"],
            },
            {
                "function_name": "get_shape",
                "description": "Return dataset shape as rows and columns",
                "parameters": [],
                "examples": ["how many rows and columns"],
            },
            {
                "function_name": "get_row_count",
                "description": "Return total number of rows",
                "parameters": [],
                "examples": ["total row count"],
            },
        ]

    # ---- EDA operations ----
    def summary_statistics(self) -> pd.DataFrame:
        if not self.has_data():
            return pd.DataFrame({"Message": ["No dataset loaded yet. Use set_data(df)."]})
        numeric_summary = self.data.describe(include=[np.number]).T
        categorical_summary = self.data.describe(include=["object", "category"]).T
        parts = []
        if not numeric_summary.empty:
            parts.append(numeric_summary)
        if not categorical_summary.empty:
            parts.append(categorical_summary)
        return pd.concat(parts, axis=0) if parts else pd.DataFrame()

    def get_data_info(self) -> Dict[str, Any]:
        if not self.has_data():
            return {
                "Shape": "0 rows × 0 columns",
                "Columns": [],
                "Data Types": {},
                "Missing Values": {},
                "Memory Usage": "0.00 MB",
            }
        return {
            "Shape": f"{self.data.shape[0]} rows × {self.data.shape[1]} columns",
            "Columns": list(self.data.columns),
            "Data Types": self.data.dtypes.astype(str).to_dict(),
            "Missing Values": self.data.isnull().sum().to_dict(),
            "Memory Usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        }

    def get_column_data_types(self) -> Dict[str, str]:
        if not self.has_data():
            return {}
        return self.data.dtypes.astype(str).to_dict()

    def get_columns(self) -> List[str]:
        return list(self.data.columns)

    def get_column_count(self, column_name: str) -> Dict[str, Any]:
        if column_name in self.data.columns:
            return {column_name: int(self.data[column_name].count())}
        return {"error": f"Column '{column_name}' not found."}

    def calculate_column_stats(
        self, column_name: str, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if column_name not in self.data.columns:
            return {"error": f"Column '{column_name}' not found."}
        if metrics is None:
            metrics = ["mean", "median", "mode", "std", "min", "max"]
        col = self.data[column_name]
        results: Dict[str, Any] = {"column": column_name}
        if pd.api.types.is_numeric_dtype(col):
            if "mean" in metrics:
                results["mean"] = round(float(col.mean()), 4)
            if "median" in metrics:
                results["median"] = round(float(col.median()), 4)
            if "mode" in metrics:
                results["mode"] = col.mode().iloc[0] if not col.mode().empty else "N/A"
            if "std" in metrics:
                results["std"] = round(float(col.std()), 4)
            if "min" in metrics:
                results["min"] = round(float(col.min()), 4)
            if "max" in metrics:
                results["max"] = round(float(col.max()), 4)
        else:
            if "mode" in metrics:
                results["mode"] = col.mode().iloc[0] if not col.mode().empty else "N/A"
            results["unique_count"] = int(col.nunique(dropna=False))
            results["most_common"] = col.value_counts(dropna=False).head(3).to_dict()
        return results

    def missing_values(self) -> pd.DataFrame:
        if not self.has_data():
            return pd.DataFrame({"Message": ["No dataset loaded yet. Use set_data(df)."]})
        missing_series = self.data.isnull().sum()
        missing_df = missing_series[missing_series > 0].to_frame(name="Missing Count")
        if missing_df.empty:
            return pd.DataFrame({"Message": ["No missing values found!"]})
        missing_df["% Missing"] = (missing_df["Missing Count"] / len(self.data)) * 100
        return missing_df.sort_values(by="Missing Count", ascending=False)

    def correlation_heatmap(self):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        corr = self.data.corr(numeric_only=True)
        if corr.empty:
            return "No numeric columns to correlate."
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        return plt

    def get_pairwise_correlation(self, col1: str, col2: str) -> str:
        if col1 not in self.data.columns or col2 not in self.data.columns:
            return f"Error: One or both columns ('{col1}', '{col2}') not found."
        if pd.api.types.is_numeric_dtype(self.data[col1]) and pd.api.types.is_numeric_dtype(
            self.data[col2]
        ):
            corr_value = self.data[col1].corr(self.data[col2], method="pearson")
            return f"Pearson Correlation between {col1} and {col2}: {corr_value:.4f}"
        return "Cannot calculate Pearson correlation between mixed or non-numeric data types."

    def value_counts(self, column_name: str) -> Union[pd.DataFrame, Dict[str, str]]:
        if column_name in self.data.columns:
            return self.data[column_name].value_counts(dropna=False).to_frame(name="count")
        return {"error": f"Column '{column_name}' not found."}

    def get_unique_values(self, column_name: str) -> Dict[str, Any]:
        if column_name in self.data.columns:
            return {"column": column_name, "unique_values": list(self.data[column_name].unique())}
        return {"error": f"Column '{column_name}' not found."}

    def get_dataset_overview(self) -> Dict[str, Any]:
        if not self.has_data():
            return {
                "shape": "0 rows × 0 columns",
                "columns": [],
                "dtypes": {},
                "missing_values": {},
                "memory_usage": "0.00 MB",
                "preview": pd.DataFrame(),
            }
        return {
            "shape": f"{self.data.shape[0]} rows × {self.data.shape[1]} columns",
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "preview": self.data.head(5),
        }

    # ---- SAFE DISPATCH expected by orchestrator ----
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        if not hasattr(self, function_name):
            return (
                f"Method '{function_name}' not found in EDAAgent."
                f" Available: {sorted(m for m in dir(self) if not m.startswith('_'))}"
            )
        method = getattr(self, function_name)
        try:
            return method(**kwargs) if callable(method) else method
        except Exception as e:
            return f"❌ Error executing {function_name}: {e}"
