"""
Enhanced Visualization Agent - Universal Plot Creator (Robust)
- Adds execute_capability() safe dispatcher
- Adds get_preview / get_shape / get_row_count
- Fixes capability metadata (no mismatched params)
- Gracefully handles missing/empty datasets
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from base_agent import BaseAgent

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# ---------- helpers ----------
def _ensure_columns_exist(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    missing = [c for c in cols if c not in df.columns]
    return f"Missing columns: {', '.join(map(str, missing))}" if missing else None


def _require_numeric(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return (
        f"Non-numeric columns for this plot: {', '.join(map(str, non_numeric))}"
        if non_numeric
        else None
    )


# ---------- agent ----------
class VisualizationAgent(BaseAgent):
    """
    Universal Plot Creator with guardrails.
    All public methods either return a matplotlib figure/plt, a DataFrame,
    a dict, or a friendly error string — no hard failures.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__()
        self.data: pd.DataFrame = (
            data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        )

    # ---- dataset utilities ----
    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    def set_data(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    def get_preview(self, n: int = 10) -> Union[pd.DataFrame, str]:
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df) with a non-empty DataFrame."
        n = max(1, int(n))
        return self.data.head(n)

    def get_shape(self) -> Dict[str, int]:
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return {"rows": 0, "columns": 0}
        r, c = self.data.shape
        return {"rows": int(r), "columns": int(c)}

    def get_row_count(self) -> int:
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return 0
        return int(self.data.shape[0])

    def get_column_data_types(self) -> Dict[str, str]:
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return {}
        return self.data.dtypes.astype(str).to_dict()

    # ---- capabilities ----
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "plot_histogram",
                "description": "Create histogram for distribution analysis of a numeric column",
                "parameters": ["column", "bins"],
                "examples": ["plot histogram of age", "show distribution of salary"],
            },
            {
                "function_name": "plot_bar",
                "description": "Create bar chart for categorical data counts",
                "parameters": ["column"],
                "examples": ["bar chart of gender", "counts by category"],
            },
            {
                "function_name": "plot_scatter",
                "description": "Scatter plot showing relationship between two numeric variables",
                "parameters": ["x_col", "y_col", "hue"],
                "examples": ["scatter age vs salary", "height vs weight"],
            },
            {
                "function_name": "plot_boxplot",
                "description": "Box plot for outlier detection and distribution visualization",
                "parameters": ["column", "by"],
                "examples": ["boxplot of salary", "age by gender"],
            },
            {
                "function_name": "plot_line",
                "description": "Line chart for time series or sequential data",
                "parameters": ["x_col", "y_col"],
                "examples": ["sales over time", "revenue trend"],
            },
            {
                "function_name": "plot_pie",
                "description": "Pie chart for proportional data",
                "parameters": ["column"],
                "examples": ["pie of categories", "status proportions"],
            },
            {
                "function_name": "plot_heatmap",
                "description": "Heatmap for correlation or matrix data",
                "parameters": ["data_matrix"],
                "examples": ["correlation heatmap", "plot correlation matrix"],
            },
            {
                "function_name": "plot_violin",
                "description": "Violin plot combining box plot with kernel density",
                "parameters": ["column", "by"],
                "examples": ["salary by department", "density by category"],
            },
            {
                "function_name": "plot_pairplot",
                "description": "Pairwise relationships plot for multiple variables",
                "parameters": ["columns", "hue"],
                "examples": ["pairplot of numeric features"],
            },
            {
                "function_name": "create_custom_plot",
                "description": "Create any custom visualization from natural language instruction",
                "parameters": ["instruction"],
                "examples": ["stacked bar by region & product"],
            },
            {
                "function_name": "get_column_data_types",
                "description": "Get data types for all columns in the dataset",
                "parameters": [],
                "examples": ["what are the data types", "show column types"],
            },
            {
                "function_name": "get_preview",
                "description": "Return first N rows of the dataset (default 10)",
                "parameters": ["n"],
                "examples": ["first 10 rows", "preview"],
            },
            {
                "function_name": "get_shape",
                "description": "Return dataset shape as rows and columns",
                "parameters": [],
                "examples": ["how many rows and columns", "dataset shape"],
            },
            {
                "function_name": "get_row_count",
                "description": "Return total number of rows",
                "parameters": [],
                "examples": ["total row count"],
            },
        ]

    # ---- plotting ----
    def plot_histogram(self, column: str, bins: int = 30):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [column]) or _require_numeric(
            self.data, [column]
        )
        if err:
            return err
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column].dropna(), kde=True, bins=int(bins))
        plt.title(f"Distribution: {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        return plt

    def plot_bar(self, column: str):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [column])
        if err:
            return err
        plt.figure(figsize=(10, 6))
        value_counts = self.data[column].value_counts(dropna=False)
        plt.bar(value_counts.index.astype(str), value_counts.values)
        plt.title(f"Count of Categories: {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        return plt

    def plot_scatter(self, x_col: str, y_col: str, hue: Optional[str] = None):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [x_col, y_col]) or _require_numeric(
            self.data, [x_col, y_col]
        )
        if err:
            return err
        plt.figure(figsize=(10, 6))
        if hue and hue in self.data.columns:
            sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue)
        else:
            sns.scatterplot(data=self.data, x=x_col, y=y_col)
        plt.title(f"Scatter Plot: {y_col} vs {x_col}")
        return plt

    def plot_boxplot(self, column: str, by: Optional[str] = None):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [column])
        if err:
            return err
        plt.figure(figsize=(10, 6))
        if by and by in self.data.columns:
            sns.boxplot(data=self.data, x=by, y=column)
        else:
            sns.boxplot(data=self.data, y=column)
        plt.title(f"Boxplot of {column}")
        return plt

    def plot_line(self, x_col: str, y_col: str):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [x_col, y_col]) or _require_numeric(
            self.data, [y_col]
        )
        if err:
            return err
        plt.figure(figsize=(10, 6))
        plt.plot(self.data[x_col], self.data[y_col], marker="o")
        plt.title(f"Line Chart: {y_col} over {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        return plt

    def plot_pie(self, column: str):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [column])
        if err:
            return err
        value_counts = self.data[column].value_counts(dropna=False)
        plt.figure(figsize=(10, 6))
        plt.pie(
            value_counts.values,
            labels=value_counts.index.astype(str),
            autopct="%1.1f%%",
        )
        plt.title(f"Distribution: {column}")
        return plt

    def plot_heatmap(self, data_matrix: Optional[pd.DataFrame] = None):
        if not self.has_data() and data_matrix is None:
            return "No dataset loaded yet. Provide data_matrix or use set_data(df)."
        plt.figure(figsize=(12, 8))
        if data_matrix is None:
            data_matrix = self.data.corr(numeric_only=True)
        if isinstance(data_matrix, pd.DataFrame) and data_matrix.empty:
            return "Heatmap input is empty."
        sns.heatmap(data_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap")
        return plt

    def plot_violin(self, column: str, by: Optional[str] = None):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        err = _ensure_columns_exist(self.data, [column])
        if err:
            return err
        plt.figure(figsize=(10, 6))
        if by and by in self.data.columns:
            sns.violinplot(data=self.data, x=by, y=column)
        else:
            sns.violinplot(data=self.data, y=column)
        plt.title(f"Violin Plot: {column}")
        return plt

    def plot_pairplot(
        self, columns: Optional[Sequence[str]] = None, hue: Optional[str] = None
    ):
        if not self.has_data():
            return "No dataset loaded yet. Use set_data(df)."
        if columns is None:
            columns = list(self.data.select_dtypes(include=np.number).columns[:5])
        err = _ensure_columns_exist(self.data, columns)
        if err:
            return err
        if len(columns) < 2:
            return "Need at least two columns for pairplot."
        hue_arg = hue if (hue and hue in self.data.columns) else None
        try:
            grid = sns.pairplot(self.data[columns], hue=hue_arg)
        except Exception as e:
            return f"Could not create pairplot: {e}"
        return grid.fig

    def create_custom_plot(self, instruction: str) -> str:
        return (
            f"Custom plot instruction received: {instruction}\n"
            "(Requires code generation - will be handled by orchestrator)"
        )

    # ---- SAFE DISPATCH expected by orchestrator ----
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        """
        Call a method by name safely.
        Your orchestrator uses this entry point: agent.execute_capability(...)
        """
        if not hasattr(self, function_name):
            return (
                f"Method '{function_name}' not found in VisualizationAgent."
                f" Available: {sorted(m for m in dir(self) if not m.startswith('_'))}"
            )
        method = getattr(self, function_name)
        try:
            return method(**kwargs) if callable(method) else method
        except Exception as e:
            return f"❌ Error executing {function_name}: {e}"
