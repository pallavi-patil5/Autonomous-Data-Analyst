"""
Enhanced Feature Engineering Agent - Fully Robust & Orchestrator-Compatible
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from base_agent import BaseAgent


class FeatureEngineeringAgent(BaseAgent):

    def __init__(self, data: Optional[pd.DataFrame]):
        super().__init__()
        self.data: pd.DataFrame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    # ------------------------------------------------------------------
    # ✅ HELPERS
    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    def set_data(self, df: pd.DataFrame):
        self.data = df.copy()

    def get_preview(self, n: int = 10):
        if not self.has_data():
            return "No dataset loaded."
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
    # ✅ CAPABILITIES
    # ------------------------------------------------------------------
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "encode_categoricals",
                "description": "Label encode all categorical columns",
                "parameters": [],
                "examples": ["encode categorical variables"]
            },
            {
                "function_name": "scale_features",
                "description": "Scale numeric columns using StandardScaler",
                "parameters": ["columns", "target_column"],
                "examples": ["scale features"]
            },
            {
                "function_name": "normalize_features",
                "description": "Normalize numeric columns using MinMaxScaler",
                "parameters": ["columns"],
                "examples": ["normalize features"]
            },
            {
                "function_name": "apply_tfidf",
                "description": "Apply TF-IDF vectorization",
                "parameters": ["column", "max_features"],
                "examples": ["apply tfidf"]
            },
            {
                "function_name": "extract_datetime_features",
                "description": "Extract date parts",
                "parameters": ["column"],
                "examples": ["extract datetime features"]
            },
            {
                "function_name": "create_interaction_features",
                "description": "Multiply two columns to make interaction",
                "parameters": ["col1", "col2", "name"],
                "examples": ["create interaction"]
            },
            {
                "function_name": "create_polynomial_features",
                "description": "Add polynomial terms",
                "parameters": ["columns", "degree"],
                "examples": ["add polynomial features"]
            },
            {
                "function_name": "bin_numeric_column",
                "description": "Discretize numeric data into bins",
                "parameters": ["column", "bins", "labels"],
                "examples": ["bin age"]
            },
            # Universal helper functions
            {
                "function_name": "get_preview",
                "description": "Preview first N rows",
                "parameters": ["n"],
                "examples": ["first 10 rows"]
            },
            {
                "function_name": "get_shape",
                "description": "Dataset shape",
                "parameters": [],
                "examples": ["dataset size"]
            },
            {
                "function_name": "get_row_count",
                "description": "Row count",
                "parameters": [],
                "examples": ["number of rows"]
            }
        ]

    # ------------------------------------------------------------------
    # ✅ ENCODING
    # ------------------------------------------------------------------
    def encode_categoricals(self):
        if not self.has_data():
            return "No dataset loaded."

        try:
            df = self.data.copy()
            le = LabelEncoder()

            cat_cols = df.select_dtypes(include="object").columns

            for col in cat_cols:
                df[col] = le.fit_transform(df[col].astype(str))

            self.data = df
            return df

        except Exception as e:
            return f"Error encoding categoricals: {e}"

    # ------------------------------------------------------------------
    # ✅ STANDARD SCALING
    # ------------------------------------------------------------------
    def scale_features(self, columns=None, target_column=None):
        if not self.has_data():
            return "No dataset loaded."

        try:
            df = self.data.copy()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if columns:
                num_cols = list(set(num_cols) & set(columns))

            if target_column in num_cols:
                num_cols.remove(target_column)

            if not num_cols:
                return "No numeric columns available for scaling."

            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

            self.data = df
            return df

        except Exception as e:
            return f"Error scaling features: {e}"

    # ------------------------------------------------------------------
    # ✅ NORMALIZATION
    # ------------------------------------------------------------------
    def normalize_features(self, columns=None):
        if not self.has_data():
            return "No dataset loaded."

        try:
            df = self.data.copy()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if columns:
                num_cols = list(set(num_cols) & set(columns))

            if not num_cols:
                return "No numeric columns available for normalization."

            scaler = MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

            self.data = df
            return df

        except Exception as e:
            return f"Error normalizing features: {e}"

    # ------------------------------------------------------------------
    # ✅ TF-IDF
    # ------------------------------------------------------------------
    def apply_tfidf(self, column: str, max_features: int = 100):
        if not self.has_data():
            return "No dataset loaded."
        if column not in self.data.columns:
            return f"Column '{column}' not found."

        try:
            df = self.data.copy()

            tfidf = TfidfVectorizer(max_features=max_features)
            matrix = tfidf.fit_transform(df[column].astype(str))

            tfidf_df = pd.DataFrame(
                matrix.toarray(),
                columns=tfidf.get_feature_names_out(),
                index=df.index
            )

            df = df.drop(columns=[column], errors="ignore")
            df = pd.concat([df, tfidf_df], axis=1)

            self.data = df
            return df

        except Exception as e:
            return f"Error applying TF-IDF: {e}"

    # ------------------------------------------------------------------
    # ✅ DATETIME FEATURE EXTRACTION
    # ------------------------------------------------------------------
    def extract_datetime_features(self, column: str):
        if not self.has_data():
            return "No dataset loaded."
        if column not in self.data.columns:
            return f"Column '{column}' not found."

        try:
            df = self.data.copy()
            df[column] = pd.to_datetime(df[column], errors="coerce")

            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek

            self.data = df
            return df

        except Exception as e:
            return f"Error extracting datetime features: {e}"

    # ------------------------------------------------------------------
    # ✅ INTERACTION FEATURES
    # ------------------------------------------------------------------
    def create_interaction_features(self, col1: str, col2: str, name: Optional[str] = None):
        if not self.has_data():
            return "No dataset loaded."
        if col1 not in self.data.columns or col2 not in self.data.columns:
            return f"One or both columns not found: {col1}, {col2}"

        try:
            df = self.data.copy()
            if name is None:
                name = f"{col1}_x_{col2}"
            df[name] = df[col1] * df[col2]

            self.data = df
            return df

        except Exception as e:
            return f"Error creating interaction features: {e}"

    # ------------------------------------------------------------------
    # ✅ POLYNOMIAL FEATURES
    # ------------------------------------------------------------------
    def create_polynomial_features(self, columns: List[str], degree: int = 2):
        if not self.has_data():
            return "No dataset loaded."

        try:
            df = self.data.copy()

            for col in columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                for d in range(2, int(degree) + 1):
                    df[f"{col}_power_{d}"] = df[col] ** d

            self.data = df
            return df

        except Exception as e:
            return f"Error creating polynomial features: {e}"

    # ------------------------------------------------------------------
    # ✅ BIN NUMERIC COLUMN
    # ------------------------------------------------------------------
    def bin_numeric_column(self, column: str, bins: int = 5, labels=None):
        if not self.has_data():
            return "No dataset loaded."
        if column not in self.data.columns:
            return f"Column '{column}' not found."

        try:
            df = self.data.copy()
            df[f"{column}_binned"] = pd.cut(df[column], bins=int(bins), labels=labels)

            self.data = df
            return df

        except Exception as e:
            return f"Error binning column: {e}"

    # ------------------------------------------------------------------
    # ✅ SAFE DISPATCHER (ORCHESTRATOR USES THIS)
    # ------------------------------------------------------------------
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        if not hasattr(self, function_name):
            return f"Method '{function_name}' not found in FeatureEngineeringAgent."

        try:
            method = getattr(self, function_name)
            if callable(method):
                return method(**kwargs)
            return method
        except Exception as e:
            return f"❌ Error executing {function_name}: {e}"
