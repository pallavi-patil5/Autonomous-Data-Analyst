"""
Enhanced ML Agent - Fully Robust, Orchestrator-Compatible, Exception-Free
Supports classification, regression, clustering, cross-validation, and feature importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

from base_agent import BaseAgent

# ML Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


class MLAgent(BaseAgent):

    def __init__(self, data: Optional[pd.DataFrame]):
        super().__init__()
        self.data: pd.DataFrame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    # ------------------------------------------------------------------
    # ✅ Helpers
    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

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
    # ✅ Capabilities exposed to orchestrator
    # ------------------------------------------------------------------
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "train_classification",
                "description": "Train classification model",
                "parameters": ["target", "features", "model_type", "test_size"],
                "examples": ["predict disease using random forest"]
            },
            {
                "function_name": "train_regression",
                "description": "Train regression model",
                "parameters": ["target", "features", "model_type", "test_size"],
                "examples": ["predict house price"]
            },
            {
                "function_name": "perform_clustering",
                "description": "Cluster data",
                "parameters": ["n_clusters", "features", "method"],
                "examples": ["cluster customers"]
            },
            {
                "function_name": "cross_validate_model",
                "description": "Cross-validate a model",
                "parameters": ["target", "features", "model_type", "cv_folds"],
                "examples": ["cross validate random forest"]
            },
            {
                "function_name": "feature_importance",
                "description": "Tree-based feature importance",
                "parameters": ["target", "features", "model_type"],
                "examples": ["rank feature importance"]
            },
            # Universal helpers
            {
                "function_name": "get_preview",
                "description": "Preview dataset",
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
                "examples": ["how many rows"]
            }
        ]

    # ------------------------------------------------------------------
    # ✅ CLASSIFICATION
    # ------------------------------------------------------------------
    def train_classification(self, target: str, features=None, model_type="logistic", test_size=0.2):

        if not self.has_data():
            return "No dataset loaded."

        if target not in self.data.columns:
            return f"Target '{target}' not found."

        # ------------------ Normalize feature list ------------------
        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        if not features:
            features = [c for c in self.data.columns if c != target]

        missing = [f for f in features if f not in self.data.columns]
        if missing:
            return f"Missing features: {missing}"

        # Drop missing rows
        df = self.data[features + [target]].dropna()
        if df.empty:
            return "No usable data after dropping missing values."

        X = df[features]
        y = df[target]

        # Convert classification labels to numeric
        try:
            y = y.astype(int)
        except:
            from sklearn.preprocessing import LabelEncoder
            try:
                y = LabelEncoder().fit_transform(y)
            except:
                return "Target must be numeric or convertible to numeric."

        # ------------------ Model name normalization ------------------
        model_aliases = {
            "lr": "logistic", "logistic_regression": "logistic",
            "rf": "random_forest", "randomforest": "random_forest",
            "svm": "svm", "svc": "svm",
            "knn": "knn",
            "decisiontree": "decision_tree",
            "gb": "gradient_boosting", "gbc": "gradient_boosting",
            "naivebayes": "naive_bayes"
        }
        model_type = model_aliases.get(model_type.lower().replace(" ", "_"), model_type)

        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "naive_bayes": GaussianNB()
        }

        if model_type not in models:
            return f"Unsupported model '{model_type}'."

        model = models[model_type]

        # ------------------ Train/test split ------------------
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42
            )

        # ------------------ Train ------------------
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            return f"Training error: {e}"

        # ------------------ Evaluate ------------------
        try:
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, zero_division=0)
            cm = confusion_matrix(y_test, preds).tolist()
        except Exception as e:
            return f"Evaluation error: {e}"

        return {
            "model": model_type,
            "accuracy": round(acc, 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "features_used": features,
            "target": target
        }

    # ------------------------------------------------------------------
    # ✅ REGRESSION
    # ------------------------------------------------------------------
    def train_regression(self, target: str, features=None, model_type="linear", test_size=0.2):

        if not self.has_data():
            return "No dataset loaded."

        if target not in self.data.columns:
            return f"Target '{target}' not found."

        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        if not features:
            features = [c for c in self.data.columns if c != target]

        for f in features:
            if f not in self.data.columns:
                return f"Feature '{f}' not found."

        df = self.data[features + [target]].dropna()
        if df.empty:
            return "No usable rows for regression."

        X = df[features]
        y = df[target]

        model_aliases = {
            "linear_regression": "linear",
            "rf": "random_forest",
            "randomforest": "random_forest",
            "gb": "gradient_boosting",
            "svr": "svr",
            "decisiontree": "decision_tree"
        }
        model_type = model_aliases.get(model_type.lower(), model_type)

        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(),
            "gradient_boosting": GradientBoostingRegressor(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "svr": SVR(),
            "decision_tree": DecisionTreeRegressor()
        }

        if model_type not in models:
            return f"Unsupported regression model '{model_type}'."

        model = models[model_type]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42
            )
        except Exception as e:
            return f"Train-test split error: {e}"

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as e:
            return f"Training error: {e}"

        return {
            "model": model_type,
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2_score": float(r2_score(y_test, preds)),
            "features_used": features,
            "target": target
        }

    # ------------------------------------------------------------------
    # ✅ CLUSTERING
    # ------------------------------------------------------------------
    def perform_clustering(self, n_clusters=3, features=None, method="kmeans"):

        if not self.has_data():
            return "No dataset loaded."

        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        if not features:
            features = self.data.select_dtypes(include=np.number).columns.tolist()

        if not features:
            return "No numeric features available for clustering."

        for f in features:
            if f not in self.data.columns:
                return f"Feature '{f}' not found."

        df = self.data[features].dropna()
        if df.empty:
            return "No usable rows for clustering."

        method = method.lower().strip()

        if method in ["kmeans", "k-means"]:
            model = KMeans(n_clusters=int(n_clusters), n_init=10)
        elif method == "dbscan":
            model = DBSCAN()
        elif method in ["hierarchical", "agglomerative"]:
            model = AgglomerativeClustering(n_clusters=int(n_clusters))
        else:
            return "Unsupported clustering method."

        try:
            labels = model.fit_predict(df)
        except Exception as e:
            return f"Clustering error: {e}"

        self.data.loc[df.index, "cluster"] = labels

        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "method": method,
            "n_clusters_found": cluster_count,
            "labels": labels.tolist(),
            "message": "Clustering completed. 'cluster' column added."
        }

    # ------------------------------------------------------------------
    # ✅ CROSS VALIDATION
    # ------------------------------------------------------------------
    def cross_validate_model(self, target: str, features=None, model_type="logistic", cv_folds=5):

        if not self.has_data():
            return "No dataset loaded."

        if target not in self.data.columns:
            return f"Target '{target}' not found."

        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        if not features:
            features = [c for c in self.data.columns if c != target]

        df = self.data[features + [target]].dropna()

        if df.empty:
            return "Not enough data for cross-validation."

        X = df[features]
        y = df[target]

        # Model selection
        
        models = {
            "logistic": LogisticRegression(max_iter=200),
            "random_forest": RandomForestClassifier(),
            "linear": LinearRegression()
        }

        model_type = model_type.lower().strip()
        if model_type not in models:
            return "Unsupported model for cross-validation."

        try:
            scores = cross_val_score(models[model_type], X, y, cv=int(cv_folds))
        except Exception as e:
            return f"Cross-validation error: {e}"

        return {
            "model": model_type,
            "folds": cv_folds,
            "scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std())
        }

    # ------------------------------------------------------------------
    # ✅ FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    def feature_importance(self, target: str, features=None, model_type="random_forest"):

        if not self.has_data():
            return "No dataset loaded."

        if target not in self.data.columns:
            return f"Target '{target}' not found."

        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        if not features:
            features = [c for c in self.data.columns if c != target]

        df = self.data[features + [target]].dropna()

        if df.empty:
            return "Not enough data for feature importance."

        X = df[features]
        y = df[target]

        model_type = model_type.lower().strip()

        if model_type == "random_forest":
            model = RandomForestClassifier()
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier()
        else:
            return "Feature importance only supports tree-based models."

        try:
            model.fit(X, y)
            importances = model.feature_importances_
        except Exception as e:
            return f"Error computing importance: {e}"

        df_out = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return df_out

    # ------------------------------------------------------------------
    # ✅ EXECUTION DISPATCHER REQUIRED BY ORCHESTRATOR
    # ------------------------------------------------------------------
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        if not hasattr(self, function_name):
            return f"Method '{function_name}' not found in MLAgent."

        try:
            method = getattr(self, function_name)
            if callable(method):
                return method(**kwargs)
            return method
        except Exception as e:
            return f"Error executing {function_name}: {e}"
