"""
Critic Agent - Autonomous Model Evaluator
Evaluates ML model performance and generates structured recommendations.
"""

import logging
from typing import Any, Dict, List, Optional

from base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CriticAgent] %(message)s")
logger = logging.getLogger("CriticAgent")


class CriticAgent(BaseAgent):

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # ✅ Capabilities
    # ------------------------------------------------------------------
    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_name": "evaluate_model_performance",
                "description": "Evaluate model metrics and return structured critique with severity",
                "parameters": ["accuracy", "precision", "recall", "f1_score", "train_score", "test_score", "model_type"],
                "examples": ["evaluate model performance", "critique model results"]
            },
            {
                "function_name": "detect_overfitting",
                "description": "Detect overfitting by comparing train vs test scores",
                "parameters": ["train_score", "test_score"],
                "examples": ["check for overfitting", "is model overfitting"]
            },
            {
                "function_name": "detect_underfitting",
                "description": "Detect underfitting from low train and test scores",
                "parameters": ["train_score", "test_score"],
                "examples": ["check for underfitting", "is model underfitting"]
            },
            {
                "function_name": "detect_class_imbalance",
                "description": "Detect class imbalance from precision/recall/f1 discrepancy",
                "parameters": ["precision", "recall", "f1_score", "accuracy"],
                "examples": ["check class imbalance", "is data imbalanced"]
            },
            {
                "function_name": "recommend_improvements",
                "description": "Generate improvement recommendations based on detected issues",
                "parameters": ["accuracy", "precision", "recall", "f1_score", "train_score", "test_score", "model_type"],
                "examples": ["how to improve model", "suggest improvements"]
            }
        ]

    # ------------------------------------------------------------------
    # ✅ Core Evaluation
    # ------------------------------------------------------------------
    def evaluate_model_performance(
        self,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1_score: float = None,
        train_score: float = None,
        test_score: float = None,
        model_type: str = "unknown"
    ) -> Dict[str, Any]:

        logger.info(f"CriticAgent invoked for model: {model_type}")

        issues = []
        recommendations = []
        severity_score = 0

        # --- Use test_score as accuracy fallback ---
        acc = accuracy if accuracy is not None else test_score

        # --- Overfitting check ---
        if train_score is not None and test_score is not None:
            gap = train_score - test_score
            if gap > 0.15:
                issues.append(f"Overfitting detected: train={train_score:.2f}, test={test_score:.2f}, gap={gap:.2f}")
                recommendations.append("Tune hyperparameters to reduce model complexity")
                recommendations.append("Apply regularization (Ridge/Lasso for regression, C param for SVM)")
                recommendations.append("Use cross-validation for more reliable evaluation")
                severity_score += 2

        # --- Underfitting check ---
        if train_score is not None and test_score is not None:
            if train_score < 0.65 and test_score < 0.65:
                issues.append(f"Underfitting detected: train={train_score:.2f}, test={test_score:.2f}")
                recommendations.append("Try a more complex model like Random Forest or Gradient Boosting")
                recommendations.append("Perform feature engineering to add more informative features")
                recommendations.append("Collect more training data")
                severity_score += 2

        # --- Low accuracy check ---
        if acc is not None:
            if acc < 0.60:
                issues.append(f"Low accuracy: {acc:.2f}")
                recommendations.append("Try Random Forest or XGBoost for better performance")
                recommendations.append("Apply feature scaling (StandardScaler or MinMaxScaler)")
                recommendations.append("Perform feature selection to remove irrelevant features")
                severity_score += 2
            elif acc < 0.75:
                issues.append(f"Moderate accuracy: {acc:.2f} — room for improvement")
                recommendations.append("Tune hyperparameters using GridSearchCV")
                recommendations.append("Try ensemble methods like Gradient Boosting or XGBoost")
                severity_score += 1

        # --- Class imbalance check ---
        if precision is not None and recall is not None:
            gap = abs(precision - recall)
            if gap > 0.15:
                issues.append(f"Class imbalance suspected: precision={precision:.2f}, recall={recall:.2f}")
                recommendations.append("Handle class imbalance using SMOTE or class_weight='balanced'")
                recommendations.append("Use F1-score or AUC-ROC as primary evaluation metric instead of accuracy")
                severity_score += 2

        # --- Low F1 check ---
        if f1_score is not None and f1_score < 0.65:
            issues.append(f"Low F1-score: {f1_score:.2f}")
            recommendations.append("Handle class imbalance using SMOTE")
            recommendations.append("Try threshold tuning to balance precision and recall")
            severity_score += 1

        # --- Model-specific suggestions ---
        if model_type.lower() in ["logistic", "logistic_regression", "lr"]:
            if acc is not None and acc < 0.80:
                recommendations.append("Try Random Forest or SVM for potentially better performance")

        if model_type.lower() in ["svm", "svc"]:
            recommendations.append("Apply feature scaling — SVM is sensitive to unscaled features")

        if model_type.lower() in ["knn"]:
            recommendations.append("Apply feature scaling — KNN is distance-based and requires scaled features")
            recommendations.append("Tune the number of neighbors (k) for better results")

        # --- Deduplicate recommendations ---
        recommendations = list(dict.fromkeys(recommendations))

        # --- Severity ---
        if severity_score == 0:
            severity = "low"
        elif severity_score <= 2:
            severity = "medium"
        else:
            severity = "high"

        if not issues:
            issues.append("No critical issues detected. Model performance looks acceptable.")

        logger.info(f"Issues detected: {issues}")
        logger.info(f"Recommendations generated: {recommendations}")

        return {
            "issues_detected": issues,
            "recommendations": recommendations,
            "severity": severity
        }

    # ------------------------------------------------------------------
    # ✅ Overfitting Detection
    # ------------------------------------------------------------------
    def detect_overfitting(
        self,
        train_score: float = None,
        test_score: float = None
    ) -> Dict[str, Any]:

        if train_score is None or test_score is None:
            return {"issues_detected": ["train_score and test_score are required"], "recommendations": [], "severity": "low"}

        gap = train_score - test_score
        if gap > 0.15:
            return {
                "issues_detected": [f"Overfitting detected: gap={gap:.2f} (train={train_score:.2f}, test={test_score:.2f})"],
                "recommendations": [
                    "Tune hyperparameters to reduce model complexity",
                    "Apply regularization",
                    "Use cross-validation",
                    "Collect more training data"
                ],
                "severity": "high" if gap > 0.25 else "medium"
            }

        return {
            "issues_detected": ["No overfitting detected"],
            "recommendations": [],
            "severity": "low"
        }

    # ------------------------------------------------------------------
    # ✅ Underfitting Detection
    # ------------------------------------------------------------------
    def detect_underfitting(
        self,
        train_score: float = None,
        test_score: float = None
    ) -> Dict[str, Any]:

        if train_score is None or test_score is None:
            return {"issues_detected": ["train_score and test_score are required"], "recommendations": [], "severity": "low"}

        if train_score < 0.65 and test_score < 0.65:
            return {
                "issues_detected": [f"Underfitting detected: train={train_score:.2f}, test={test_score:.2f}"],
                "recommendations": [
                    "Try a more complex model like Random Forest or Gradient Boosting",
                    "Perform feature engineering",
                    "Collect more training data"
                ],
                "severity": "high"
            }

        return {
            "issues_detected": ["No underfitting detected"],
            "recommendations": [],
            "severity": "low"
        }

    # ------------------------------------------------------------------
    # ✅ Class Imbalance Detection
    # ------------------------------------------------------------------
    def detect_class_imbalance(
        self,
        precision: float = None,
        recall: float = None,
        f1_score: float = None,
        accuracy: float = None
    ) -> Dict[str, Any]:

        issues = []
        recommendations = []

        if precision is not None and recall is not None:
            gap = abs(precision - recall)
            if gap > 0.15:
                issues.append(f"Class imbalance suspected: precision={precision:.2f}, recall={recall:.2f}, gap={gap:.2f}")
                recommendations.append("Handle class imbalance using SMOTE")
                recommendations.append("Use class_weight='balanced' in your model")
                recommendations.append("Use F1-score or AUC-ROC as primary metric")

        if accuracy is not None and f1_score is not None:
            if accuracy - f1_score > 0.10:
                issues.append(f"Accuracy much higher than F1 ({accuracy:.2f} vs {f1_score:.2f}) — likely imbalanced classes")
                recommendations.append("Oversample minority class using SMOTE or undersample majority class")

        if not issues:
            issues.append("No class imbalance detected")

        severity = "high" if len(issues) > 1 else ("medium" if issues[0] != "No class imbalance detected" else "low")

        return {
            "issues_detected": issues,
            "recommendations": list(dict.fromkeys(recommendations)),
            "severity": severity
        }

    # ------------------------------------------------------------------
    # ✅ Recommend Improvements
    # ------------------------------------------------------------------
    def recommend_improvements(
        self,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1_score: float = None,
        train_score: float = None,
        test_score: float = None,
        model_type: str = "unknown"
    ) -> Dict[str, Any]:
        return self.evaluate_model_performance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            train_score=train_score,
            test_score=test_score,
            model_type=model_type
        )

    # ------------------------------------------------------------------
    # ✅ Safe dispatcher
    # ------------------------------------------------------------------
    def execute_capability(self, function_name: str, **kwargs) -> Any:
        if not hasattr(self, function_name):
            return f"Method '{function_name}' not found in CriticAgent."
        try:
            method = getattr(self, function_name)
            return method(**kwargs) if callable(method) else method
        except Exception as e:
            return f"❌ Error executing {function_name} in CriticAgent: {e}"
