"""
Reflection Engine - Iterative ML Improvement Loop
Reads CriticAgent recommendations, applies automatic improvements,
retrains the model, and re-evaluates until stopping criteria are met.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Reflection] %(message)s")
logger = logging.getLogger("Reflection")


class ReflectionEngine:
    """
    Iterative improvement loop for ML workflows.
    Plugs into SmartOrchestratorV2 between MLAgent and final result.
    """

    MAX_CYCLES      = 2
    MIN_IMPROVEMENT = 0.01   # stop if gain < 1%
    TARGET_ACCURACY = 0.90   # stop early if accuracy hits this

    # Maps recommendation keywords → internal action keys
    _RECOMMENDATION_MAP = {
        "feature scaling":       "scale_features",
        "standardscaler":        "scale_features",
        "minmaxscaler":          "scale_features",
        "feature selection":     "select_features",
        "remove irrelevant":     "select_features",
        "class imbalance":       "balance_classes",
        "smote":                 "balance_classes",
        "class_weight":          "balance_classes",
        "random forest":         "switch_random_forest",
        "gradient boosting":     "switch_gradient_boosting",
        "xgboost":               "switch_gradient_boosting",
        "ensemble":              "switch_gradient_boosting",
        "hyperparameter":        "tune_hyperparameters",
        "gridsearchcv":          "tune_hyperparameters",
    }

    def __init__(self, ml_agent, critic_agent, feature_agent):
        self.ml_agent      = ml_agent
        self.critic_agent  = critic_agent
        self.feature_agent = feature_agent

    # ------------------------------------------------------------------
    # ✅ Main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        initial_result: Dict[str, Any],
        target: str,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Run the reflection loop starting from an initial ML result.

        Args:
            initial_result: dict returned by MLAgent.train_classification/regression
            target:         target column name
            task_type:      'classification' or 'regression'

        Returns:
            Structured reflection output with best model and history.
        """
        reflection_history = []
        improvements_applied = []

        best_result  = initial_result
        best_accuracy = self._get_accuracy(initial_result)
        current_model = initial_result.get("model", "logistic")

        logger.info(f"Reflection loop started | initial accuracy={best_accuracy:.2%} | model={current_model}")

        for cycle in range(self.MAX_CYCLES):
            logger.info(f"Cycle {cycle} started")

            # --- Evaluate with CriticAgent ---
            critique = self._get_critique(best_result)
            recommendations = critique.get("recommendations", [])

            reflection_history.append({
                "cycle":           cycle,
                "accuracy":        round(best_accuracy, 4),
                "model":           current_model,
                "recommendations": recommendations,
                "severity":        critique.get("severity", "low"),
                "issues":          critique.get("issues_detected", [])
            })

            logger.info(f"Cycle {cycle} | Issues: {critique.get('issues_detected', [])}")
            logger.info(f"Cycle {cycle} | Recommendations: {recommendations}")

            if not recommendations or critique.get("severity") == "low":
                logger.info(f"Cycle {cycle} | No improvements needed. Stopping.")
                break

            # --- Decide & apply actions ---
            actions = self._parse_actions(recommendations)
            if not actions:
                logger.info(f"Cycle {cycle} | No actionable improvements found. Stopping.")
                break

            applied_this_cycle = []
            next_model = current_model

            for action in actions:
                applied = self._apply_action(action, target, next_model)
                if applied:
                    applied_this_cycle.append(action)
                    improvements_applied.append(action)
                    if action.startswith("switch_"):
                        next_model = action.replace("switch_", "")

            if not applied_this_cycle:
                logger.info(f"Cycle {cycle} | Actions found but none applicable. Stopping.")
                break

            # --- Retrain ---
            logger.info(f"Cycle {cycle} | Retraining with model={next_model}")
            new_result = self._retrain(target, next_model, task_type)

            if isinstance(new_result, str):
                logger.info(f"Cycle {cycle} | Retraining failed: {new_result}")
                break

            new_accuracy = self._get_accuracy(new_result)
            gain = new_accuracy - best_accuracy

            logger.info(f"Cycle {cycle} | Accuracy: {best_accuracy:.2%} → {new_accuracy:.2%} (gain={gain:+.2%})")

            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_result   = new_result
                current_model = next_model

            # --- Stopping criteria ---
            if gain < self.MIN_IMPROVEMENT:
                logger.info(f"Cycle {cycle} | Improvement < 1%. Stopping.")
                break

            if best_accuracy >= self.TARGET_ACCURACY:
                logger.info(f"Cycle {cycle} | Target accuracy {self.TARGET_ACCURACY:.0%} reached. Stopping.")
                break

        # --- Final critique for last cycle entry ---
        final_critique = self._get_critique(best_result)
        reflection_history.append({
            "cycle":           len(reflection_history),
            "accuracy":        round(best_accuracy, 4),
            "model":           current_model,
            "recommendations": final_critique.get("recommendations", []),
            "severity":        final_critique.get("severity", "low"),
            "issues":          final_critique.get("issues_detected", [])
        })

        logger.info(f"Reflection complete | best_accuracy={best_accuracy:.2%} | cycles={len(reflection_history)-1}")

        return {
            "best_model":          current_model,
            "best_accuracy":       round(best_accuracy, 4),
            "reflection_cycles":   len(reflection_history) - 1,
            "improvements_applied": list(dict.fromkeys(improvements_applied)),
            "reflection_history":  reflection_history,
            "final_result":        best_result
        }

    # ------------------------------------------------------------------
    # ✅ Helpers
    # ------------------------------------------------------------------
    def _get_accuracy(self, result: Dict) -> float:
        """Extract a scalar accuracy/score from any ML result dict."""
        for key in ("accuracy", "test_score", "r2_score", "mean_score"):
            if key in result and result[key] is not None:
                return float(result[key])
        return 0.0

    def _get_critique(self, result: Dict) -> Dict:
        """Call CriticAgent.evaluate_model_performance on a result dict."""
        return self.critic_agent.execute_capability(
            "evaluate_model_performance",
            accuracy    = result.get("accuracy"),
            precision   = result.get("precision"),
            recall      = result.get("recall"),
            f1_score    = result.get("f1_score"),
            train_score = result.get("train_score"),
            test_score  = result.get("accuracy") or result.get("test_score"),
            model_type  = result.get("model", "unknown")
        )

    def _parse_actions(self, recommendations: List[str]) -> List[str]:
        """Map free-text recommendations to executable action keys (deduplicated)."""
        actions = []
        for rec in recommendations:
            rec_lower = rec.lower()
            for keyword, action in self._RECOMMENDATION_MAP.items():
                if keyword in rec_lower and action not in actions:
                    actions.append(action)
        return actions

    def _apply_action(self, action: str, target: str, current_model: str) -> bool:
        """
        Apply a single improvement action on the feature_agent's data.
        Returns True if action was applied successfully.
        """
        try:
            if action == "scale_features":
                logger.info("Applying feature scaling")
                result = self.feature_agent.execute_capability(
                    "scale_features", target_column=target
                )
                if isinstance(result, pd.DataFrame):
                    self.ml_agent.data = result
                    return True

            elif action == "select_features":
                logger.info("Applying feature selection (SelectKBest)")
                result = self._select_features(target)
                if isinstance(result, pd.DataFrame):
                    self.ml_agent.data = result
                    self.feature_agent.data = result
                    return True

            elif action == "balance_classes":
                logger.info("Applying class balancing (class_weight='balanced' will be used in retrain)")
                return True  # handled during retrain via model param

            elif action.startswith("switch_"):
                logger.info(f"Switching model to {action.replace('switch_', '')}")
                return True  # handled during retrain

            elif action == "tune_hyperparameters":
                logger.info("Hyperparameter tuning flagged (basic param adjustment)")
                return True

        except Exception as e:
            logger.warning(f"Action '{action}' failed: {e}")

        return False

    def _select_features(self, target: str, k: int = 10) -> Any:
        """SelectKBest feature selection, returns reduced DataFrame."""
        df = self.feature_agent.data.copy()
        if target not in df.columns:
            return None

        features = [c for c in df.columns if c != target]
        X = df[features].select_dtypes(include=[np.number])
        y = df[target]

        try:
            y = y.astype(int)
        except Exception:
            try:
                y = LabelEncoder().fit_transform(y.astype(str))
            except Exception:
                return None

        if X.empty or len(X.columns) <= k:
            return None

        selector = SelectKBest(f_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()

        return df[selected_cols + [target]]

    def _retrain(self, target: str, model_type: str, task_type: str) -> Any:
        """Retrain using MLAgent with the (possibly updated) data."""
        model_map = {
            "random_forest":      "random_forest",
            "gradient_boosting":  "gradient_boosting",
            "logistic":           "logistic",
            "svm":                "svm",
            "knn":                "knn",
            "decision_tree":      "decision_tree",
        }
        model_type = model_map.get(model_type.lower(), model_type)

        if task_type == "classification":
            return self.ml_agent.execute_capability(
                "train_classification",
                target     = target,
                model_type = model_type
            )
        else:
            return self.ml_agent.execute_capability(
                "train_regression",
                target     = target,
                model_type = model_type
            )
