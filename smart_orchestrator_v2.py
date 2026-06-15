"""
Smart Orchestrator v2 - Zero Hardcoded Rules
Autonomous Data Science AI System with Dynamic Capability Discovery
"""
from groq import Groq
import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional

# Import enhanced agents
from enhanced_data_loader_agent import DataLoaderAgent
from enhanced_data_cleaning_agent import DataCleaningAgent
from enhanced_eda_agent import EDAAgent
from enhanced_data_visualization_agent import VisualizationAgent
from enhanced_feature_engineering_agent import FeatureEngineeringAgent
from enhanced_ml_agent import MLAgent
from critic_agent import CriticAgent
from reflection_engine import ReflectionEngine
from report_agent import ReportAgent


class SmartOrchestratorV2:
    """
    Autonomous orchestrator that discovers agent capabilities dynamically.
    NO HARDCODED RULES - fully adaptable to any query.
    """

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

        self.data = None
        self.agents = {}
        self.capability_map = {}

        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents and discover their capabilities."""
        self.agents['loader'] = DataLoaderAgent()
        self.agents['critic'] = CriticAgent()
        self.agents['report'] = ReportAgent()
        self._discover_capabilities()

    def set_data(self, df: pd.DataFrame):
        """Set dataset and reinitialize data-dependent agents."""
        self.data = df

        self.agents['cleaning'] = DataCleaningAgent(self.data)
        self.agents['eda'] = EDAAgent(self.data)
        self.agents['visualization'] = VisualizationAgent(self.data)
        self.agents['feature_engineering'] = FeatureEngineeringAgent(self.data)
        self.agents['ml'] = MLAgent(self.data)
        self.agents['critic'] = CriticAgent()
        self.agents['report'] = ReportAgent()

        self._discover_capabilities()

    def _discover_capabilities(self):
        """
        Dynamically discover all agent capabilities.
        This is the KEY to zero hardcoded rules!
        """
        self.capability_map = {}

        for agent_name, agent_instance in self.agents.items():
            try:
                capabilities = agent_instance.get_capabilities()

                for cap in capabilities:
                    func_name = cap['function_name']
                    self.capability_map[func_name] = {
                        'agent_name': agent_name,
                        'agent_instance': agent_instance,
                        'description': cap['description'],
                        'parameters': cap.get('parameters', []),
                        'examples': cap.get('examples', [])
                    }
            except Exception as e:
                print(f"Warning: Could not discover capabilities for {agent_name}: {e}")

    def get_capability_summary(self) -> str:
        """Generate a summary of all available capabilities for the LLM."""
        summary_lines = ["AVAILABLE CAPABILITIES:\n"]

        for func_name, cap_info in self.capability_map.items():
            summary_lines.append(
                f"- {func_name}: {cap_info['description']}\n"
                f"  Parameters: {cap_info['parameters']}\n"
                f"  Examples: {', '.join(cap_info['examples'][:2])}\n"
            )

        return "\n".join(summary_lines)

    def _call_llm(self, prompt: str) -> str:
        """Call Groq LLM and return response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def process_query(self, query: str):
        """Process user query with intelligent planning."""
        if self.data is None and "load" not in query.lower():
            return [{"task": "error", "result": "⚠️ No dataset loaded. Please upload data first."}]

        available_cols = list(self.data.columns) if self.data is not None else []
        col_list_str = ", ".join([f"'{col}'" for col in available_cols])

        capabilities_summary = self.get_capability_summary()

        prompt = f"""
You are an autonomous data science orchestrator. Your job is to analyze the user's request and create an execution plan.

USER QUERY: "{query}"

DATASET CONTEXT:
- Available columns: {col_list_str}
- Dataset loaded: {'Yes' if self.data is not None else 'No'}

{capabilities_summary}

INSTRUCTIONS:
1. Analyze what the user wants to accomplish
2. Select the appropriate capabilities from the list above
3. Determine the correct parameters based on available columns
4. Handle dependencies (e.g., encoding before ML, scaling for SVM)
5. Return ONLY a valid JSON array of tasks

JSON FORMAT:
[
  {{
    "task": "function_name",
    "params": {{"param1": "value1", "param2": "value2"}}
  }}
]

CRITICAL RULES:
- Use ONLY function names from the AVAILABLE CAPABILITIES list
- Use ONLY column names from the available columns list: {col_list_str}
- For ML tasks, automatically add preprocessing if needed (encode_categoricals, scale_features)
- If the query cannot be solved with available capabilities, use task "respond_text" with explanation
- NEVER invent function names or column names
- Return ONLY valid JSON, no explanations
- For summary/describe queries, use "summary_statistics" task
- For data info queries, use "get_data_info" task

EXAMPLES:
Query: "show distribution of age and predict heart disease with random forest"
Output:
[
  {{"task": "plot_histogram", "params": {{"column": "age"}}}},
  {{"task": "encode_categoricals", "params": {{}}}},
  {{"task": "train_classification", "params": {{"target": "heart_disease", "model_type": "random_forest"}}}}
]

Query: "what's the correlation between age and salary"
Output:
[
  {{"task": "get_pairwise_correlation", "params": {{"col1": "age", "col2": "salary"}}}}
]

Query: "summarize the data"
Output:
[
  {{"task": "summary_statistics", "params": {{}}}}
]

Query: "what columns do we have"
Output:
[
  {{"task": "get_columns", "params": {{}}}}
]

Query: "calculate statistics for age"
Output:
[
  {{"task": "calculate_column_stats", "params": {{"column_name": "age"}}}}
]

Query: "what are the data types"
Output:
[
  {{"task": "get_column_data_types", "params": {{}}}}
]

Now process the user query above.
        """

        try:
            raw_text = self._call_llm(prompt)
        except Exception as e:
            return [{"task": "error", "result": f"❌ LLM error: {e}"}]

        json_text = re.sub(r"```(json)?", "", raw_text).replace("```", "").strip()

        try:
            task_list = json.loads(json_text)
        except json.JSONDecodeError as e:
            return [{"task": "error", "result": f"❌ JSON parse error: {e}\nRaw: {raw_text}"}]

        if isinstance(task_list, dict):
            task_list = [task_list]

        return self._execute_task_list(task_list, query)

    def _execute_task_list(self, task_list: List[Dict], original_query: str) -> List[Dict]:
        """Execute list of tasks sequentially."""
        results = []

        ML_TASKS = {"train_classification", "train_regression"}

        for task_info in task_list:
            task_name = task_info.get('task')
            params = task_info.get('params', {})

            if not task_name:
                results.append({"task": "error", "result": "⚠️ Invalid task in plan"})
                continue

            if task_name == "respond_text":
                results.append({
                    "task": "response",
                    "result": params.get('text', 'No response generated')
                })
                continue

            result = self._execute_capability(task_name, params, original_query)

            if isinstance(result, pd.DataFrame):
                transformation_tasks = [
                    'handle_missing', 'handle_outliers', 'standardize_columns',
                    'remove_duplicates', 'convert_types', 'encode_categoricals',
                    'scale_features', 'normalize_features', 'apply_tfidf',
                    'extract_datetime_features', 'create_interaction_features',
                    'create_polynomial_features', 'bin_numeric_column'
                ]

                if task_name in transformation_tasks:
                    self.data = result
                    self.set_data(self.data)
                    result = f"✅ Data transformation complete. Dataset updated."

            results.append({"task": task_name, "result": result})

            # After ML training: run Reflection Loop, then show final critic evaluation
            if task_name in ML_TASKS and isinstance(result, dict):
                feature_agent = self.agents.get('feature_engineering')
                ml_agent      = self.agents.get('ml')
                critic        = self.agents.get('critic')

                if ml_agent and critic and feature_agent:
                    task_type = "regression" if task_name == "train_regression" else "classification"
                    target    = params.get("target", "")

                    engine = ReflectionEngine(ml_agent, critic, feature_agent)
                    reflection_output = engine.run(result, target=target, task_type=task_type)
                    results.append({"task": "reflection_loop", "result": reflection_output})

                    # Final critic evaluation on best result
                    best = reflection_output.get("final_result", result)
                    critique = critic.execute_capability(
                        "evaluate_model_performance",
                        accuracy    = best.get("accuracy"),
                        precision   = best.get("precision"),
                        recall      = best.get("recall"),
                        f1_score    = best.get("f1_score"),
                        train_score = best.get("train_score"),
                        test_score  = best.get("accuracy") or best.get("test_score"),
                        model_type  = best.get("model", "unknown")
                    )
                    results.append({"task": "critic_evaluation", "result": critique})

            if isinstance(result, str) and result.startswith("❌"):
                break

        # Auto-generate report after any ML workflow
        has_ml = any(r.get("task") in ("train_classification", "train_regression") for r in results)
        if has_ml:
            report_agent: ReportAgent = self.agents.get('report')
            if report_agent:
                dataset_name = getattr(self, '_dataset_name', 'Dataset')
                rows = self.data.shape[0] if self.data is not None else 0
                cols = self.data.shape[1] if self.data is not None else 0
                report_agent.collect_results(results)
                report_data = report_agent.build_report(
                    dataset_name=dataset_name,
                    objective=original_query,
                    rows=rows,
                    columns=cols
                )
                results.append({"task": "generated_report", "result": report_data})

        return results

    def _execute_capability(self, func_name: str, params: Dict, original_query: str) -> Any:
        """Dynamically execute a capability by name."""
        if func_name not in self.capability_map:
            return self._generate_fallback_response(func_name, original_query)

        cap_info = self.capability_map[func_name]
        agent_instance = cap_info['agent_instance']

        try:
            result = agent_instance.execute_capability(func_name, **params)
            return result
        except Exception as e:
            return f"❌ Error executing {func_name}: {str(e)}"

    def _generate_fallback_response(self, unknown_task: str, original_query: str) -> str:
        """Generate helpful fallback when capability not found."""
        fallback_prompt = f"""
The user asked: "{original_query}"
The system tried to execute: "{unknown_task}" but this capability doesn't exist.

Available capabilities include: data cleaning, EDA, visualization, feature engineering, and ML training.

Provide a short, helpful response explaining what went wrong and what the user can try instead.
Output ONLY the text response, no JSON.
        """

        try:
            return self._call_llm(fallback_prompt)
        except:
            return f"⚠️ Unknown task '{unknown_task}'. Please rephrase your query."

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "data_loaded": self.data is not None,
            "num_agents": len(self.agents),
            "num_capabilities": len(self.capability_map),
            "available_functions": list(self.capability_map.keys())
        }
