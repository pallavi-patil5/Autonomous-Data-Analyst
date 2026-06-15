# API Documentation

## SmartOrchestratorV2

The main orchestrator that coordinates all agents and processes user queries.

### Initialization

```python
from smart_orchestrator_v2 import SmartOrchestratorV2

orchestrator = SmartOrchestratorV2(api_key="your_gemini_api_key")
```

### Methods

#### `set_data(df: pd.DataFrame)`
Load a dataset into the system.

**Parameters:**
- `df`: Pandas DataFrame containing the dataset

**Example:**
```python
import pandas as pd
df = pd.read_csv("data.csv")
orchestrator.set_data(df)
```

#### `process_query(query: str)`
Process a natural language query.

**Parameters:**
- `query`: Natural language query string

**Returns:**
- List of dictionaries containing task results

**Example:**
```python
results = orchestrator.process_query("Show distribution of age")
```

#### `get_system_status()`
Get current system status and available capabilities.

**Returns:**
- Dictionary with system information

**Example:**
```python
status = orchestrator.get_system_status()
print(f"Agents: {status['num_agents']}")
print(f"Capabilities: {status['num_capabilities']}")
```

---

## BaseAgent

Abstract base class for all agents.

### Creating Custom Agents

```python
from base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, data=None):
        super().__init__()
        self.data = data
    
    def get_capabilities(self):
        return [
            {
                "function_name": "custom_function",
                "description": "Description of what this does",
                "parameters": ["param1", "param2"],
                "examples": ["Example query"]
            }
        ]
    
    def custom_function(self, param1, param2):
        # Implementation
        return result
```

### Methods

#### `get_capabilities()`
Returns list of agent capabilities (must be implemented).

#### `execute_capability(function_name: str, **kwargs)`
Safely execute a capability by name.

---

## Available Agents

### DataLoaderAgent
- `load_csv(file_path)`
- `load_excel(file_path)`
- `get_data_info()`

### DataCleaningAgent
- `handle_missing(strategy, columns)`
- `handle_outliers(method, columns)`
- `remove_duplicates()`
- `standardize_columns()`

### EDAAgent
- `summary_statistics()`
- `get_columns()`
- `get_column_data_types()`
- `calculate_column_stats(column_name)`
- `get_pairwise_correlation(col1, col2)`

### VisualizationAgent
- `plot_histogram(column, bins)`
- `plot_boxplot(column)`
- `plot_scatter(x_col, y_col)`
- `plot_correlation_heatmap()`
- `plot_bar_chart(column)`

### FeatureEngineeringAgent
- `encode_categoricals(columns, method)`
- `scale_features(columns, method)`
- `normalize_features(columns)`
- `create_polynomial_features(columns, degree)`
- `bin_numeric_column(column, bins)`

### MLAgent
- `train_classification(target, model_type, test_size)`
- `train_regression(target, model_type, test_size)`
- `get_feature_importance(target, model_type)`
- `compare_models(target, model_types)`

---

## Query Examples

### Data Exploration
```python
orchestrator.process_query("Give me a complete summary of the dataset")
orchestrator.process_query("Show me the first 10 rows")
orchestrator.process_query("What are the data types?")
```

### Visualization
```python
orchestrator.process_query("Plot distribution of age")
orchestrator.process_query("Show correlation heatmap")
orchestrator.process_query("Create scatter plot of age vs salary")
```

### Machine Learning
```python
orchestrator.process_query("Predict target using random forest")
orchestrator.process_query("Compare RF, LR, and SVM models")
orchestrator.process_query("Show feature importance")
```

### Data Cleaning
```python
orchestrator.process_query("Handle missing values")
orchestrator.process_query("Remove outliers from age column")
orchestrator.process_query("Encode categorical variables")
```
