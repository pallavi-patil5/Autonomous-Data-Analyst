# 🤖 Autonomous Data Science AI

An intelligent, zero-hardcoded-rules data science assistant powered by **Groq LLaMA 3.3 70B** and an autonomous multi-agent architecture. The system dynamically discovers capabilities and executes complex data science workflows through natural language queries — including automatic model evaluation, iterative self-improvement, and comprehensive report generation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3.3_70B-orange.svg)](https://console.groq.com)

---

## ✨ Key Features

- **Zero Hardcoded Rules** — Fully autonomous agent system with dynamic capability discovery
- **Natural Language Interface** — Describe what you want in plain English
- **Universal Dataset Support** — Works with any CSV or Excel file
- **Complete Data Science Pipeline** — Cleaning, EDA, Visualization, Feature Engineering, and ML
- **Multi-Model ML Support** — 10+ algorithms: Random Forest, SVM, KNN, Logistic Regression, Gradient Boosting, Naive Bayes, Ridge, Lasso, and more
- **Intelligent Query Planning** — LLM automatically handles task dependencies and preprocessing
- **Critic Agent** — Evaluates model performance and detects overfitting, underfitting, and class imbalance
- **Reflection Loop** — Iteratively improves models by applying critic recommendations (max 2 cycles)
- **Report Agent** — Generates comprehensive 9-section reports exportable as PDF, DOCX, and JSON
- **Smart Suggestions** — Context-aware query recommendations based on dataset columns
- **Interactive Streamlit UI** — Clean, responsive web interface

---

## 🏗️ Architecture

```
User Query
    ↓
SmartOrchestratorV2 (Brain)
    ├── DataLoaderAgent          — Load CSV/Excel files
    ├── DataCleaningAgent        — Handle missing values, outliers, duplicates
    ├── EDAAgent                 — Statistical summaries, correlations
    ├── VisualizationAgent       — Histograms, boxplots, heatmaps, scatter plots
    ├── FeatureEngineeringAgent  — Encoding, scaling, feature selection
    ├── MLAgent                  — Train classification/regression/clustering models
    ├── CriticAgent              — Evaluate model metrics, detect issues
    ├── ReflectionEngine         — Iterative self-improvement loop (max 2 cycles)
    └── ReportAgent              — Generate PDF / DOCX / JSON reports
```

### ML Workflow (Auto-triggered after every training query)

```
Train Model  →  CriticAgent evaluates
             →  ReflectionEngine applies improvements & retrains
             →  CriticAgent re-evaluates best result
             →  ReportAgent generates full report
             →  Dashboard shows all results + download buttons
```

Each agent inherits from `BaseAgent` and self-describes its capabilities, enabling the orchestrator to dynamically discover and execute tasks without hardcoded rules.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key — free at [https://console.groq.com](https://console.groq.com)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Autonomous Data Analyst"
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Run the application:
```bash
venv\Scripts\streamlit.exe run enhanced_app.py
```

4. Open your browser at `http://localhost:8501`

5. Enter your **Groq API key** in the sidebar

6. Upload a dataset and start querying!

---

## 📊 Usage Examples

### Data Exploration
- "Give me a complete summary of the dataset"
- "Show me data types and missing value counts"
- "What columns do we have?"
- "Show the top 10 rows"

### Visualization
- "Plot the distribution of age"
- "Show boxplot for detecting outliers in salary"
- "Generate a correlation heatmap"
- "Plot age vs income with scatter plot"

### Machine Learning
- "Predict heart_disease using random forest"
- "Train logistic regression to predict diabetes"
- "Compare performance of RF, LR, SVM, and KNN for predicting target"
- "Find the most important features"

### Data Cleaning & Engineering
- "Handle missing values"
- "Remove outliers from age column"
- "Encode categorical variables"
- "Scale features for machine learning"

---

## 📊 Dashboard Output After ML Query

Every ML training query automatically produces 4 result cards:

| Card | What it shows |
|------|--------------|
| Model Result | Accuracy, classification report, confusion matrix |
| Reflection Loop | Improvement cycles, best accuracy, gains achieved |
| Critic Evaluation | Issues detected, recommendations, severity level |
| Generated Report | Full 9-section report with PDF/DOCX/JSON download |

---

## 📁 Project Structure

```
Autonomous Data Analyst/
├── base_agent.py                          # Base agent protocol (all agents inherit this)
├── smart_orchestrator_v2.py               # Main orchestrator — brain of the system
├── enhanced_app.py                        # Streamlit UI
├── enhanced_data_loader_agent.py          # Data loading capabilities
├── enhanced_data_cleaning_agent.py        # Data cleaning operations
├── enhanced_eda_agent.py                  # Exploratory data analysis
├── enhanced_data_visualization_agent.py   # Visualization generation
├── enhanced_feature_engineering_agent.py  # Feature engineering
├── enhanced_ml_agent.py                   # Machine learning models
├── critic_agent.py                        # Model evaluation & issue detection
├── reflection_engine.py                   # Iterative self-improvement loop
├── report_agent.py                        # PDF / DOCX / JSON report generation
├── __init__.py                            # Package initializer
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup
├── .env.example                           # Environment variable template
├── .gitignore                             # Git ignore rules
├── LICENSE                                # MIT License
├── README.md                              # This file
├── CHANGELOG.md                           # Version history
├── CONTRIBUTING.md                        # Contribution guidelines
├── SECURITY.md                            # Security policies
└── API_DOCUMENTATION.md                   # API reference
```

---

## 🧠 How It Works

1. **Capability Discovery** — Each agent self-describes its capabilities (function names, descriptions, parameters, examples)
2. **Query Processing** — User query is sent to Groq LLaMA along with available capabilities and dataset context
3. **Intelligent Planning** — LLM generates an execution plan with proper task sequencing and dependencies
4. **Dynamic Execution** — Orchestrator executes tasks by dynamically calling agent methods
5. **Critic Evaluation** — CriticAgent evaluates ML results and flags issues
6. **Reflection Loop** — Automatically applies improvements and retrains (up to 2 cycles, stops if gain < 1%)
7. **Report Generation** — ReportAgent collects all results and builds a 9-section structured report
8. **Result Presentation** — All results displayed in Streamlit with download buttons

---

## 🔁 Critic Agent

Automatically evaluates every trained model and detects:
- Low accuracy (< 60%) or moderate accuracy (< 75%)
- Overfitting (train score − test score > 15%)
- Underfitting (both train and test scores < 65%)
- Class imbalance (precision vs recall gap > 15%)
- Low F1-score (< 65%)

Returns structured output:
```json
{
  "issues_detected": ["Moderate accuracy: 0.72 — room for improvement"],
  "recommendations": ["Try Random Forest", "Apply feature scaling"],
  "severity": "medium"
}
```

---

## 🔄 Reflection Loop

Reads CriticAgent recommendations and automatically applies improvements:
- Feature scaling (StandardScaler / MinMaxScaler)
- Feature selection (SelectKBest)
- Model switching (Random Forest, Gradient Boosting)
- Class balancing (`class_weight='balanced'`)
- Hyperparameter tuning hints

Stopping criteria:
- Maximum 2 cycles
- Improvement gain < 1%
- Accuracy ≥ 90%

---

## 📄 Report Agent

Generates comprehensive 9-section reports:

1. Executive Summary
2. Data Quality Report
3. Exploratory Data Analysis
4. Visualizations
5. Machine Learning Results
6. Reflection Analysis
7. Feature Importance
8. Recommendations
9. Conclusion

Export formats: **PDF**, **DOCX**, **JSON**

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Groq LLaMA 3.3 70B | Natural language understanding and task planning |
| Streamlit ≥ 1.31 | Interactive web interface |
| Pandas ≥ 2.1 | Data manipulation |
| Scikit-learn ≥ 1.4 | Machine learning algorithms |
| Matplotlib / Seaborn | Data visualization |
| NumPy / SciPy | Numerical computing |
| python-docx | Word document generation |
| ReportLab | PDF generation |
| openpyxl | Excel file support |

---

## 🔒 Security Notes

- API keys are handled securely through Streamlit's password input and are never stored on disk
- No dataset is sent to external servers — all data processing happens locally
- Only the natural language query and column metadata are sent to the Groq API for planning
- See [SECURITY.md](SECURITY.md) for full security policies

---

## 🤝 Contributing

Contributions are welcome! To add new capabilities:

1. Create a new agent class that inherits from `BaseAgent`
2. Implement `get_capabilities()` to self-describe your agent's functions
3. Implement `execute_capability()` as the dispatcher
4. Register your agent in `SmartOrchestratorV2._initialize_agents()`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Groq](https://console.groq.com) for blazing-fast LLaMA inference
- [Streamlit](https://streamlit.io) for the excellent web framework
- The open-source data science community

---

*Built with ❤️ using Autonomous AI Agents | Powered by Groq & Streamlit*
