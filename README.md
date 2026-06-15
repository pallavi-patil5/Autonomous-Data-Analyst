# 🤖 Autonomous Data Science AI

An intelligent, zero-hardcoded-rules data science assistant powered by **Groq LLaMA AI** and autonomous agent architecture. This system dynamically discovers capabilities and executes complex data science workflows through natural language queries — including automatic model evaluation, iterative self-improvement, and comprehensive report generation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange.svg)](https://console.groq.com)

## ✨ Key Features

- **Zero Hardcoded Rules** – Fully autonomous agent system with dynamic capability discovery
- **Natural Language Interface** – Describe what you want in plain English
- **Universal Dataset Support** – Works with any CSV or Excel file
- **Complete Data Science Pipeline** – Cleaning, EDA, Visualization, Feature Engineering, and ML
- **Multi-Model ML Support** – 10+ machine learning algorithms (RF, SVM, KNN, Logistic Regression, etc.)
- **Intelligent Query Planning** – Automatically handles task dependencies and preprocessing
- **Critic Agent** – Automatically evaluates model performance and detects issues
- **Reflection Loop** – Iteratively improves models by applying recommendations automatically
- **Report Agent** – Generates comprehensive reports exportable as PDF, DOCX, and JSON
- **Smart Suggestions** – Context-aware query recommendations based on your dataset
- **Interactive Streamlit UI** – Beautiful, responsive web interface

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

2. Create and activate virtual environment:
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

## 📊 Dashboard Output After ML Query

Every ML training query automatically produces 4 result cards:

| Card | What it shows |
|------|--------------|
| Model Result | Accuracy, classification report, confusion matrix |
| Reflection Loop | Improvement cycles, best accuracy, gains achieved |
| Critic Evaluation | Issues detected, recommendations, severity level |
| Generated Report | Full 9-section report with PDF/DOCX/JSON download |

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

## 🧠 How It Works

1. **Capability Discovery** — Each agent self-describes its capabilities (function names, descriptions, parameters, examples)
2. **Query Processing** — User query is sent to Groq LLaMA along with available capabilities and dataset context
3. **Intelligent Planning** — LLM generates an execution plan with proper task sequencing and dependencies
4. **Dynamic Execution** — Orchestrator executes tasks by dynamically calling agent methods
5. **Critic Evaluation** — CriticAgent evaluates ML results and flags issues
6. **Reflection Loop** — Automatically applies improvements and retrains (up to 2 cycles, stops if gain < 1%)
7. **Report Generation** — ReportAgent collects all results and builds a 9-section structured report
8. **Result Presentation** — All results displayed in Streamlit with download buttons

## 🔁 Critic Agent

Automatically evaluates every trained model and detects:
- Low accuracy
- Overfitting (train score >> test score)
- Underfitting (both scores low)
- Class imbalance (precision vs recall gap)

Returns structured output:
```json
{
  "issues_detected": ["Moderate accuracy: 0.72"],
  "recommendations": ["Try Random Forest", "Apply feature scaling"],
  "severity": "medium"
}
```

## 🔄 Reflection Loop

Reads CriticAgent recommendations and automatically applies improvements:
- Feature scaling
- Feature selection (SelectKBest)
- Model switching (Random Forest, Gradient Boosting)
- Class balancing

Stopping criteria:
- Maximum 2 cycles
- Improvement < 1%
- Accuracy ≥ 90%

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

## 🛠️ Technologies Used

- **Groq LLaMA 3.3 70B** – Natural language understanding and planning
- **Streamlit** – Interactive web interface
- **Pandas** – Data manipulation
- **Scikit-learn** – Machine learning algorithms
- **Matplotlib & Seaborn** – Data visualization
- **NumPy & SciPy** – Numerical computing
- **python-docx** – Word document generation
- **ReportLab** – PDF generation

## 📚 Documentation

- [API Documentation](API_DOCUMENTATION.md) – Detailed API reference
- [Contributing Guide](CONTRIBUTING.md) – How to contribute
- [Changelog](CHANGELOG.md) – Version history
- [Security Policy](SECURITY.md) – Security guidelines

## 🔒 Security Notes

- API keys are handled securely through Streamlit's password input
- No data is sent to external servers except Groq API for query processing
- All data processing happens locally
- See [SECURITY.md](SECURITY.md) for more details

## 🤝 Contributing

To add new capabilities:

1. Create a new agent class inheriting from `BaseAgent`
2. Implement `get_capabilities()` method
3. Add capability methods with clear docstrings
4. Register the agent in `SmartOrchestratorV2`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Groq for the blazing fast LLaMA inference API
- Streamlit for the amazing web framework
- The open-source data science community

---

**Built with ❤️ using Autonomous AI Agents**
