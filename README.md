# 🤖 Autonomous Data Science AI

An intelligent, zero-hardcoded-rules data science assistant powered by Google Gemini AI and autonomous agent architecture. This system dynamically discovers capabilities and executes complex data science workflows through natural language queries.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)

## ✨ Key Features

- **Zero Hardcoded Rules** – Fully autonomous agent system with dynamic capability discovery
- **Natural Language Interface** – Describe what you want in plain English
- **Universal Dataset Support** – Works with any CSV or Excel file
- **Complete Data Science Pipeline** – Cleaning, EDA, Visualization, Feature Engineering, and ML
- **Multi-Model ML Support** – 10+ machine learning algorithms (RF, SVM, KNN, Logistic Regression, etc.)
- **Intelligent Query Planning** – Automatically handles task dependencies and preprocessing
- **Smart Suggestions** – Context-aware query recommendations based on your dataset
- **Interactive Streamlit UI** – Beautiful, responsive web interface

## 🏗️ Architecture

The system uses a modular agent-based architecture:

```
SmartOrchestratorV2 (Brain)
    ├── DataLoaderAgent
    ├── DataCleaningAgent
    ├── EDAAgent
    ├── VisualizationAgent
    ├── FeatureEngineeringAgent
    └── MLAgent
```

Each agent inherits from `BaseAgent` and self-describes its capabilities, enabling the orchestrator to dynamically discover and execute tasks without hardcoded rules.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Autonomous Data Analyst"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

4. Run the application:
```bash
streamlit run enhanced_app.py
```

5. Open your browser at `http://localhost:8501`

6. Enter your Gemini API key in the sidebar

7. Upload a dataset and start querying!

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

## 📁 Project Structure

```
Autonomous Data Analyst/
├── base_agent.py                          # Base agent protocol
├── smart_orchestrator_v2.py               # Main orchestrator (brain)
├── enhanced_app.py                        # Streamlit UI
├── enhanced_data_loader_agent.py          # Data loading capabilities
├── enhanced_data_cleaning_agent.py        # Data cleaning operations
├── enhanced_eda_agent.py                  # Exploratory data analysis
├── enhanced_data_visualization_agent.py   # Visualization generation
├── enhanced_feature_engineering_agent.py  # Feature engineering
├── enhanced_ml_agent.py                   # Machine learning models
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup
├── README.md                              # This file
├── CONTRIBUTING.md                        # Contribution guidelines
├── CHANGELOG.md                           # Version history
├── SECURITY.md                            # Security policies
├── API_DOCUMENTATION.md                   # API reference
└── LICENSE                                # MIT License
```

## 🧠 How It Works

1. **Capability Discovery**: Each agent self-describes its capabilities with function names, descriptions, parameters, and examples
2. **Query Processing**: User query is sent to Gemini AI along with available capabilities and dataset context
3. **Intelligent Planning**: LLM generates an execution plan with proper task sequencing and dependencies
4. **Dynamic Execution**: Orchestrator executes tasks by dynamically calling agent methods
5. **Result Presentation**: Results are formatted and displayed in the Streamlit interface

## 🛠️ Technologies Used

- **Google Gemini AI** – Natural language understanding and planning
- **Streamlit** – Interactive web interface
- **Pandas** – Data manipulation
- **Scikit-learn** – Machine learning algorithms
- **Matplotlib & Seaborn** – Data visualization
- **NumPy & SciPy** – Numerical computing

## 📚 Documentation

- [API Documentation](API_DOCUMENTATION.md) – Detailed API reference
- [Contributing Guide](CONTRIBUTING.md) – How to contribute
- [Changelog](CHANGELOG.md) – Version history
- [Security Policy](SECURITY.md) – Security guidelines

## 🔒 Security Notes

- API keys are handled securely through Streamlit's password input
- No data is sent to external servers except Gemini API for query processing
- All data processing happens locally
- See [SECURITY.md](SECURITY.md) for more details

## 🤝 Contributing

Contributions are welcome! To add new capabilities:

1. Create a new agent class inheriting from `BaseAgent`
2. Implement `get_capabilities()` method
3. Add capability methods with clear docstrings
4. Register the agent in `SmartOrchestratorV2`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI for natural language processing
- Streamlit for the amazing web framework
- The open-source data science community

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review the API documentation

---

**Built with ❤️ using Autonomous AI Agents**
