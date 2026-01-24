"""
Enhanced Streamlit App - Autonomous Data Science AI
(Integrated with Smart Query Suggestions)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from smart_orchestrator_v2 import SmartOrchestratorV2


# ==============================================================
# 🔹 Smart Query Suggestions Module
# ==============================================================

def generate_smart_suggestions(df):
    """Generate highly intelligent, impressive queries based on dataset."""
    
    suggestions = []

    columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ================================================================
    # 1. Dataset Understanding
    # ================================================================
    suggestions.append({
        'category': '📌 Dataset Understanding',
        'queries': [
            "Give me a complete summary of the dataset",
            "Show me data types and missing value counts",
            "which column has the highest importance",
            "show the top 10 rows of the dataset",
            "provide me the total mean of age column",
        ]
    })

    # ================================================================
    # 2. Visualization Suggestions (Dynamic)
    # ================================================================
    viz_queries = []

    if len(numeric_cols) > 0:
        viz_queries.append(f"Plot the distribution of {numeric_cols[0]}")
        viz_queries.append(f"Show boxplot for detecting outliers in {numeric_cols[0]}")
        if len(numeric_cols) > 1:
            viz_queries.append(f"Plot {numeric_cols[0]} vs {numeric_cols[1]} with scatter plot")
            viz_queries.append(f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        viz_queries.append("Generate a correlation heatmap of all numeric columns")

    
    suggestions.append({
        'category': '📈 Visual Analytics',
        'queries': viz_queries
    })

   
    
    # ================================================================
    # 3. Machine Learning — Dynamic Target Guess
    # ================================================================
    target_col = None
    keywords = ["target", "label", "class", "diagnosis", "default", "outcome", "result"]

    for col in columns:
        if any(k in col.lower() for k in keywords):
            target_col = col
            break

    if target_col is None:
        target_col = columns[-1]  # fallback automatic target

    ml_queries = [
        f"Predict {target_col} using random forest",
        f"Train logistic regression to predict {target_col}",
        f"Use SVM model to classify {target_col}",
        f"predict performance of RF, LR, SVM, and KNN for predicting {target_col}",
        f"Find the most important features contributing to {target_col}"
    ]

    suggestions.append({
        'category': '🤖 Machine Learning',
        'queries': ml_queries
    })


    return suggestions



def display_suggestions(suggestions):
    """Display suggestions with clickable buttons."""
    st.markdown("### 💡 Suggested Queries for Your Dataset")
    st.markdown("*Click any suggestion to use it, or write your own query below*")

    for suggestion_group in suggestions:
        with st.expander(f"{suggestion_group['category']}", expanded=True):
            queries = suggestion_group['queries']
            for i in range(0, len(queries), 2):
                cols = st.columns(2)
                with cols[0]:
                    if i < len(queries):
                        if st.button(f"💬 {queries[i]}", key=f"suggest_{suggestion_group['category']}_{i}", use_container_width=True):
                            st.session_state.selected_query = queries[i]
                            st.rerun()
                with cols[1]:
                    if i + 1 < len(queries):
                        if st.button(f"💬 {queries[i+1]}", key=f"suggest_{suggestion_group['category']}_{i+1}", use_container_width=True):
                            st.session_state.selected_query = queries[i+1]
                            st.rerun()


# ==============================================================
# 🔹 Page Config and CSS
# ==============================================================

st.set_page_config(page_title="Autonomous Data Science AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# 🔹 Header and Sidebar
# ==============================================================

st.markdown('<p class="main-header">🤖 Autonomous Data Science AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI Data Scientist - No Rules, Just Intelligence</p>', unsafe_allow_html=True)

# Initialize session state
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""


with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key:", type="password", help="Enter your Google Gemini API key")

    if api_key:
        st.success("✅ API Key Set")

    st.markdown("---")
    st.header("📊 System Info")

    if 'orchestrator' in st.session_state and api_key:
        status = st.session_state.orchestrator.get_system_status()
        st.metric("Agents Active", status['num_agents'])
        st.metric("Capabilities", status['num_capabilities'])

        with st.expander("View All Capabilities"):
            for func in status['available_functions']:
                st.text(f"• {func}")


# ==============================================================
# 🔹 Main App Logic
# ==============================================================

if api_key:
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = SmartOrchestratorV2(api_key=api_key)
        st.session_state.conversation_history = []

    orchestrator = st.session_state.orchestrator

    st.header("📁 Upload Your Dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"],
                                         help="Upload your dataset to begin analysis")
    with col2:
        st.markdown("### Quick Tips")
        st.info("💡 Try natural language queries like:\n- Show distribution of age\n- Predict disease with RF\n- What's the correlation?")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            orchestrator.set_data(df)
            st.success(f"✅ Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            with st.expander("📋 Preview Dataset", expanded=False):
                col_preview1, col_preview2 = st.columns(2)
                with col_preview1:
                    st.dataframe(df.head(10), use_container_width=True)
                with col_preview2:
                    st.markdown("**Column Info:**")
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        null_count = df[col].isnull().sum()
                        st.text(f"• {col}: {dtype} ({null_count} nulls)")

            st.markdown("---")
            suggestions = generate_smart_suggestions(df)
            display_suggestions(suggestions)
            st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            df = None

    # Chat Interface
    st.markdown("---")
    st.header("💬 Ask Your AI Data Scientist")

    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        for i, exchange in enumerate(st.session_state.conversation_history):
            with st.container():
                st.markdown(f"**🧑 You:** {exchange['query']}")
                st.markdown(f"**🤖 AI:** Executed {len(exchange['results'])} task(s)")
                if st.button(f"Show Details", key=f"detail_{i}"):
                    for result in exchange['results']:
                        st.json(result)
                st.markdown("---")

    default_query = st.session_state.get('selected_query', '')
    query = st.text_area(
        "Enter your analysis request:",
        value=default_query,
        height=100,
        help="Click a suggestion above or write your own query"
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)

    if clear_button:
        st.session_state.conversation_history = []
        st.rerun()

    if run_button and query:
        st.session_state.selected_query = ""
        if orchestrator.data is None:
            st.warning("⚠️ Please upload a dataset first!")
        else:
            with st.spinner(' AI is thinking and planning execution...'):
                results = orchestrator.process_query(query)
            st.session_state.conversation_history.append({'query': query, 'results': results})

            st.markdown("---")
            st.header("📊 Results")

            if isinstance(results, list):
                for idx, item in enumerate(results):
                    task_name = item.get('task', 'Unknown')
                    current_result = item.get('result')

                    with st.expander(f"**{idx + 1}. {task_name.replace('_', ' ').title()}**", expanded=True):

                        # Plot figures
                        if isinstance(current_result, plt.Figure):
                            st.pyplot(current_result)
                            plt.close('all')
                        elif hasattr(current_result, "gca"):
                            st.pyplot(current_result)
                            plt.close('all')

                        # DataFrame outputs
                        elif isinstance(current_result, pd.DataFrame):
                            st.dataframe(current_result, use_container_width=True)
                            csv = current_result.to_csv(index=False)
                            st.download_button("📥 Download CSV", csv, f"{task_name}_result.csv", "text/csv", key=f"download_{idx}")

                        # ML model outputs (dict)
                        elif isinstance(current_result, dict):
                            if 'accuracy' in current_result or 'rmse' in current_result:
                                st.markdown("### 🎯 Model Performance")
                                metric_cols = st.columns(3)
                                col_idx = 0
                                for key, value in current_result.items():
                                    if key in ['accuracy', 'rmse', 'mae', 'r2_score', 'mse']:
                                        with metric_cols[col_idx % 3]:
                                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                                        col_idx += 1
                                if 'classification_report' in current_result:
                                    st.markdown("### 📋 Classification Report")
                                    st.code(current_result['classification_report'], language='text')
                                if 'confusion_matrix' in current_result:
                                    st.markdown("### 🔢 Confusion Matrix")
                                    st.write(current_result['confusion_matrix'])
                                with st.expander("View Full Details"):
                                    st.json(current_result)
                            else:
                                for k, v in current_result.items():
                                    st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")

                        # String output
                        elif isinstance(current_result, str):
                            if current_result.startswith("❌"):
                                st.error(current_result)
                            elif current_result.startswith("✅"):
                                st.success(current_result)
                            elif current_result.startswith("⚠️"):
                                st.warning(current_result)
                            else:
                                st.write(current_result)

                        else:
                            st.write(current_result)

            else:
                st.write(results)

else:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to start.")
    st.markdown("""
    ### Features:
    - Zero Hardcoded Rules – Fully autonomous agent system  
    - Natural Language Queries – Just describe what you want  
    - Any Dataset – Works with any CSV or Excel file  
    - Complete Data Science Workflow – Cleaning, EDA, Visualization, ML  
    - Multi-Model Support – 10+ ML algorithms  
    - Intelligent Planning – Automatically handles dependencies
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🤖 Powered by Autonomous AI Agents | Built with Gemini & Streamlit</div>",
    unsafe_allow_html=True
)
