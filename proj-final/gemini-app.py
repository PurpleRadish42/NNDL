import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection Intelligence Hub",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Polished Look ---
st.markdown("""
<style>
    /* General Styling */
    .main {
        background-color: #f0f2f6; /* Light mode background */
    }
    [data-theme="dark"] .main {
        background-color: #0e1117; /* Dark mode background */
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #667eea;
    }
    .section-header {
        font-size: 1.75rem;
        font-weight: bold;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
    }

    /* Metric Cards */
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #764ba2;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 16px rgba(0,0,0,0.08);
    }
    [data-theme="dark"] .metric-container {
        background-color: #1c2127;
        border: 1px solid #30363d;
        border-left: 5px solid #764ba2;
    }
    .metric-container h3 {
        font-size: 2rem;
        font-weight: 700;
        color: #764ba2;
    }
    .metric-container p {
        font-size: 1rem;
        color: #555;
    }
    [data-theme="dark"] .metric-container p {
        color: #adbac7;
    }

    /* Custom Buttons */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #667eea;
        background-color: transparent;
        color: #667eea;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #667eea;
        color: white;
    }

    /* Expander Styling */
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    [data-theme="dark"] .stExpander {
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# --- FT-Transformer Definition ---
# This class needs to be defined for the model to be loaded correctly.
class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data - Memory optimized"""
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(FTTransformer, self).__init__()
        self.feature_tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x_tokens = self.feature_tokenizer(x.unsqueeze(-1))
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x_tokens = torch.cat([cls_tokens, x_tokens], dim=1)
        x_transformed = self.transformer(x_tokens)
        return self.classifier(x_transformed[:, 0]).squeeze()


# --- Data and Model Loading with Caching ---
@st.cache_data
def load_dataset():
    if os.path.exists('creditcard.csv'):
        return pd.read_csv('creditcard.csv')
    return None

@st.cache_resource
def load_scaler():
    if os.path.exists('models/scaler.pkl'):
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_results_summary():
    if os.path.exists('results/model_summary.json'):
        with open('results/model_summary.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_all_models():
    models = {}
    # Load ML models
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl', 'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl', 'LightGBM': 'lightgbm.pkl', 'Gradient Boosting': 'gradient_boosting.pkl',
        'Extra Trees': 'extra_trees.pkl', 'SVM': 'svm.pkl', 'KNN': 'knn.pkl',
        'Decision Tree': 'decision_tree.pkl', 'Naive Bayes': 'naive_bayes.pkl'
    }
    for name, filename in model_files.items():
        path = f'models/{filename}'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    # Load TabNet
    tabnet_path = 'models/tabnet/tabnet_model.zip'
    if os.path.exists(tabnet_path):
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model(tabnet_path)
        models['TabNet'] = tabnet_model

    # Load FT-Transformer
    ft_path = 'models/ft_transformer.pth'
    if os.path.exists(ft_path):
        data = load_dataset()
        if data is not None:
            input_dim = data.shape[1] - 1
            ft_model = FTTransformer(input_dim=input_dim)
            ft_model.load_state_dict(torch.load(ft_path, map_location='cpu'))
            ft_model.eval()
            models['FT-Transformer'] = ft_model
    return models


# --- Main Application Class ---
class FraudDetectionDashboard:
    def __init__(self):
        self.data = load_dataset()
        self.scaler = load_scaler()
        self.results_summary = load_results_summary()
        self.models = load_all_models()

    def display_overview(self):
        st.markdown('<div class="section-header">🏠 Dataset Overview</div>', unsafe_allow_html=True)

        if self.data is None:
            st.error("❌ creditcard.csv not found. Please place the dataset in the root directory.")
            return

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        fraud_count = self.data['Class'].sum()
        total_count = len(self.data)
        fraud_rate = (fraud_count / total_count) * 100

        with col1:
            st.markdown(f'<div class="metric-container"><h3>{total_count:,}</h3><p>Total Transactions</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h3>{fraud_count:,}</h3><p>Fraudulent Cases</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-container"><h3>{fraud_rate:.3f}%</h3><p>Fraud Rate</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-container"><h3>${self.data["Amount"].sum():,.2f}</h3><p>Total Transaction Value</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Interactive Charts
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Transaction Amount Distribution")
            fig = px.histogram(self.data, x="Amount", color="Class", nbins=100,
                               log_y=True, title="Distribution of Transaction Amounts (Log Scale)",
                               labels={'Amount': 'Transaction Amount ($)', 'Class': 'Transaction Type'},
                               color_discrete_map={0: '#667eea', 1: '#ff6b6b'})
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Inferences")
            st.info("""
            * **Highly Imbalanced:** The dataset is extremely imbalanced, with fraudulent transactions making up a tiny fraction of the total. This is why accuracy is a poor metric for this problem.
            * **Amount Distribution:** The vast majority of transactions are small. Fraudulent transactions also tend to be smaller in value, possibly to avoid triggering standard security alerts.
            * **Log Scale:** The y-axis is on a logarithmic scale to make the rare fraud cases visible.
            """)

        with st.expander("Explore the Raw Data"):
            st.dataframe(self.data.sample(1000), use_container_width=True)

    def display_model_comparison(self):
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

        if not self.results_summary:
            st.warning("⚠️ Model results summary not found. Please run `models.py` to generate results.")
            return

        df = pd.DataFrame(self.results_summary['model_results']).T
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Model'}, inplace=True)

        metric_cols = ['auc_score', 'precision', 'recall', 'f1_score']
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.sort_values('auc_score', ascending=False, inplace=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AUC Score Comparison")
            fig = px.bar(df, x='auc_score', y='Model', orientation='h', color='type',
                         text='auc_score', title="Model AUC Scores (Higher is Better)",
                         color_discrete_map={'ML': '#667eea', 'DL': '#764ba2'}, template='plotly_white')
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Inferences")
            best_model = df.iloc[0]
            st.success(f"""
            **Best Performer: {best_model['Model']}**
            * **AUC Score:** {best_model['auc_score']:.4f}
            * **Type:** {best_model['type']}
            
            **Key Takeaways:**
            * Deep Learning models (FT-Transformer and TabNet) significantly outperform traditional Machine Learning methods on the AUC metric.
            * This suggests that their ability to learn complex, non-linear patterns is highly effective for this dataset.
            """)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precision, Recall & F1-Score Radar")
            metrics = ['precision', 'recall', 'f1_score']
            fig_radar = go.Figure()
            for model_type, color in zip(['ML', 'DL'], ['#667eea', '#764ba2']):
                subset = df[df['type'] == model_type].nlargest(3, 'auc_score')
                for _, row in subset.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row[metrics].values, theta=metrics, fill='toself' if model_type == 'DL' else 'none',
                        name=row['Model'], line_color=color
                    ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])), title="Top Models on Key Metrics", height=500)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.subheader("Inferences")
            st.info("""
            **The Precision-Recall Trade-off:**
            * **Recall** is crucial: We want to catch as many fraudulent transactions as possible (minimize false negatives).
            * **Precision** is also important: We don't want to incorrectly flag too many legitimate transactions (minimize false positives).
            * The **F1-Score** provides a balance between the two. The radar chart shows how the top models navigate this trade-off. Deep Learning models tend to show a more balanced profile.
            """)

    def display_individual_analysis(self):
        st.markdown('<div class="section-header">🔬 Individual Model Deep-Dive</div>', unsafe_allow_html=True)

        if not self.models:
            st.error("❌ No models loaded. Please run `models.py` to train and save models.")
            return

        selected_model = st.selectbox("Select a Model for Analysis", self.models.keys())

        if selected_model:
            model_info = self.results_summary['model_results'].get(selected_model, {})
            
            # Performance Metrics for Selected Model
            st.subheader(f"Performance Dashboard: {selected_model}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Type", model_info.get('type', 'N/A'))
            col2.metric("AUC Score", f"{model_info.get('auc_score', 0):.4f}")
            col3.metric("Precision (Fraud)", f"{model_info.get('precision', 0):.4f}")
            col4.metric("Recall (Fraud)", f"{model_info.get('recall', 0):.4f}")
            
            st.markdown("---")
            
            # Prediction Interface
            st.subheader("🎯 Real-Time Prediction")
            st.write("Enter transaction details below to get a fraud prediction from the selected model.")
            
            input_features = {}
            # Use an expander for the numerous PCA features
            with st.expander("Enter Transaction Features", expanded=True):
                c1, c2, c3 = st.columns(3)
                input_features['Time'] = c1.number_input("Time (seconds)", value=float(self.data['Time'].mean()))
                input_features['Amount'] = c2.number_input("Amount ($)", value=float(self.data['Amount'].mean()), min_value=0.0)
                
                pca_cols = [f'V{i}' for i in range(1, 29)]
                pca_c = st.columns(4)
                for i, col_name in enumerate(pca_cols):
                    input_features[col_name] = pca_c[i % 4].number_input(col_name, value=0.0)

            if st.button("Predict Fraud Risk", type="primary"):
                self.make_and_display_prediction(selected_model, input_features)

    def make_and_display_prediction(self, model_name, features_dict):
        # Ensure features are in the correct order
        ordered_features = [features_dict[col] for col in self.data.drop('Class', axis=1).columns]
        features_array = np.array(ordered_features).reshape(1, -1)
        
        # Scale if necessary
        if self.scaler:
            features_scaled = self.scaler.transform(features_array)
        else:
            st.warning("Scaler not found, using unscaled data.")
            features_scaled = features_array

        model = self.models[model_name]
        
        # Prediction logic for different model types
        try:
            if model_name == 'TabNet':
                prob = model.predict_proba(features_scaled)[0][1]
            elif model_name == 'FT-Transformer':
                with torch.no_grad():
                    prob = model(torch.FloatTensor(features_scaled)).item()
            else: # Standard sklearn models
                prob = model.predict_proba(features_scaled)[0][1]
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return

        # Display result
        st.subheader("Risk Assessment Result")
        if prob > 0.5:
            st.error(f"🚨 High Risk: Fraud Detected (Probability: {prob:.2%})")
            st.markdown("**Recommendation:** This transaction should be flagged for immediate review. The model has high confidence that this is a fraudulent transaction based on its learned patterns.")
        else:
            st.success(f"✅ Low Risk: Legitimate Transaction (Fraud Probability: {prob:.2%})")
            st.markdown("**Recommendation:** This transaction appears to be safe. The model's confidence in its legitimacy is high.")
        
        # Probability Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Fraud Probability Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#764ba2"},
                   'steps': [
                       {'range': [0, 50], 'color': 'lightgreen'},
                       {'range': [50, 100], 'color': 'lightcoral'}],
                   }))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    def display_about_page(self):
        st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
        st.info("""
        This interactive dashboard provides a comprehensive platform for analyzing and comparing various machine learning and deep learning models for credit card fraud detection.
        
        **Key Objectives:**
        - To demonstrate the effectiveness of modern Deep Learning architectures (TabNet, FT-Transformer) on tabular data.
        - To provide a clear, interactive comparison against a wide baseline of traditional ML models.
        - To build an intuitive tool for real-time risk assessment on new transactions.
        
        **Technologies Used:**
        - **Frontend:** Streamlit
        - **Data Processing:** Pandas, Scikit-learn
        - **Visualizations:** Plotly
        - **Machine Learning:** Scikit-learn, XGBoost, LightGBM
        - **Deep Learning:** PyTorch, Pytorch-TabNet
        """)


# --- Main App Execution ---
def main():
    st.markdown('<div class="main-header">💳 Fraud Detection Intelligence Hub</div>', unsafe_allow_html=True)
    
    dashboard = FraudDetectionDashboard()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Overview", "📊 Model Comparison", "🔬 Individual Analysis", "ℹ️ About"])

    st.sidebar.title("📈 System Status")
    if dashboard.data is not None:
        st.sidebar.success(f"Dataset loaded ({len(dashboard.data):,} rows)")
    else:
        st.sidebar.error("Dataset not found")
    
    if dashboard.models:
        st.sidebar.success(f"{len(dashboard.models)} models loaded")
    else:
        st.sidebar.warning("No models loaded")

    if page == "🏠 Overview":
        dashboard.display_overview()
    elif page == "📊 Model Comparison":
        dashboard.display_model_comparison()
    elif page == "🔬 Individual Analysis":
        dashboard.display_individual_analysis()
    elif page == "ℹ️ About":
        dashboard.display_about_page()

if __name__ == "__main__":
    main()
