import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from datetime import datetime
import zipfile
import tempfile
import shutil

warnings.filterwarnings('ignore')

# FT-Transformer class definition (same as in models.py)
class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data - Memory optimized"""
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(FTTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature tokenization
        self.feature_tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim + 1, d_model))
        
        # Transformer with reduced parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,  # Reduced from default 4x
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize features
        x_tokens = self.feature_tokenizer(x.unsqueeze(-1))  # (batch, features, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_tokens = torch.cat([cls_tokens, x_tokens], dim=1)
        
        # Add positional encoding
        x_tokens = x_tokens + self.positional_encoding
        
        # Transformer
        x_transformed = self.transformer(x_tokens)
        
        # Use CLS token for classification
        cls_output = x_transformed[:, 0]  # First token is CLS
        
        return self.classifier(cls_output).squeeze()

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode compatibility and fancy styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        border-left: 5px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 0 10px 10px 0;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .model-card {
        background: rgba(102, 126, 234, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-input {
        background: rgba(118, 75, 162, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .stSelectbox {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionDashboard:
    def __init__(self):
        self.data = None
        self.scaler = None
        self.models = {}
        self.results_summary = {}
        self.load_data_and_models()
    
    def load_data_and_models(self):
        """Load the dataset and trained models"""
        try:
            # Load dataset
            if os.path.exists('creditcard.csv'):
                self.data = pd.read_csv('creditcard.csv')
                st.success("✅ Dataset loaded successfully!")
            else:
                st.error("❌ creditcard.csv not found. Please ensure the file is in the correct directory.")
                return
            
            # Load scaler
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")
            else:
                st.warning("⚠️ Scaler not found. Please run models.py first.")
            
            # Load results summary
            if os.path.exists('results/model_summary.json'):
                with open('results/model_summary.json', 'r') as f:
                    self.results_summary = json.load(f)
                st.success("✅ Results summary loaded!")
            else:
                st.warning("⚠️ Results summary not found. Model comparison will be limited.")
            
            # Load ML models
            model_files = {
                'Logistic Regression': 'logistic_regression.pkl',
                'Random Forest': 'random_forest.pkl',
                'XGBoost': 'xgboost.pkl',
                'LightGBM': 'lightgbm.pkl',
                'Gradient Boosting': 'gradient_boosting.pkl',
                'Extra Trees': 'extra_trees.pkl',
                'SVM': 'svm.pkl',
                'KNN': 'knn.pkl',
                'Decision Tree': 'decision_tree.pkl',
                'Naive Bayes': 'naive_bayes.pkl'
            }
            
            loaded_models = 0
            for model_name, filename in model_files.items():
                filepath = f'models/{filename}'
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        loaded_models += 1
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {model_name}: {str(e)}")
                else:
                    st.warning(f"⚠️ {model_name} model file not found: {filepath}")
            
            if loaded_models > 0:
                st.success(f"✅ {loaded_models} ML models loaded successfully!")
            
            # Load TabNet
            tabnet_folder = 'models/tabnet'
            if os.path.exists(tabnet_folder):
                try:
                    from pytorch_tabnet.tab_model import TabNetClassifier
                    tabnet_model = TabNetClassifier()

                    # --- Load from the newly created ZIP file ---
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, 'models', 'tabnet', 'tabnet_model.zip')
                    
                    tabnet_model.load_model(model_path)
                    
                    self.models['TabNet'] = tabnet_model
                    st.success("✅ TabNet model loaded successfully!")

                except Exception as e:
                    st.warning(f"⚠️ Could not load TabNet model: {str(e)}")
            
            # Load FT-Transformer
            ft_transformer_path = 'models/ft_transformer.pth'
            if os.path.exists(ft_transformer_path):
                try:
                    # We need to recreate the model architecture
                    if hasattr(self, 'data') and self.data is not None:
                        input_dim = self.data.shape[1] - 1  # Exclude Class column
                        ft_model = FTTransformer(input_dim=input_dim)
                        ft_model.load_state_dict(torch.load(ft_transformer_path, map_location='cpu'))
                        ft_model.eval()
                        self.models['FT-Transformer'] = ft_model
                        st.success("✅ FT-Transformer model loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Could not load FT-Transformer model: {str(e)}")
            else:
                st.warning("⚠️ FT-Transformer model not found")
            
            # Summary of loaded components
            if len(self.models) == 0:
                st.error("❌ No models loaded! Please run 'python models.py' first to train the models.")
            else:
                st.info(f"📊 Summary: {len(self.models)} models loaded and ready for use!")
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.info("💡 Make sure to run 'python models.py' first to train and save the models.")
    
    def display_dataset_overview(self):
        """Display comprehensive dataset overview"""
        st.markdown('<div class="section-header">📊 Dataset Overview & Analysis</div>', 
                   unsafe_allow_html=True)
        
        if self.data is not None:
            # Basic dataset info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{self.data.shape[0]:,}</h3>
                    <p>Total Transactions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{self.data.shape[1]}</h3>
                    <p>Features</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                fraud_count = self.data['Class'].sum()
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{fraud_count:,}</h3>
                    <p>Fraud Cases</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                fraud_rate = (fraud_count / len(self.data)) * 100
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{fraud_rate:.2f}%</h3>
                    <p>Fraud Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Dataset preview
            st.subheader("📋 Dataset Preview")
            st.dataframe(self.data.head(10), use_container_width=True)
            
            # Dataset description
            st.subheader("📈 Statistical Summary")
            st.dataframe(self.data.describe(), use_container_width=True)
            
            # Dataset visualizations
            if os.path.exists('plots/dataset_overview.png'):
                st.subheader("📊 Dataset Visualizations")
                img = Image.open('plots/dataset_overview.png')
                st.image(img, caption="Comprehensive Dataset Analysis", use_container_width=True)
            
            # Detailed explanations
            st.subheader("🔍 Dataset Explanation")
            
            with st.expander("📖 About the Credit Card Fraud Detection Dataset", expanded=True):
                st.markdown("""
                ### Dataset Overview
                This dataset contains transactions made by credit cards in September 2013 by European cardholders. 
                It presents transactions that occurred in two days, where we have **492 frauds out of 284,807 transactions**. 
                
                ### Features Explanation
                - **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset
                - **V1-V28**: Principal components obtained with PCA (due to confidentiality issues, original features are not provided)
                - **Amount**: Transaction amount (this feature can be used for example-dependant cost-sensitive learning)
                - **Class**: Response variable (1 in case of fraud, 0 otherwise)
                
                ### Key Characteristics
                - **Highly Imbalanced**: Only 0.172% of transactions are fraudulent
                - **Anonymized Features**: V1-V28 are PCA-transformed features for privacy
                - **Real-world Data**: Actual credit card transactions from European cardholders
                - **Binary Classification**: Fraud (1) vs Normal (0) transaction classification
                
                ### Challenges
                - **Class Imbalance**: Traditional accuracy metrics are misleading
                - **Feature Interpretation**: PCA-transformed features are hard to interpret
                - **Cost-sensitive**: False negatives (missing fraud) are more costly than false positives
                - **Real-time Detection**: Models need to be fast for real-time deployment
                """)
            
            with st.expander("🎯 Project Objectives & Methodology"):
                st.markdown("""
                ### Project Goals
                1. **Compare Traditional ML vs Deep Learning** approaches for fraud detection
                2. **Evaluate Multiple Algorithms** across different paradigms
                3. **Build Interactive Dashboard** for model comparison and prediction
                4. **Analyze Performance Metrics** relevant to fraud detection
                
                ### Models Implemented
                
                **Traditional Machine Learning (10 models):**
                - Logistic Regression
                - Random Forest  
                - XGBoost
                - LightGBM
                - Gradient Boosting
                - Extra Trees
                - Support Vector Machine (SVM)
                - K-Nearest Neighbors (KNN)
                - Decision Tree
                - Naive Bayes
                
                **Deep Learning (2 models):**
                - **TabNet**: Attention-based neural network for tabular data
                - **FT-Transformer**: Feature Tokenizer + Transformer architecture
                
                ### Evaluation Strategy
                - **Primary Metric**: AUC-ROC (handles class imbalance well)
                - **Secondary Metrics**: Precision, Recall, F1-Score for fraud class
                - **Cross-validation**: Stratified to maintain class distribution
                - **Visualization**: ROC curves, confusion matrices, feature importance
                """)
        else:
            st.error("❌ Dataset not loaded. Please ensure 'creditcard.csv' is available.")
    
    def display_model_comparison(self):
        """Display overall model comparison using interactive Plotly charts"""
        st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>',
                   unsafe_allow_html=True)

        if not self.results_summary or 'model_results' not in self.results_summary:
            st.warning("⚠️ Model results not found. Please run 'python models.py' first.")
            return

        results = self.results_summary['model_results']
        comparison_df = pd.DataFrame(results).T.sort_values('auc_score', ascending=False)
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # --- Create a 2x2 layout for the charts ---
        col1, col2 = st.columns(2)

        with col1:
            # 1. Main Performance Bar Chart (AUC Score)
            st.subheader("📊 AUC Score Comparison")
            fig_bar = px.bar(
                comparison_df,
                x='auc_score',
                y='Model',
                orientation='h',
                color='type',
                text='auc_score',
                title="Model AUC Scores (Higher is Better)",
                color_discrete_map={'ML': '#667eea', 'DL': '#764ba2'},
                template='plotly_white'
            )
            fig_bar.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="AUC Score",
                yaxis_title="Model",
                height=500,
                legend_title_text='Model Type'
            )
            fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # 2. Detailed Metrics Radar Chart
            st.subheader("🎯 Multi-Metric Radar")
            metrics = ['precision', 'recall', 'f1_score']
            
            fig_radar = go.Figure()

            for model_type, color in zip(['ML', 'DL'], ['#667eea', '#764ba2']):
                df_subset = comparison_df[comparison_df['type'] == model_type].nlargest(3, 'auc_score')
                for i, row in df_subset.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row[metrics].values.tolist() + [row[metrics].values[0]], # Loop back to start
                        theta=metrics + [metrics[0]],
                        fill='toself' if model_type == 'DL' else 'none',
                        name=row['Model'],
                        line_color=color,
                        hovertemplate=f"<b>{row['Model']}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>"
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0.5, 1.0])
                ),
                title="Top Models on Precision, Recall & F1",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns([3, 2])

        with col3:
            # 3. Performance Summary Table
            st.subheader("📋 Performance Summary Table")
            display_df = comparison_df[['Model', 'type', 'auc_score', 'precision', 'recall', 'f1_score']]
            st.dataframe(
                display_df.style.background_gradient(cmap='viridis', subset=['auc_score'])\
                                .background_gradient(cmap='plasma', subset=['precision', 'recall', 'f1_score'])\
                                .format("{:.4f}", subset=['auc_score', 'precision', 'recall', 'f1_score']),
                use_container_width=True,
                hide_index=True
            )

        with col4:
            # 4. ML vs DL Performance Distribution
            st.subheader("🤖 ML vs. DL Distribution")
            fig_box = px.box(
                comparison_df,
                x='type',
                y='auc_score',
                color='type',
                title="AUC Score Distribution",
                points="all",
                notched=True,
                color_discrete_map={'ML': '#667eea', 'DL': '#764ba2'}
            )
            fig_box.update_layout(
                 xaxis_title="Model Type",
                 yaxis_title="AUC Score"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    
    def display_model_comparison(self):
        """Display overall model comparison using interactive Plotly charts"""
        st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>',
                   unsafe_allow_html=True)

        if not self.results_summary or 'model_results' not in self.results_summary:
            st.warning("⚠️ Model results not found. Please run 'python models.py' first.")
            return

        results = self.results_summary['model_results']
        comparison_df = pd.DataFrame(results).T
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # --- FIX: Convert metric columns to a numeric type ---
        # Data loaded from JSON can sometimes be treated as text (object).
        # We must explicitly convert metric columns to numbers (float) before plotting or sorting.
        metric_cols = ['auc_score', 'precision', 'recall', 'f1_score']
        for col in metric_cols:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
        # --- END OF FIX ---

        # Now we can safely sort the DataFrame
        comparison_df.sort_values('auc_score', ascending=False, inplace=True)

        # --- Create a 2x2 layout for the charts ---
        col1, col2 = st.columns(2)

        with col1:
            # 1. Main Performance Bar Chart (AUC Score)
            st.subheader("📊 AUC Score Comparison")
            fig_bar = px.bar(
                comparison_df,
                x='auc_score',
                y='Model',
                orientation='h',
                color='type',
                text='auc_score',
                title="Model AUC Scores (Higher is Better)",
                color_discrete_map={'ML': '#667eea', 'DL': '#764ba2'},
                template='plotly_white'
            )
            fig_bar.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="AUC Score",
                yaxis_title="Model",
                height=500,
                legend_title_text='Model Type'
            )
            fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # 2. Detailed Metrics Radar Chart
            st.subheader("🎯 Multi-Metric Radar")
            metrics_to_plot = ['precision', 'recall', 'f1_score']
            
            fig_radar = go.Figure()

            # Plot the top 3 ML models and top 2 DL models
            for model_type, color, n_largest in zip(['ML', 'DL'], ['#667eea', '#764ba2'], [3, 2]):
                df_subset = comparison_df[comparison_df['type'] == model_type].nlargest(n_largest, 'auc_score')
                for i, row in df_subset.iterrows():
                    # Create a closed loop for the radar chart
                    r_values = row[metrics_to_plot].values.tolist()
                    r_values.append(r_values[0])
                    theta_values = metrics_to_plot + [metrics_to_plot[0]]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=theta_values,
                        fill='toself' if model_type == 'DL' else 'none',
                        name=row['Model'],
                        line_color=color,
                        hovertemplate=f"<b>{row['Model']}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>"
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0.5, 1.0])
                ),
                title="Top Models on Precision, Recall & F1",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns([3, 2])

        with col3:
            # 3. Performance Summary Table
            st.subheader("📋 Performance Summary Table")
            display_df = comparison_df[['Model', 'type', 'auc_score', 'precision', 'recall', 'f1_score']]
            st.dataframe(
                display_df.style.background_gradient(cmap='viridis', subset=['auc_score'])\
                                .background_gradient(cmap='plasma', subset=['precision', 'recall', 'f1_score'])\
                                .format("{:.4f}", subset=['auc_score', 'precision', 'recall', 'f1_score']),
                use_container_width=True,
                hide_index=True
            )

        with col4:
            # 4. ML vs. DL Distribution
            st.subheader("🤖 ML vs. DL Distribution")
            fig_box = px.box(
                comparison_df,
                x='type',
                y='auc_score',
                color='type',
                title="AUC Score Distribution",
                points="all",
                notched=True,
                color_discrete_map={'ML': '#667eea', 'DL': '#764ba2'}
            )
            fig_box.update_layout(
                 xaxis_title="Model Type",
                 yaxis_title="AUC Score"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    def display_individual_model_analysis(self):
        """Display individual model analysis and prediction interface"""
        st.markdown('<div class="section-header">🔬 Individual Model Analysis & Prediction</div>', 
                   unsafe_allow_html=True)
        
        # Model selection
        available_models = list(self.models.keys())
        if 'TabNet' in available_models:
            available_models.append('TabNet')
        
        selected_model = st.selectbox(
            "🎯 Select Model for Detailed Analysis",
            available_models,
            index=0 if available_models else None
        )
        
        if selected_model and selected_model in available_models:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Model information
                st.subheader(f"📋 {selected_model} Information")
                
                if self.results_summary and selected_model in self.results_summary.get('model_results', {}):
                    model_info = self.results_summary['model_results'][selected_model]
                    
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>Performance Metrics</h4>
                        <p><strong>Type:</strong> {model_info['type']}</p>
                        <p><strong>AUC Score:</strong> {model_info['auc_score']:.4f}</p>
                        <p><strong>Precision:</strong> {model_info['precision']:.4f}</p>
                        <p><strong>Recall:</strong> {model_info['recall']:.4f}</p>
                        <p><strong>F1-Score:</strong> {model_info['f1_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display model-specific analysis plots
                plot_filename = f"plots/{selected_model.lower().replace(' ', '_').replace('-', '_')}_analysis.png"
                if os.path.exists(plot_filename):
                    img = Image.open(plot_filename)
                    st.image(img, caption=f"{selected_model} Detailed Analysis", use_container_width=True)
            
            with col2:
                # Prediction interface
                st.subheader("🎯 Make Predictions")
                
                if selected_model in self.models and self.scaler is not None:
                    st.markdown("""
                    <div class="prediction-input">
                        <p>Enter transaction details to predict fraud probability:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction input methods
                    prediction_method = st.radio(
                        "Choose input method:",
                        ["Manual Input", "Random Sample", "Upload CSV"]
                    )
                    
                    if prediction_method == "Manual Input":
                        self.manual_prediction_input(selected_model)
                    elif prediction_method == "Random Sample":
                        self.random_sample_prediction(selected_model)
                    elif prediction_method == "Upload CSV":
                        self.batch_prediction_interface(selected_model)
                
                else:
                    st.warning("⚠️ Model or scaler not available for predictions.")
    
    def manual_prediction_input(self, model_name):
        """Manual input interface for predictions"""
        st.subheader("✏️ Manual Input")
        
        # Create input fields for key features
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time", value=0.0, help="Time elapsed since first transaction")
            amount = st.number_input("Amount", value=0.0, min_value=0.0, help="Transaction amount")
            v1 = st.number_input("V1", value=0.0, help="PCA feature 1")
            v2 = st.number_input("V2", value=0.0, help="PCA feature 2")
            v3 = st.number_input("V3", value=0.0, help="PCA feature 3")
            
        with col2:
            v4 = st.number_input("V4", value=0.0, help="PCA feature 4")
            v5 = st.number_input("V5", value=0.0, help="PCA feature 5")
            v6 = st.number_input("V6", value=0.0, help="PCA feature 6")
            v7 = st.number_input("V7", value=0.0, help="PCA feature 7")
            v8 = st.number_input("V8", value=0.0, help="PCA feature 8")
        
        # Additional features (V9-V28) with expander
        with st.expander("Additional Features (V9-V28)"):
            additional_features = {}
            cols = st.columns(4)
            for i, col in enumerate(cols * 5):  # 20 more features
                if i < 20:
                    with col:
                        additional_features[f'V{i+9}'] = st.number_input(f"V{i+9}", value=0.0, key=f"v{i+9}")
        
        # Combine all features
        features = [time, v1, v2, v3, v4, v5, v6, v7, v8] + list(additional_features.values()) + [amount]
        
        if st.button("🎯 Predict", key="manual_predict"):
            prediction_result = self.make_prediction(model_name, features)
            self.display_prediction_result(prediction_result)
    
    def random_sample_prediction(self, model_name):
        """Random sample prediction interface"""
        st.subheader("🎲 Random Sample Prediction")
        
        if st.button("🎲 Generate Random Sample"):
            if self.data is not None:
                # Get random sample from dataset
                sample = self.data.drop('Class', axis=1).sample(1).iloc[0].values
                
                st.subheader("📊 Sample Features:")
                sample_df = pd.DataFrame([sample], columns=self.data.drop('Class', axis=1).columns)
                st.dataframe(sample_df, use_container_width=True)
                
                # Make prediction
                prediction_result = self.make_prediction(model_name, sample)
                self.display_prediction_result(prediction_result)
    
    def batch_prediction_interface(self, model_name):
        """Batch prediction interface"""
        st.subheader("📁 Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch prediction",
            type=['csv'],
            help="Upload a CSV file with the same features as the training data (without Class column)"
        )
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.subheader("📊 Uploaded Data Preview:")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🎯 Predict Batch", key="batch_predict"):
                    predictions = []
                    probabilities = []
                    
                    for i in range(len(batch_data)):
                        features = batch_data.iloc[i].values
                        pred_result = self.make_prediction(model_name, features)
                        predictions.append(pred_result['prediction'])
                        probabilities.append(pred_result['probability'])
                    
                    # Create results dataframe
                    results_df = batch_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Fraud_Probability'] = probabilities
                    results_df['Risk_Level'] = results_df['Fraud_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                    )
                    
                    st.subheader("📊 Prediction Results:")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fraud_count = sum(predictions)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{fraud_count}</h3>
                            <p>Predicted Frauds</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        avg_prob = np.mean(probabilities)
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{avg_prob:.3f}</h3>
                            <p>Avg Fraud Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        high_risk = sum(results_df['Risk_Level'] == 'High')
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{high_risk}</h3>
                            <p>High Risk Transactions</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"fraud_predictions_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def make_prediction(self, model_name, features):
        """Make prediction using selected model"""
        try:
            # Prepare features
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if needed
            if model_name in ['SVM', 'KNN', 'Logistic Regression', 'Naive Bayes', 'FT-Transformer']:
                if self.scaler is None:
                    st.error("❌ Scaler not available. Please run models.py first.")
                    return None
                features_scaled = self.scaler.transform(features_array)
                features_to_use = features_scaled
            else:
                features_to_use = features_array
            
            # Make prediction
            model = self.models[model_name]
            
            if model_name == 'TabNet':
                prediction_prob = model.predict_proba(features_to_use)[0][1]
            elif model_name == 'FT-Transformer':
                # Handle FT-Transformer separately with PyTorch
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_to_use)
                    model.eval()
                    prediction_prob = model(features_tensor).item()
            else:
                prediction_prob = model.predict_proba(features_to_use)[0][1]
            
            prediction = 1 if prediction_prob > 0.5 else 0
            
            return {
                'prediction': prediction,
                'probability': prediction_prob,
                'model': model_name
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None
    
    def display_prediction_result(self, result):
        """Display prediction result with fancy styling"""
        if result:
            probability = result['probability']
            prediction = result['prediction']
            
            # Determine risk level and styling
            if probability > 0.7:
                risk_level = "HIGH RISK"
                box_class = "warning-box"
                emoji = "🚨"
            elif probability > 0.3:
                risk_level = "MEDIUM RISK"
                box_class = "model-card"
                emoji = "⚠️"
            else:
                risk_level = "LOW RISK"
                box_class = "success-box"
                emoji = "✅"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h2>{emoji} PREDICTION RESULT</h2>
                <h3>Risk Level: {risk_level}</h3>
                <p><strong>Fraud Probability:</strong> {probability:.4f} ({probability*100:.2f}%)</p>
                <p><strong>Classification:</strong> {'FRAUD' if prediction == 1 else 'LEGITIMATE'}</p>
                <p><strong>Model Used:</strong> {result['model']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Main title
    st.markdown('<div class="main-header"> Credit Card Fraud Detection Dashboard</div>', 
               unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Choose Section:",
        [
            "ℹ️ About",
            "📊 Dataset Overview",
            "🏆 Model Comparison",
            "🔬 Individual Analysis",
            
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Display current status
    st.sidebar.subheader("📈 System Status")
    
    if dashboard.data is not None:
        st.sidebar.success("✅ Dataset Loaded")
        st.sidebar.info(f"📊 {len(dashboard.data):,} transactions")
        st.sidebar.info(f"🎯 {dashboard.data['Class'].sum():,} fraud cases")
    else:
        st.sidebar.error("❌ Dataset Not Found")
    
    if dashboard.models:
        st.sidebar.success(f"✅ {len(dashboard.models)} Models Loaded")
    else:
        st.sidebar.warning("⚠️ No Models Loaded")
    
    st.sidebar.markdown("---")
    
    # Model performance summary in sidebar
    if dashboard.results_summary and 'model_results' in dashboard.results_summary:
        st.sidebar.subheader("🏆 Top 3 Models")
        results = dashboard.results_summary['model_results']
        sorted_models = sorted(results.items(), key=lambda x: x[1]['auc_score'], reverse=True)[:3]
        
        for i, (model, metrics) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉"][i]
            st.sidebar.metric(
                label=f"{medal} {model}",
                value=f"{metrics['auc_score']:.4f}",
                delta=f"{metrics['type']}"
            )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tip:** Use the navigation above to explore different sections of the dashboard!")
    
    # Main content based on selected page
    if page == "📊 Dataset Overview":
        dashboard.display_dataset_overview()
        
    elif page == "🏆 Model Comparison":
        dashboard.display_model_comparison()
        
    elif page == "🔬 Individual Analysis":
        dashboard.display_individual_model_analysis()
        
    elif page == "ℹ️ About":
        display_about_page()

def display_about_page():
    """Display information about the project"""
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Project Overview
        
        This comprehensive **Credit Card Fraud Detection** project demonstrates the application of both 
        traditional Machine Learning and modern Deep Learning techniques to solve a real-world financial problem.
        
        ### 🚀 Key Features
        
        #### 📊 **Comprehensive Model Comparison**
        - **10 Traditional ML Models**: From Logistic Regression to XGBoost
        - **2 Deep Learning Models**: TabNet and FT-Transformer
        - **Performance Metrics**: AUC, Precision, Recall, F1-Score
        - **Visual Analysis**: ROC curves, confusion matrices, feature importance
        
        #### 🔬 **Advanced Deep Learning**
        - **TabNet**: Attention-based neural network specifically designed for tabular data
        - **FT-Transformer**: Feature Tokenizer + Transformer architecture for structured data
        - **Custom Implementation**: Built without external libraries like rtdl
        
        #### 🎨 **Interactive Dashboard**
        - **Real-time Predictions**: Individual and batch transaction analysis
        - **Model Switching**: Compare different models on the same data
        - **Visualizations**: Interactive plots with Plotly
        - **Dark/Light Mode**: Compatible with both themes
        
        ### 🛠 **Technical Implementation**
        
        #### **Data Pipeline**
        ```python
        # Complete preprocessing and feature engineering
        - Standardization for distance-based models
        - Stratified splitting to maintain class balance
        - Cross-validation for robust evaluation
        ```
        
        #### **Model Training**
        ```python
        # Automated training pipeline
        - Hyperparameter optimization
        - Early stopping for deep learning
        - Model serialization and persistence
        ```
        
        #### **Evaluation Framework**
        ```python
        # Comprehensive evaluation metrics
        - AUC-ROC for imbalanced dataset
        - Precision/Recall for fraud detection focus
        - Confusion matrices for detailed analysis
        ```
        
        ### 📈 **Business Impact**
        
        **Fraud Detection Performance:**
        - High recall to catch fraud cases
        - Balanced precision to minimize false alarms  
        - Real-time prediction capability
        - Scalable architecture for production deployment
        
        **Cost-Benefit Analysis:**
        - Reduced financial losses from fraud
        - Improved customer satisfaction
        - Automated decision making
        - Risk assessment and monitoring
        """)
    
    with col2:
        st.markdown("""
        ### 🏗 **Architecture**
        
        **Frontend:**
        - Streamlit Dashboard
        - Interactive Visualizations
        - Responsive Design
        - Real-time Updates
        
        **Backend:**
        - Scikit-learn Models
        - PyTorch Deep Learning
        - TabNet Implementation
        - Custom FT-Transformer
        
        **Data Pipeline:**
        - Pandas Processing
        - Feature Engineering
        - Model Persistence
        - Result Caching
        
        ### 📊 **Dataset Stats**
        - **284,807** Total Transactions
        - **492** Fraud Cases (0.172%)
        - **30** Features (Time, V1-V28, Amount)
        - **Highly Imbalanced** Classification Problem
        
        ### 🎓 **Learning Outcomes**
        - Imbalanced dataset handling
        - Deep learning for tabular data
        - Model comparison methodologies
        - Production-ready ML systems
        - Interactive dashboard development
        
        ### 🔧 **Technologies Used**
        - **Python** - Core programming
        - **Streamlit** - Web dashboard
        - **PyTorch** - Deep learning
        - **Scikit-learn** - Traditional ML
        - **Plotly** - Interactive visualizations
        - **TabNet** - Tabular neural networks
        
        ### 📚 **References**
        - [TabNet Paper](https://arxiv.org/abs/1908.07442)
        - [FT-Transformer Paper](https://arxiv.org/abs/2106.11959)
        - [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        """)
    
    # Technical details expander
    with st.expander("🔧 Technical Implementation Details", expanded=False):
        st.markdown("""
        ### Model Architectures
        
        #### **TabNet Architecture**
        ```python
        TabNet(
            input_dim=30,           # Number of features
            output_dim=1,           # Binary classification
            n_d=64,                 # Decision prediction layer width
            n_a=64,                 # Attention layer width  
            n_steps=5,              # Number of decision steps
            gamma=1.5,              # Feature reusage coefficient
            n_independent=2,        # Number of independent GLU layers
            n_shared=2,             # Number of shared GLU layers
            epsilon=1e-15,          # Small number for numerical stability
            virtual_batch_size=128, # Virtual batch size
            momentum=0.98,          # Momentum for batch normalization
            mask_type='entmax'      # Sparse attention mechanism
        )
        ```
        
        #### **FT-Transformer Architecture**  
        ```python
        FTTransformer(
            input_dim=30,           # Number of input features
            d_model=64,             # Model dimension
            nhead=8,                # Number of attention heads
            num_layers=3,           # Number of transformer layers
            dropout=0.1,            # Dropout rate
            activation='relu'       # Activation function
        )
        
        # Components:
        - Feature Tokenization Layer
        - Positional Encoding
        - Multi-Head Attention
        - Feed-Forward Networks
        - Layer Normalization
        - Classification Head
        ```
        
        ### Training Strategy
        
        #### **Data Splitting**
        - **Training**: 60% (stratified)
        - **Validation**: 20% (stratified) 
        - **Test**: 20% (stratified)
        - **Stratification**: Maintains 0.172% fraud rate
        
        #### **Hyperparameter Optimization**
        - Grid search for traditional ML
        - Bayesian optimization for deep learning
        - Cross-validation for robust evaluation
        - Early stopping to prevent overfitting
        
        #### **Evaluation Metrics**
        ```python
        Primary: AUC-ROC Score
        - Handles class imbalance effectively
        - Threshold-independent evaluation
        - Standard metric for fraud detection
        
        Secondary Metrics:
        - Precision (Fraud class focus)
        - Recall (Fraud class focus) 
        - F1-Score (Balanced measure)
        - Confusion Matrix (Detailed analysis)
        ```
        
        ### Performance Optimizations
        
        #### **Memory Efficiency**
        - Batch processing for large datasets
        - Model serialization with pickle
        - Lazy loading of visualizations
        - Efficient data structures
        
        #### **Compute Efficiency**
        - CPU-optimized inference
        - Vectorized operations
        - Caching of repeated computations
        - Streamlined prediction pipeline
        
        #### **User Experience**
        - Progressive loading
        - Responsive design
        - Error handling
        - Clear feedback messages
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎓 **Educational Value**
        - Machine Learning Comparison
        - Deep Learning for Tabular Data
        - Production ML Systems
        - Interactive Dashboards
        """)
    
    with col2:
        st.markdown("""
        ### 🏭 **Production Ready**
        - Scalable Architecture
        - Error Handling
        - Model Persistence  
        - Real-time Inference
        """)
    
    with col3:
        st.markdown("""
        ### 🔬 **Research Quality**
        - Rigorous Evaluation
        - Statistical Significance
        - Reproducible Results
        - Open Source Implementation
        """)

if __name__ == "__main__":
    main()