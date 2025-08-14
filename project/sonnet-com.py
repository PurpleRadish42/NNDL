import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🔍 Advanced Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .ml-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .dl-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# FT-Transformer Implementation (without rtdl)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, d_ff=128, n_classes=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        
        # Feature tokenization
        self.feature_tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features + 1, d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize features
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        x = self.feature_tokenizer(x)  # (batch_size, n_features, d_model)
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, n_features + 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Use CLS token for classification
        x = self.norm(x[:, 0, :])  # Take CLS token
        output = self.classifier(x)
        
        return output

# Helper functions
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def get_ml_models():
    """Initialize traditional ML models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'Naive Bayes': GaussianNB(),
        'Isolation Forest': IsolationForest(random_state=42, contamination=0.1)
    }
    return models

def plot_class_distribution(df):
    fig = px.pie(
        values=df['Class'].value_counts().values,
        names=['Normal', 'Fraud'],
        title="Class Distribution",
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_correlation_heatmap(df):
    # Select a subset of features for visualization
    features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V17', 'V14', 'V12', 'V10', 'Amount', 'Class']
    corr = df[features_to_plot].corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu_r'
    )
    return fig

def plot_model_comparison(results_df):
    """Plot comprehensive model comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AUC Score Comparison', 'F1 Score Comparison', 
                       'Precision Comparison', 'Recall Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sort by AUC for better visualization
    results_sorted = results_df.sort_values('AUC', ascending=True)
    
    colors = ['#2ecc71' if 'ML' in cat else '#e74c3c' for cat in results_sorted['Category']]
    
    # AUC Score
    fig.add_trace(
        go.Bar(x=results_sorted['AUC'], y=results_sorted['Model'], 
               orientation='h', name='AUC', marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # F1 Score
    fig.add_trace(
        go.Bar(x=results_sorted['F1'], y=results_sorted['Model'], 
               orientation='h', name='F1', marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # Precision
    fig.add_trace(
        go.Bar(x=results_sorted['Precision'], y=results_sorted['Model'], 
               orientation='h', name='Precision', marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # Recall
    fig.add_trace(
        go.Bar(x=results_sorted['Recall'], y=results_sorted['Model'], 
               orientation='h', name='Recall', marker_color=colors, showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Comprehensive Model Performance Comparison")
    return fig

def plot_roc_curves_comprehensive(y_true, ml_predictions, dl_predictions):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    # ML Models
    for model_name, y_pred_proba in ml_predictions.items():
        if model_name != 'Isolation Forest':  # Skip isolation forest for ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=2, dash='dot'),
                opacity=0.7
            ))
    
    # DL Models
    for model_name, y_pred_proba in dl_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=4),
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray', width=2)
    ))
    
    fig.update_layout(
        title='ROC Curves: ML vs DL Models Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600
    )
    return fig

def plot_feature_importance_comparison(feature_names, ml_importances, tabnet_importance):
    """Compare feature importance between ML and DL models"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Random Forest Feature Importance', 'TabNet Feature Importance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Random Forest importance (top 15)
    rf_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': ml_importances['Random Forest']
    }).sort_values('Importance', ascending=False).head(15)
    
    fig.add_trace(
        go.Bar(x=rf_df['Importance'], y=rf_df['Feature'], 
               orientation='h', name='Random Forest',
               marker_color='#2ecc71'),
        row=1, col=1
    )
    
    # TabNet importance (top 15)
    tabnet_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': tabnet_importance
    }).sort_values('Importance', ascending=False).head(15)
    
    fig.add_trace(
        go.Bar(x=tabnet_df['Importance'], y=tabnet_df['Feature'], 
               orientation='h', name='TabNet',
               marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_yaxes(categoryorder='total ascending', row=1, col=1)
    fig.update_yaxes(categoryorder='total ascending', row=1, col=2)
    
    return fig

def train_ml_models(X_train, X_test, y_train, y_test, use_sampling=True):
    """Train all ML models"""
    models = get_ml_models()
    results = {}
    ml_predictions = {}
    ml_feature_importance = {}
    
    # Apply SMOTE for imbalanced data
    if use_sampling:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        try:
            if name == 'Isolation Forest':
                # Isolation Forest is unsupervised
                model.fit(X_train)
                y_pred = model.predict(X_test)
                y_pred = np.where(y_pred == -1, 1, 0)  # Convert outliers to fraud
                y_pred_proba = model.decision_function(X_test)
                # Normalize to [0,1] range
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba) if name != 'Isolation Forest' else roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            results[name] = {
                'AUC': auc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            }
            
            ml_predictions[name] = y_pred_proba
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                ml_feature_importance[name] = model.feature_importances_
                
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(models))
    
    progress_bar.empty()
    status_text.empty()
    
    return results, ml_predictions, ml_feature_importance

def train_pytorch_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    progress_bar.empty()
    status_text.empty()
    
    return model, train_losses, val_losses, val_accuracies

def main():
   st.markdown('<h1 class="main-header">🔍 Advanced Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
   st.markdown("### Comprehensive Comparison: Traditional ML vs Deep Learning Models")
   
   # Sidebar
   st.sidebar.title("🎛️ Configuration")
   
   # File upload
   uploaded_file = st.sidebar.file_uploader(
       "Upload Credit Card Dataset",
       type=['csv'],
       help="Upload the creditcard.csv file"
   )
   
   if uploaded_file is not None:
       # Load data
       with st.spinner("Loading dataset..."):
           df = load_data(uploaded_file)
       
       st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
       
       # Dataset Overview
       st.header("📊 Dataset Overview")
       
       col1, col2, col3, col4 = st.columns(4)
       with col1:
           st.markdown(f"""
           <div class="metric-card">
               <h3>{len(df):,}</h3>
               <p>Total Transactions</p>
           </div>
           """, unsafe_allow_html=True)
       
       with col2:
           fraud_count = df['Class'].sum()
           st.markdown(f"""
           <div class="metric-card">
               <h3>{fraud_count:,}</h3>
               <p>Fraudulent Transactions</p>
           </div>
           """, unsafe_allow_html=True)
       
       with col3:
           fraud_rate = (fraud_count / len(df)) * 100
           st.markdown(f"""
           <div class="metric-card">
               <h3>{fraud_rate:.2f}%</h3>
               <p>Fraud Rate</p>
           </div>
           """, unsafe_allow_html=True)
       
       with col4:
           st.markdown(f"""
           <div class="metric-card">
               <h3>{df.shape[1]-1}</h3>
               <p>Features</p>
           </div>
           """, unsafe_allow_html=True)
       
       # Visualizations
       st.header("📈 Exploratory Data Analysis")
       
       col1, col2 = st.columns(2)
       
       with col1:
           st.plotly_chart(plot_class_distribution(df), use_container_width=True)
       
       with col2:
           # Amount distribution by class
           fig = px.box(
               df, 
               x='Class', 
               y='Amount',
               title="Transaction Amount Distribution by Class",
               color='Class',
               color_discrete_sequence=['#3498db', '#e74c3c']
           )
           fig.update_xaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
           st.plotly_chart(fig, use_container_width=True)
       
       # Correlation heatmap
       st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
       
       # Model Training Configuration
       st.header("⚙️ Model Training Configuration")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.subheader("🔧 Data Splitting")
           test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
           use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
       
       with col2:
           st.subheader("🤖 ML Models")
           train_ml = st.checkbox("Train Traditional ML Models", value=True)
           ml_cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
       
       with col3:
           st.subheader("🧠 DL Models")
           train_dl = st.checkbox("Train Deep Learning Models", value=True)
           batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
           epochs = st.slider("Epochs", 5, 50, 20, 5)
       
       if st.button("🚀 Start Comprehensive Training & Comparison"):
           # Data preparation
           with st.spinner("Preparing data..."):
               X = df.drop('Class', axis=1)
               y = df['Class']
               
               # Split data
               X_train, X_test, y_train, y_test = train_test_split(
                   X, y, test_size=test_size, random_state=42, stratify=y
               )
               
               # Scale features
               scaler = StandardScaler()
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.transform(X_test)
               
               feature_names = X.columns.tolist()
           
           all_results = []
           ml_predictions = {}
           dl_predictions = {}
           ml_feature_importance = {}
           
           # Train ML Models
           if train_ml:
               st.header("🤖 Training Traditional ML Models")
               ml_results, ml_predictions, ml_feature_importance = train_ml_models(
                   X_train_scaled, X_test_scaled, y_train, y_test, use_smote
               )
               
               # Add ML results to comparison
               for model_name, metrics in ml_results.items():
                   all_results.append({
                       'Model': model_name,
                       'Category': 'Traditional ML',
                       'AUC': metrics['AUC'],
                       'F1': metrics['F1'],
                       'Precision': metrics['Precision'],
                       'Recall': metrics['Recall']
                   })
               
               st.success(f"✅ Trained {len(ml_results)} ML models successfully!")
           
           # Train DL Models
           if train_dl:
               st.header("🧠 Training Deep Learning Models")
               
               # Further split training data for validation
               X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
                   X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
               )
               
               # Apply SMOTE to DL training data if selected
               if use_smote:
                   smote = SMOTE(random_state=42)
                   X_train_dl, y_train_dl = smote.fit_resample(X_train_dl, y_train_dl)
               
               # Train TabNet
               st.subheader("🔥 Training TabNet")
               
               tabnet_model = TabNetClassifier(
                   n_d=64, n_a=64, n_steps=5, gamma=1.3,
                   n_independent=2, n_shared=2, lambda_sparse=1e-3,
                   optimizer_fn=torch.optim.Adam,
                   optimizer_params=dict(lr=2e-2),
                   mask_type='sparsemax',
                   scheduler_params=dict(step_size=50, gamma=0.9),
                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                   verbose=0
               )
               
               with st.spinner("Training TabNet..."):
                   tabnet_model.fit(
                       X_train_dl, y_train_dl,
                       eval_set=[(X_val_dl, y_val_dl)],
                       eval_name=['valid'], eval_metric=['auc'],
                       max_epochs=epochs, patience=10,
                       batch_size=batch_size, virtual_batch_size=128,
                       num_workers=0, drop_last=False
                   )
               
               # TabNet predictions and metrics
               tabnet_pred_proba = tabnet_model.predict_proba(X_test_scaled)[:, 1]
               tabnet_pred = (tabnet_pred_proba > 0.5).astype(int)
               
               tabnet_metrics = {
                   'AUC': roc_auc_score(y_test, tabnet_pred_proba),
                   'F1': f1_score(y_test, tabnet_pred),
                   'Precision': precision_score(y_test, tabnet_pred),
                   'Recall': recall_score(y_test, tabnet_pred)
               }
               
               dl_predictions['TabNet'] = tabnet_pred_proba
               
               all_results.append({
                   'Model': 'TabNet',
                   'Category': 'Deep Learning',
                   'AUC': tabnet_metrics['AUC'],
                   'F1': tabnet_metrics['F1'],
                   'Precision': tabnet_metrics['Precision'],
                   'Recall': tabnet_metrics['Recall']
               })
               
               st.success("✅ TabNet training completed!")
               
               # Train FT-Transformer
               st.subheader("🔮 Training FT-Transformer")
               
               # Prepare PyTorch datasets
               train_dataset = TensorDataset(
                   torch.FloatTensor(X_train_dl),
                   torch.LongTensor(y_train_dl.values if hasattr(y_train_dl, 'values') else y_train_dl)
               )
               val_dataset = TensorDataset(
                   torch.FloatTensor(X_val_dl),
                   torch.LongTensor(y_val_dl.values if hasattr(y_val_dl, 'values') else y_val_dl)
               )
               
               train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
               val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
               
               ft_transformer = FTTransformer(
                   n_features=X_train_dl.shape[1],
                   d_model=64, n_heads=4, n_layers=3,
                   d_ff=128, n_classes=2, dropout=0.1
               )
               
               with st.spinner("Training FT-Transformer..."):
                   ft_transformer, train_losses, val_losses, val_accuracies = train_pytorch_model(
                       ft_transformer, train_loader, val_loader, epochs=epochs
                   )
               
               # FT-Transformer predictions
               ft_transformer.eval()
               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               ft_transformer = ft_transformer.to(device)
               
               test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
               test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
               
               ft_pred_proba = []
               with torch.no_grad():
                   for batch in test_loader:
                       batch_X = batch[0].to(device)
                       outputs = ft_transformer(batch_X)
                       proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                       ft_pred_proba.extend(proba)
               
               ft_pred_proba = np.array(ft_pred_proba)
               ft_pred = (ft_pred_proba > 0.5).astype(int)
               
               ft_metrics = {
                   'AUC': roc_auc_score(y_test, ft_pred_proba),
                   'F1': f1_score(y_test, ft_pred),
                   'Precision': precision_score(y_test, ft_pred),
                   'Recall': recall_score(y_test, ft_pred)
               }
               
               dl_predictions['FT-Transformer'] = ft_pred_proba
               
               all_results.append({
                   'Model': 'FT-Transformer',
                   'Category': 'Deep Learning',
                   'AUC': ft_metrics['AUC'],
                   'F1': ft_metrics['F1'],
                   'Precision': ft_metrics['Precision'],
                   'Recall': ft_metrics['Recall']
               })
               
               st.success("✅ FT-Transformer training completed!")
           
           # Comprehensive Results Analysis
           st.header("📊 Comprehensive Model Comparison & Analysis")
           
           if all_results:
               results_df = pd.DataFrame(all_results)
               
               # Overall Performance Summary
               st.subheader("🏆 Performance Leaderboard")
               
               # Sort by AUC score
               leaderboard = results_df.sort_values('AUC', ascending=False)
               
               # Color-code by category
               def get_category_color(category):
                   return '#2ecc71' if category == 'Traditional ML' else '#e74c3c'
               
               # Display top performers
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   winner = leaderboard.iloc[0]
                   color = get_category_color(winner['Category'])
                   st.markdown(f"""
                   <div style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%); 
                               padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                       <h3>🥇 Best Overall</h3>
                       <h4>{winner['Model']}</h4>
                       <p>AUC: {winner['AUC']:.4f}</p>
                       <small>{winner['Category']}</small>
                   </div>
                   """, unsafe_allow_html=True)
               
               with col2:
                   best_ml = leaderboard[leaderboard['Category'] == 'Traditional ML'].iloc[0] if len(leaderboard[leaderboard['Category'] == 'Traditional ML']) > 0 else None
                   if best_ml is not None:
                       st.markdown(f"""
                       <div class="ml-card">
                           <h3>🤖 Best ML Model</h3>
                           <h4>{best_ml['Model']}</h4>
                           <p>AUC: {best_ml['AUC']:.4f}</p>
                       </div>
                       """, unsafe_allow_html=True)
               
               with col3:
                   best_dl = leaderboard[leaderboard['Category'] == 'Deep Learning'].iloc[0] if len(leaderboard[leaderboard['Category'] == 'Deep Learning']) > 0 else None
                   if best_dl is not None:
                       st.markdown(f"""
                       <div class="dl-card">
                           <h3>🧠 Best DL Model</h3>
                           <h4>{best_dl['Model']}</h4>
                           <p>AUC: {best_dl['AUC']:.4f}</p>
                       </div>
                       """, unsafe_allow_html=True)
               
               # Detailed Performance Table
               st.subheader("📋 Detailed Performance Metrics")
               
               # Format the results table
               display_df = leaderboard.copy()
               display_df['AUC'] = display_df['AUC'].round(4)
               display_df['F1'] = display_df['F1'].round(4)
               display_df['Precision'] = display_df['Precision'].round(4)
               display_df['Recall'] = display_df['Recall'].round(4)
               
               st.dataframe(
                   display_df,
                   use_container_width=True,
                   hide_index=True
               )
               
               # Comprehensive Visualization
               st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)
               
               # ROC Curves Comparison
               if ml_predictions and dl_predictions:
                   st.subheader("📈 ROC Curves: ML vs DL Comparison")
                   st.plotly_chart(
                       plot_roc_curves_comprehensive(y_test, ml_predictions, dl_predictions),
                       use_container_width=True
                   )
               
               # Feature Importance Comparison
               if train_ml and train_dl and 'Random Forest' in ml_feature_importance:
                   st.subheader("🎯 Feature Importance Analysis: ML vs DL")
                   tabnet_importance = tabnet_model.feature_importances_
                   st.plotly_chart(
                       plot_feature_importance_comparison(
                           feature_names, ml_feature_importance, tabnet_importance
                       ),
                       use_container_width=True
                   )
               
               # Advanced Analysis Section
               st.header("🔬 Advanced Analysis & Insights")
               
               # Performance by Category
               st.subheader("📊 Performance by Model Category")
               
               category_stats = results_df.groupby('Category').agg({
                   'AUC': ['mean', 'std', 'max'],
                   'F1': ['mean', 'std', 'max'],
                   'Precision': ['mean', 'std', 'max'],
                   'Recall': ['mean', 'std', 'max']
               }).round(4)
               
               category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
               st.dataframe(category_stats, use_container_width=True)
               
               # Statistical Significance Testing
               if train_ml and train_dl:
                   st.subheader("🔍 Statistical Analysis")
                   
                   ml_scores = results_df[results_df['Category'] == 'Traditional ML']['AUC'].values
                   dl_scores = results_df[results_df['Category'] == 'Deep Learning']['AUC'].values
                   
                   if len(ml_scores) > 0 and len(dl_scores) > 0:
                       try:
                           from scipy import stats
                           # Perform t-test
                           t_stat, p_value = stats.ttest_ind(ml_scores, dl_scores)
                       except ImportError:
                           st.warning("scipy not installed. Install it with: pip install scipy")
                           t_stat, p_value = 0, 1.0
                       
                       col1, col2, col3 = st.columns(3)
                       
                       with col1:
                           st.metric("ML Average AUC", f"{np.mean(ml_scores):.4f}")
                       with col2:
                           st.metric("DL Average AUC", f"{np.mean(dl_scores):.4f}")
                       with col3:
                           st.metric("P-value", f"{p_value:.4f}")
                       
                       if p_value < 0.05:
                           st.success("📊 **Statistically Significant Difference** between ML and DL models!")
                       else:
                           st.info("📊 No statistically significant difference between ML and DL models.")
               
               # Business Impact Analysis
               st.header("💼 Business Impact Analysis")
               
               # Calculate cost-benefit for top models
               top_3_models = leaderboard.head(3)
               
               st.subheader("💰 Cost-Benefit Analysis")
               
               # Assumptions for business impact
               avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
               total_transactions = len(y_test)
               fraud_cases = sum(y_test)
               
               st.write(f"""
               **Business Context:**
               - Average fraud amount: ${avg_fraud_amount:.2f}
               - Total test transactions: {total_transactions:,}
               - Actual fraud cases: {fraud_cases:,}
               """)
               
               impact_data = []
               
               for _, model_row in top_3_models.iterrows():
                   model_name = model_row['Model']
                   precision = model_row['Precision']
                   recall = model_row['Recall']
                   
                   # Calculate business metrics
                   predicted_frauds = int(fraud_cases / recall) if recall > 0 else 0
                   true_positives = int(fraud_cases * recall)
                   false_positives = int(predicted_frauds - true_positives)
                   false_negatives = fraud_cases - true_positives
                   
                   # Financial impact
                   fraud_prevented = true_positives * avg_fraud_amount
                   fraud_missed = false_negatives * avg_fraud_amount
                   investigation_cost = predicted_frauds * 50  # Assume $50 per investigation
                   
                   impact_data.append({
                       'Model': model_name,
                       'Fraud Prevented ($)': f"${fraud_prevented:,.2f}",
                       'Fraud Missed ($)': f"${fraud_missed:,.2f}",
                       'Investigation Cost ($)': f"${investigation_cost:,.2f}",
                       'Net Benefit ($)': f"${fraud_prevented - fraud_missed - investigation_cost:,.2f}",
                       'False Alarms': false_positives
                   })
               
               impact_df = pd.DataFrame(impact_data)
               st.dataframe(impact_df, use_container_width=True)
               
               # Model Recommendations
               st.header("🎯 Model Recommendations & Deployment Strategy")
               
               best_model = leaderboard.iloc[0]
               
               recommendations = f"""
               ### 🏆 Recommended Model: **{best_model['Model']}**
               
               **Performance Highlights:**
               - **AUC Score:** {best_model['AUC']:.4f} (Top performer)
               - **F1 Score:** {best_model['F1']:.4f}
               - **Precision:** {best_model['Precision']:.4f}
               - **Recall:** {best_model['Recall']:.4f}
               - **Category:** {best_model['Category']}
               
               ### 📈 Why This Model Excels:
               """
               
               if best_model['Category'] == 'Traditional ML':
                   recommendations += f"""
                   - **Interpretability:** Easy to understand and explain to stakeholders
                   - **Speed:** Fast inference for real-time fraud detection
                   - **Reliability:** Proven track record in production environments
                   - **Maintenance:** Lower computational requirements
                   """
               else:
                   recommendations += f"""
                   - **Complex Patterns:** Captures intricate feature relationships
                   - **State-of-the-art:** Latest deep learning techniques
                   - **Scalability:** Handles large-scale data effectively
                   - **Adaptability:** Can learn from new fraud patterns
                   """
               
               recommendations += f"""
               
               ### 🚀 Deployment Strategy:
               
               1. **Phase 1 - Shadow Mode:** Deploy alongside current system for comparison
               2. **Phase 2 - A/B Testing:** Gradually route traffic to new model
               3. **Phase 3 - Full Deployment:** Complete migration with monitoring
               4. **Phase 4 - Continuous Learning:** Regular retraining and updates
               
               ### ⚠️ Key Considerations:
               
               - **Model Monitoring:** Track performance drift and retrain regularly
               - **Explainability:** Implement SHAP or LIME for model interpretability
               - **Threshold Tuning:** Optimize decision threshold based on business requirements
               - **Compliance:** Ensure model meets regulatory requirements
               """
               
               st.markdown(recommendations)
               
               # Export Results
               st.header("💾 Export Results & Models")
               
               col1, col2 = st.columns(2)
               
               with col1:
                   if st.button("📊 Download Performance Report"):
                       # Create comprehensive report
                       report_data = {
                           'Model Performance': results_df,
                           'Business Impact': impact_df,
                       }
                       
                       # Convert to Excel or CSV
                       csv_buffer = results_df.to_csv(index=False)
                       st.download_button(
                           label="📥 Download Performance CSV",
                           data=csv_buffer,
                           file_name="fraud_detection_comprehensive_results.csv",
                           mime="text/csv"
                       )
               
               with col2:
                   if st.button("🔮 Generate Predictions"):
                       # Create predictions file
                       predictions_data = {
                           'True_Label': y_test.values,
                       }
                       
                       # Add ML predictions
                       for model_name, preds in ml_predictions.items():
                           predictions_data[f'{model_name}_Probability'] = preds
                           predictions_data[f'{model_name}_Prediction'] = (preds > 0.5).astype(int)
                       
                       # Add DL predictions
                       for model_name, preds in dl_predictions.items():
                           predictions_data[f'{model_name}_Probability'] = preds
                           predictions_data[f'{model_name}_Prediction'] = (preds > 0.5).astype(int)
                       
                       pred_df = pd.DataFrame(predictions_data)
                       csv_pred = pred_df.to_csv(index=False)
                       
                       st.download_button(
                           label="📥 Download Predictions CSV",
                           data=csv_pred,
                           file_name="fraud_detection_predictions.csv",
                           mime="text/csv"
                       )
               
               # Final Summary
               st.header("📝 Executive Summary")
               
               summary = f"""
               ### 🎯 Project Summary
               
               **Dataset:** {len(df):,} transactions with {fraud_rate:.2f}% fraud rate
               
               **Models Trained:** {len(all_results)} models across Traditional ML and Deep Learning
               
               **Best Performing Model:** {best_model['Model']} ({best_model['Category']})
               - AUC: {best_model['AUC']:.4f}
               - This model can potentially save significant financial losses while maintaining customer satisfaction
               
               **Key Insights:**
               - {'Deep Learning models outperformed traditional ML' if best_model['Category'] == 'Deep Learning' else 'Traditional ML models remain competitive with Deep Learning'}
               - Feature importance analysis reveals critical fraud indicators
               - The imbalanced nature of fraud data requires careful model evaluation
               
               **Next Steps:**
               1. Deploy the recommended model in a controlled environment
               2. Set up real-time monitoring and alerting systems
               3. Establish a model retraining pipeline
               4. Create explainability dashboards for stakeholders
               """
               
               st.markdown(summary)
   
   else:
       st.info("👆 Please upload the credit card dataset to begin comprehensive analysis")
       
       st.markdown("""
       ### 📋 What This Advanced System Provides
       
       #### 🤖 Traditional ML Models:
       - Logistic Regression
       - Random Forest
       - Decision Tree
       - Support Vector Machine
       - K-Nearest Neighbors
       - Gradient Boosting
       - XGBoost
       - LightGBM
       - Naive Bayes
       - Isolation Forest
       
       #### 🧠 Deep Learning Models:
       - TabNet (Attentive Interpretable Tabular Learning)
       - FT-Transformer (Feature Tokenizer Transformer)
       
       #### 📊 Advanced Analytics:
       - Comprehensive model comparison
       - Statistical significance testing
       - Feature importance analysis
       - Business impact assessment
       - ROC/PR curve analysis
       - Cost-benefit analysis
       - Deployment recommendations
    
       """)

if __name__ == "__main__":
   main()