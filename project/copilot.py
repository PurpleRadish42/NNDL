import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    st.error("Please install pytorch-tabnet: pip install pytorch-tabnet")

# FT-Transformer
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Set page config
st.set_page_config(
    page_title="🔍 Advanced Fraud Detection Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# FT-Transformer Implementation
class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, num_classes=2, dropout=0.1):
        super(FTTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x.repeat(1, self.input_dim, 1)  # Repeat to create sequence
        
        # Project to model dimension
        x = self.input_projection(x.view(batch_size, self.input_dim, -1))
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced Fraud Detection Laboratory</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Deep Learning Models: TabNet vs FT-Transformer")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Credit Card Dataset",
        type=['csv'],
        help="Upload the Kaggle Credit Card Fraud Detection dataset"
    )
    
    if uploaded_file is None:
        # Create sample data demonstration
        st.info("🎯 **Demo Mode**: Upload your dataset or use the sample data below")
        if st.button("🎲 Generate Sample Dataset"):
            df = generate_sample_data()
            st.session_state['data'] = df
            st.success("✅ Sample dataset generated!")
    else:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Data Overview
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{len(df):,}</h3><p>Total Transactions</p></div>', unsafe_allow_html=True)
        
        with col2:
            fraud_count = df['Class'].sum() if 'Class' in df.columns else 0
            st.markdown(f'<div class="metric-card"><h3>{fraud_count:,}</h3><p>Fraud Cases</p></div>', unsafe_allow_html=True)
        
        with col3:
            fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f'<div class="metric-card"><h3>{fraud_rate:.2f}%</h3><p>Fraud Rate</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Features</p></div>', unsafe_allow_html=True)
        
        # Exploratory Data Analysis
        st.markdown("## 🔍 Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution Analysis", "🎯 Feature Analysis", "🔗 Correlation Matrix", "📊 Time Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Class distribution
                fig_pie = px.pie(
                    values=[len(df) - fraud_count, fraud_count],
                    names=['Normal', 'Fraud'],
                    title="🎯 Transaction Class Distribution",
                    color_discrete_sequence=['#00cc96', '#ff6b6b']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Amount distribution
                if 'Amount' in df.columns:
                    fig_hist = px.histogram(
                        df, x='Amount', color='Class',
                        title="💰 Transaction Amount Distribution",
                        nbins=50,
                        color_discrete_sequence=['#00cc96', '#ff6b6b']
                    )
                    fig_hist.update_layout(bargap=0.1)
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            # Feature importance analysis
            st.markdown("### 🎯 Feature Analysis")
            
            # Select features for analysis
            feature_cols = [col for col in df.columns if col.startswith('V') or col in ['Amount', 'Time']]
            
            if feature_cols:
                selected_features = st.multiselect(
                    "Select features to analyze:",
                    feature_cols,
                    default=feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
                )
                
                if selected_features:
                    # Box plots for selected features
                    fig_box = make_subplots(
                        rows=len(selected_features), cols=1,
                        subplot_titles=[f"Feature: {feat}" for feat in selected_features],
                        vertical_spacing=0.05
                    )
                    
                    for i, feature in enumerate(selected_features):
                        for class_val in [0, 1]:
                            data = df[df['Class'] == class_val][feature]
                            fig_box.add_trace(
                                go.Box(
                                    y=data,
                                    name=f"Class {class_val}",
                                    showlegend=(i == 0)
                                ),
                                row=i+1, col=1
                            )
                    
                    fig_box.update_layout(height=200*len(selected_features), title="📊 Feature Distributions by Class")
                    st.plotly_chart(fig_box, use_container_width=True)
        
        with tab3:
            # Correlation matrix
            st.markdown("### 🔗 Feature Correlation Matrix")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="🔗 Feature Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            # Time analysis
            if 'Time' in df.columns:
                st.markdown("### ⏰ Temporal Analysis")
                
                df_time = df.copy()
                df_time['Hour'] = (df_time['Time'] / 3600) % 24
                df_time['Day'] = df_time['Time'] // (24 * 3600)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    hourly_fraud = df_time.groupby('Hour')['Class'].agg(['count', 'sum']).reset_index()
                    hourly_fraud['fraud_rate'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
                    
                    fig_hourly = px.bar(
                        hourly_fraud, x='Hour', y='fraud_rate',
                        title="📈 Fraud Rate by Hour of Day",
                        color='fraud_rate',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                with col2:
                    daily_stats = df_time.groupby('Day')['Class'].agg(['count', 'sum']).reset_index()
                    
                    fig_daily = px.scatter(
                        daily_stats, x='Day', y='sum',
                        size='count', title="📊 Daily Fraud Transactions",
                        color='sum', color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
        
        # Model Selection and Training
        st.markdown("## 🤖 Model Training & Comparison")
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "🎯 Select Model",
                ["TabNet", "FT-Transformer", "Both Models (Comparison)"],
                help="Choose which deep learning model to train"
            )
        
        with col2:
            train_test_ratio = st.slider(
                "📊 Train/Test Split",
                min_value=0.6, max_value=0.9, value=0.8, step=0.05,
                help="Percentage of data for training"
            )
        
        # Training parameters
        st.markdown("### ⚙️ Training Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            epochs = st.number_input("🔄 Epochs", min_value=1, max_value=100, value=10)
        
        with col2:
            batch_size = st.number_input("📦 Batch Size", min_value=32, max_value=1024, value=256)
        
        with col3:
            learning_rate = st.number_input("📈 Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        with col4:
            patience = st.number_input("⏱️ Early Stopping Patience", min_value=3, max_value=20, value=5)
        
        # Train button
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("🔄 Training models... This may take a while."):
                results = train_models(df, model_choice, train_test_ratio, epochs, batch_size, learning_rate, patience)
                
                if results:
                    st.session_state['results'] = results
                    st.success("✅ Training completed successfully!")
        
        # Display results if available
        if 'results' in st.session_state:
            display_results(st.session_state['results'])

def generate_sample_data():
    """Generate sample credit card fraud dataset"""
    np.random.seed(42)
    n_samples = 10000
    n_features = 28
    
    # Generate normal transactions
    normal_data = np.random.normal(0, 1, (int(n_samples * 0.999), n_features))
    normal_labels = np.zeros(int(n_samples * 0.999))
    normal_amounts = np.random.lognormal(2, 1, int(n_samples * 0.999))
    normal_times = np.sort(np.random.uniform(0, 172800, int(n_samples * 0.999)))
    
    # Generate fraud transactions (different distribution)
    fraud_data = np.random.normal(0, 2, (int(n_samples * 0.001), n_features))
    fraud_data[:, :5] += np.random.normal(3, 1, (int(n_samples * 0.001), 5))  # Make some features more extreme
    fraud_labels = np.ones(int(n_samples * 0.001))
    fraud_amounts = np.random.lognormal(4, 1.5, int(n_samples * 0.001))
    fraud_times = np.sort(np.random.uniform(0, 172800, int(n_samples * 0.001)))
    
    # Combine data
    data = np.vstack([normal_data, fraud_data])
    labels = np.hstack([normal_labels, fraud_labels])
    amounts = np.hstack([normal_amounts, fraud_amounts])
    times = np.hstack([normal_times, fraud_times])
    
    # Create DataFrame
    columns = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(data, columns=columns)
    df['Time'] = times
    df['Amount'] = amounts
    df['Class'] = labels.astype(int)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def prepare_data(df, train_test_ratio):
    """Prepare data for training"""
    # Separate features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_test_ratio, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_tabnet(X_train, X_test, y_train, y_test, epochs, batch_size, learning_rate, patience):
    """Train TabNet model"""
    try:
        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=learning_rate),
            mask_type='entmax',
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0
        )
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return model, y_pred, y_pred_proba
    
    except Exception as e:
        st.error(f"TabNet training failed: {str(e)}")
        return None, None, None

def train_ft_transformer(X_train, X_test, y_train, y_test, epochs, batch_size, learning_rate, patience):
    """Train FT-Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train.values).to(device)
    y_test_tensor = torch.LongTensor(y_test.values).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = FTTransformer(
        input_dim=X_train.shape[1],
        d_model=64,
        nhead=8,
        num_layers=3,
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            y_pred.extend(predictions.cpu().numpy())
            y_pred_proba.extend(probabilities[:, 1].cpu().numpy())
    
    return model, np.array(y_pred), np.array(y_pred_proba)

def train_models(df, model_choice, train_test_ratio, epochs, batch_size, learning_rate, patience):
    """Train selected models"""
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, train_test_ratio)
        
        results = {}
        
        if model_choice in ["TabNet", "Both Models (Comparison)"]:
            st.info("🔄 Training TabNet...")
            tabnet_model, tabnet_pred, tabnet_proba = train_tabnet(
                X_train, X_test, y_train, y_test, epochs, batch_size, learning_rate, patience
            )
            
            if tabnet_model is not None:
                results['TabNet'] = {
                    'model': tabnet_model,
                    'predictions': tabnet_pred,
                    'probabilities': tabnet_proba,
                    'y_test': y_test
                }
        
        if model_choice in ["FT-Transformer", "Both Models (Comparison)"]:
            st.info("🔄 Training FT-Transformer...")
            ft_model, ft_pred, ft_proba = train_ft_transformer(
                X_train, X_test, y_train, y_test, epochs, batch_size, learning_rate, patience
            )
            
            if ft_model is not None:
                results['FT-Transformer'] = {
                    'model': ft_model,
                    'predictions': ft_pred,
                    'probabilities': ft_proba,
                    'y_test': y_test
                }
        
        return results
    
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None

def display_results(results):
    """Display training results and metrics"""
    st.markdown("## 📊 Model Performance Results")
    
    # Performance metrics
    metrics_data = []
    
    for model_name, result in results.items():
        y_test = result['y_test']
        y_pred = result['predictions']
        y_proba = result['probabilities']
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics_data.append({
            'Model': model_name,
            'AUC-ROC': auc_score,
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score'],
            'Accuracy': report['accuracy']
        })
    
    # Display metrics table
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 ROC Curves", "🎯 Confusion Matrices", "📊 Feature Importance", "🔍 Predictions Analysis"])
    
    with tab1:
        # ROC Curves
        fig_roc = go.Figure()
        
        for model_name, result in results.items():
            y_test = result['y_test']
            y_proba = result['probabilities']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{model_name} (AUC = {auc_score:.3f})',
                mode='lines',
                line=dict(width=3)
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=True
        ))
        
        fig_roc.update_layout(
            title="📈 ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab2:
        # Confusion Matrices
        if len(results) == 1:
            model_name = list(results.keys())[0]
            result = results[model_name]
            
            cm = confusion_matrix(result['y_test'], result['predictions'])
            
            fig_cm = px.imshow(
                cm, text_auto=True,
                aspect='auto',
                title=f"🎯 Confusion Matrix - {model_name}",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        else:
            # Multiple models
            fig_cm = make_subplots(
                rows=1, cols=len(results),
                subplot_titles=[f"{name}" for name in results.keys()],
                specs=[[{"type": "heatmap"} for _ in results]]
            )
            
            for i, (model_name, result) in enumerate(results.items()):
                cm = confusion_matrix(result['y_test'], result['predictions'])
                
                fig_cm.add_trace(
                    go.Heatmap(
                        z=cm,
                        text=cm,
                        texttemplate="%{text}",
                        colorscale='Blues',
                        showscale=(i == len(results) - 1)
                    ),
                    row=1, col=i+1
                )
            
            fig_cm.update_layout(title="🎯 Confusion Matrices Comparison", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        # Feature importance (for TabNet)
        if 'TabNet' in results:
            st.markdown("### 🎯 TabNet Feature Importance")
            
            try:
                model = results['TabNet']['model']
                feature_importance = model.feature_importances_
                
                # Create feature names
                feature_names = [f'V{i}' for i in range(1, len(feature_importance)-1)] + ['Time', 'Amount']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df.head(15),
                    x='Importance', y='Feature',
                    orientation='h',
                    title="🎯 Top 15 Most Important Features (TabNet)",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not extract feature importance: {str(e)}")
        else:
            st.info("Feature importance is only available for TabNet model.")
    
    with tab4:
        # Predictions analysis
        st.markdown("### 🔍 Prediction Distribution Analysis")
        
        for model_name, result in results.items():
            st.markdown(f"#### {model_name}")
            
            y_test = result['y_test']
            y_proba = result['probabilities']
            
            # Create probability distribution plot
            fraud_probs = y_proba[y_test == 1]
            normal_probs = y_proba[y_test == 0]
            
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=normal_probs,
                name='Normal Transactions',
                opacity=0.7,
                nbinsx=50,
                marker_color='lightblue'
            ))
            
            fig_dist.add_trace(go.Histogram(
                x=fraud_probs,
                name='Fraud Transactions',
                opacity=0.7,
                nbinsx=50,
                marker_color='red'
            ))
            
            fig_dist.update_layout(
                title=f"📊 Prediction Probability Distribution - {model_name}",
                xaxis_title="Fraud Probability",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Model comparison summary
    if len(results) > 1:
        st.markdown("## 🏆 Model Comparison Summary")
        
        best_model = max(metrics_data, key=lambda x: x['AUC-ROC'])
        
        st.markdown(f"""
        <div class="success-card">
            <h3>🏆 Best Performing Model: {best_model['Model']}</h3>
            <p><strong>AUC-ROC:</strong> {best_model['AUC-ROC']:.4f}</p>
            <p><strong>F1-Score:</strong> {best_model['F1-Score']:.4f}</p>
            <p><strong>Precision:</strong> {best_model['Precision']:.4f}</p>
            <p><strong>Recall:</strong> {best_model['Recall']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()