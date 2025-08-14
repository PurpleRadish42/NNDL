import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🔍 Credit Card Fraud Detection",
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

def plot_feature_importance(feature_names, importances, model_name):
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'{model_name} - Top 15 Feature Importances',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_roc_curves(y_true, predictions_dict):
    fig = go.Figure()
    
    for model_name, y_pred_proba in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=3)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600, height=500
    )
    return fig

def plot_precision_recall_curves(y_true, predictions_dict):
    fig = go.Figure()
    
    for model_name, y_pred_proba in predictions_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=model_name,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title='Precision-Recall Curves Comparison',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600, height=500
    )
    return fig

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

# Main App
def main():
    st.markdown('<h1 class="main-header">🔍 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Deep Learning Models: TabNet vs FT-Transformer")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
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
        st.header("📈 Data Visualizations")
        
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
        
        # Model Training Section
        st.header("🤖 Model Training")
        
        # Sidebar parameters
        st.sidebar.subheader("Model Parameters")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        epochs = st.sidebar.slider("Epochs", 5, 50, 20, 5)
        
        if st.sidebar.button("🚀 Train Models"):
            with st.spinner("Preparing data..."):
                # Prepare data
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
                
                # Further split training data for validation
                X_train_scaled, X_val_scaled, y_train_split, y_val_split = train_test_split(
                    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            
            # Train TabNet
            st.subheader("🔥 Training TabNet")
            
            tabnet_model = TabNetClassifier(
                n_d=64,
                n_a=64,
                n_steps=5,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='sparsemax',
                scheduler_params=dict(step_size=50, gamma=0.9),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0
            )
            
            with st.spinner("Training TabNet..."):
                tabnet_model.fit(
                    X_train_scaled, y_train_split,
                    eval_set=[(X_val_scaled, y_val_split)],
                    eval_name=['valid'],
                    eval_metric=['auc'],
                    max_epochs=epochs,
                    patience=10,
                    batch_size=batch_size,
                    virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False
                )
            
            st.success("✅ TabNet training completed!")
            
            # Train FT-Transformer
            st.subheader("🔮 Training FT-Transformer")
            
            # Prepare PyTorch datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.LongTensor(y_train_split.values)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.LongTensor(y_val_split.values)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            ft_transformer = FTTransformer(
                n_features=X_train_scaled.shape[1],
                d_model=64,
                n_heads=4,
                n_layers=3,
                d_ff=128,
                n_classes=2,
                dropout=0.1
            )
            
            with st.spinner("Training FT-Transformer..."):
                ft_transformer, train_losses, val_losses, val_accuracies = train_pytorch_model(
                    ft_transformer, train_loader, val_loader, epochs=epochs
                )
            
            st.success("✅ FT-Transformer training completed!")
            
            # Model Evaluation
            st.header("📊 Model Evaluation")
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                # TabNet predictions
                tabnet_pred_proba = tabnet_model.predict_proba(X_test_scaled)[:, 1]
                tabnet_pred = (tabnet_pred_proba > 0.5).astype(int)
                
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
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 TabNet Results")
                tabnet_auc = roc_auc_score(y_test, tabnet_pred_proba)
                st.metric("AUC Score", f"{tabnet_auc:.4f}")
                
                st.text("Classification Report:")
                st.text(classification_report(y_test, tabnet_pred))
            
            with col2:
                st.subheader("🔮 FT-Transformer Results")
                ft_auc = roc_auc_score(y_test, ft_pred_proba)
                st.metric("AUC Score", f"{ft_auc:.4f}")
                
                st.text("Classification Report:")
                st.text(classification_report(y_test, ft_pred))
            
            # ROC and PR Curves
            st.subheader("📈 Performance Curves")
            
            predictions_dict = {
                'TabNet': tabnet_pred_proba,
                'FT-Transformer': ft_pred_proba
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_roc_curves(y_test, predictions_dict), use_container_width=True)
            with col2:
                st.plotly_chart(plot_precision_recall_curves(y_test, predictions_dict), use_container_width=True)
            
            # Feature Importance
            st.subheader("🎯 Feature Importance Analysis")
            
            # TabNet feature importance
            tabnet_importance = tabnet_model.feature_importances_
            feature_names = X.columns.tolist()
            
            st.plotly_chart(
                plot_feature_importance(feature_names, tabnet_importance, "TabNet"),
                use_container_width=True
            )
            
            # Training curves for FT-Transformer
            st.subheader("📊 FT-Transformer Training Progress")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss Curves', 'Validation Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs_range = list(range(1, len(train_losses) + 1))
            
            fig.add_trace(
                go.Scatter(x=epochs_range, y=train_losses, name='Train Loss', mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=val_losses, name='Val Loss', mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_range, y=val_accuracies, name='Val Accuracy', mode='lines'),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)

            fig.update_layout(height=400, showlegend=True)
            
            # Model Comparison Summary
            st.header("🏆 Model Comparison Summary")
            
            comparison_data = {
                'Model': ['TabNet', 'FT-Transformer'],
                'AUC Score': [tabnet_auc, ft_auc],
                'Best For': [
                    'Tabular data with feature selection',
                    'Complex feature interactions'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Winner announcement
            if tabnet_auc > ft_auc:
                st.success("🏆 **TabNet wins!** Better performance on this dataset.")
            elif ft_auc > tabnet_auc:
                st.success("🏆 **FT-Transformer wins!** Better performance on this dataset.")
            else:
                st.info("🤝 **It's a tie!** Both models perform equally well.")
            
            # Insights and Analysis
            st.header("🔍 Key Insights & Analysis")
            
            insights = f"""
            ### 📈 Performance Analysis
            
            **Dataset Characteristics:**
            - Total transactions: {len(df):,}
            - Fraud rate: {fraud_rate:.2f}% (highly imbalanced)
            - Features: {df.shape[1]-1} (mostly anonymized PCA components)
            
            **Model Performance:**
            - **TabNet AUC:** {tabnet_auc:.4f}
            - **FT-Transformer AUC:** {ft_auc:.4f}
            
            **Key Observations:**
            1. **Class Imbalance:** With only {fraud_rate:.2f}% fraud cases, both models need to handle severe class imbalance
            2. **Feature Importance:** TabNet provides interpretable feature importance, showing which V-features are most predictive
            3. **Architecture Benefits:** 
               - TabNet: Sequential attention and feature selection
               - FT-Transformer: Multi-head attention for complex interactions
            
            **Business Impact:**
            - High precision is crucial to minimize false positives (customer frustration)
            - High recall ensures we catch actual fraud cases
            - The winning model could save millions in fraud losses while maintaining customer satisfaction
            """
            
            st.markdown(insights)
            
            # Download section
            st.header("💾 Export Results")
            
            if st.button("Download Predictions"):
                results_df = pd.DataFrame({
                    'True_Label': y_test.values,
                    'TabNet_Prediction': tabnet_pred,
                    'TabNet_Probability': tabnet_pred_proba,
                    'FT_Transformer_Prediction': ft_pred,
                    'FT_Transformer_Probability': ft_pred_proba
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please upload the credit card dataset to begin analysis")
        
        st.markdown("""
        ### 📋 Expected Dataset Format
        
        The dataset should contain:
        - **V1, V2, ..., V28**: PCA-transformed features
        - **Time**: Seconds elapsed between transactions
        - **Amount**: Transaction amount
        - **Class**: Target variable (0=Normal, 1=Fraud)
        
        ### 🎯 What This App Does
        
        1. **Data Exploration**: Visualize class distribution, correlations, and patterns
        2. **Model Training**: Train both TabNet and FT-Transformer models
        3. **Performance Comparison**: Compare models using AUC, precision, recall
        4. **Feature Analysis**: Understand which features are most important
        5. **Business Insights**: Provide actionable insights for fraud prevention
        """)

if __name__ == "__main__":
    main()