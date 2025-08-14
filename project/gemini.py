import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
import math

# Set page config
st.set_page_config(page_title="Fraud Detection with Deep Learning", layout="wide")

# Custom CSS for a more professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .stSelectbox {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- FT-Transformer Implementation (No rtdl) ---
class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, d_token):
        super().__init__()
        self.d_token = d_token
        self.weight = nn.Parameter(torch.randn(n_num_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_num_features, d_token))

    def forward(self, x_num):
        return x_num[..., None] * self.weight + self.bias

class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ffn, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num_features, d_token, n_blocks, n_heads, d_ffn, dropout, d_out):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(n_num_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_blocks)]
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, d_out)
        )

    def forward(self, x_num):
        x = self.feature_tokenizer(x_num)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        # Get the CLS token representation
        cls_token_output = x[:, 0]
        
        # Pass through the head
        out = self.head(cls_token_output)
        return out

# --- End of FT-Transformer Implementation ---

@st.cache_data
def load_data(file):
    """Load and preprocess the data."""
    df = pd.read_csv(file)
    return df

def plot_confusion_matrix(cm, classes):
    """Plot the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

def plot_roc_curve(y_test, y_pred_proba):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(fig)

def main():
    """Main function of the Streamlit app."""
    st.title("Credit Card Fraud Detection with Deep Learning")
    st.markdown("### A comparative study of TabNet and FT-Transformer models.")

    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
    model_choice = st.sidebar.selectbox("Choose a model", ["TabNet", "FT-Transformer"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        # Preprocessing
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Scale 'Time' and 'Amount'
        scaler = StandardScaler()
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_np = X_train.values
        y_train_np = y_train.values # Target arrays should be 1D
        X_test_np = X_test.values
        y_test_np = y_test.values

        if st.sidebar.button("Train Model"):
            with st.spinner(f"Training {model_choice}... this may take a moment."):
                if model_choice == "TabNet":
                    # TabNet Model
                    clf = TabNetClassifier(
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=2e-2),
                        scheduler_params={"step_size":10, "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='sparsemax' 
                    )
                    clf.fit(
                        X_train=X_train_np, y_train=y_train_np,
                        eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
                        eval_name=['train', 'valid'],
                        eval_metric=['auc'],
                        max_epochs=50, patience=10,
                        batch_size=1024, virtual_batch_size=128,
                        num_workers=0,
                        weights=1,
                        drop_last=False
                    )
                    y_pred = clf.predict(X_test_np)
                    y_pred_proba = clf.predict_proba(X_test_np)[:, 1]
                    feature_importances = clf.feature_importances_

                elif model_choice == "FT-Transformer":
                    # FT-Transformer Model
                    model = FTTransformer(
                        n_num_features=X_train_np.shape[1],
                        d_token=192,
                        n_blocks=3,
                        n_heads=8,
                        d_ffn=768,
                        dropout=0.1,
                        d_out=1
                    )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    
                    # Create a simple data loader
                    # Reshape y_train_np to (n_samples, 1) for BCEWithLogitsLoss
                    train_dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(X_train_np).float(), 
                        torch.from_numpy(y_train_np.reshape(-1, 1)).float()
                    )
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

                    # Training loop
                    for epoch in range(5): # Reduced epochs for faster demo
                        model.train()
                        for x_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            y_pred_logits = model(x_batch)
                            loss = loss_fn(y_pred_logits, y_batch)
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        x_test_tensor = torch.from_numpy(X_test_np).float()
                        y_pred_logits = model(x_test_tensor)
                        y_pred_proba = torch.sigmoid(y_pred_logits).numpy().flatten()
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    feature_importances = None


            st.success(f"{model_choice} model trained successfully!")

            # Display Results
            st.subheader(f"Results for {model_choice}")
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Performance Metrics")
                st.write(f"**Accuracy:** {accuracy_score(y_test_np, y_pred):.4f}")
                st.write(f"**Precision:** {precision_score(y_test_np, y_pred):.4f}")
                st.write(f"**Recall:** {recall_score(y_test_np, y_pred):.4f}")
                st.write(f"**F1-Score:** {f1_score(y_test_np, y_pred):.4f}")
                st.write(f"**AUC-ROC:** {roc_auc_score(y_test_np, y_pred_proba):.4f}")

            with col2:
                cm = confusion_matrix(y_test_np, y_pred)
                plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])

            plot_roc_curve(y_test_np, y_pred_proba)
            
            if feature_importances is not None:
                st.subheader("Feature Importances")
                fig, ax = plt.subplots(figsize=(10, 8))
                sorted_idx = feature_importances.argsort()
                plt.barh(X.columns[sorted_idx], feature_importances[sorted_idx])
                plt.xlabel("TabNet Feature Importance")
                st.pyplot(fig)

if __name__ == '__main__':
    main()
