import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

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

class CreditCardFraudDetection:
    def __init__(self, data_path='creditcard.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        
        # Split the data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Fraud cases: {self.y.sum()} ({self.y.mean()*100:.2f}%)")
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def create_dataset_visualizations(self):
        """Create and save dataset visualizations"""
        plt.style.use('default')
        
        # Class distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        class_counts = self.y.value_counts()
        plt.bar(['Normal', 'Fraud'], class_counts.values, color=['skyblue', 'salmon'])
        plt.title('Class Distribution')
        plt.ylabel('Count')
        
        plt.subplot(2, 3, 2)
        plt.pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.2f%%', 
                colors=['skyblue', 'salmon'])
        plt.title('Class Distribution (%)')
        
        # Amount distribution
        plt.subplot(2, 3, 3)
        plt.hist(self.df[self.df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Normal', color='skyblue')
        plt.hist(self.df[self.df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='salmon')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.title('Transaction Amount Distribution')
        plt.legend()
        plt.yscale('log')
        
        # Time distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.df[self.df['Class']==0]['Time'], bins=50, alpha=0.7, label='Normal', color='skyblue')
        plt.hist(self.df[self.df['Class']==1]['Time'], bins=50, alpha=0.7, label='Fraud', color='salmon')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Transaction Time Distribution')
        plt.legend()
        
        # Correlation heatmap (subset)
        plt.subplot(2, 3, 5)
        corr_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
        corr_matrix = self.df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation (Subset)')
        
        # Feature importance preview
        plt.subplot(2, 3, 6)
        # Quick RF for feature importance
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_temp.fit(self.X_train, self.y_train)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('plots/dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def train_ml_models(self):
        """Train traditional ML models"""
        print("\nTraining traditional ML models...")
        
        ml_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        for name, model in ml_models.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['SVM', 'KNN', 'Logistic Regression', 'Naive Bayes']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_val_scaled)
                y_pred_proba = model.predict_proba(self.X_val_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_val)
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_val, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'type': 'ML',
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(self.y_val, y_pred, output_dict=True)
            }
            
            # Save model
            with open(f'models/{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                pickle.dump(model, f)
                
            # Create visualizations
            self.create_model_visualizations(name)
            
            print(f"{name} - AUC: {auc_score:.4f}")
    
    def train_tabnet(self):
        """Train TabNet model"""
        print("\nTraining TabNet...")
        
        # TabNet parameters
        tabnet_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 2,
            'optimizer_fn': optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': 'entmax',
            'scheduler_params': {"step_size": 50, "gamma": 0.9},
            'scheduler_fn': optim.lr_scheduler.StepLR,
            'verbose': 0,
        }
        
        model = TabNetClassifier(**tabnet_params)
        
        # Train
        model.fit(
            X_train=self.X_train.values, y_train=self.y_train.values,
            eval_set=[(self.X_val.values, self.y_val.values)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(self.X_val.values)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_score = roc_auc_score(self.y_val, y_pred_proba)
        
        # Store results
        self.results['TabNet'] = {
            'type': 'DL',
            'model': model,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(self.y_val, y_pred, output_dict=True)
        }
        
        # Save model
        model.save_model('models/tabnet')
        
        # Create visualizations
        self.create_model_visualizations('TabNet')
        
        print(f"TabNet - AUC: {auc_score:.4f}")
    
    def train_ft_transformer(self):
        """Train FT-Transformer model"""
        print("\nTraining FT-Transformer...")
        
        # Prepare data with smaller batch size to save memory
        X_train_tensor = torch.FloatTensor(self.X_train_scaled)
        y_train_tensor = torch.FloatTensor(self.y_train.values)
        X_val_tensor = torch.FloatTensor(self.X_val_scaled)
        y_val_tensor = torch.FloatTensor(self.y_val.values)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Smaller batch sizes to fit in GPU memory
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Initialize model with smaller dimensions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FTTransformer(
            input_dim=self.X_train.shape[1], 
            d_model=32,  # Reduced from 64
            nhead=4,     # Reduced from 8
            num_layers=2, # Reduced from 3
            dropout=0.1
        ).to(device)
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        
        # Training loop with memory management
        train_losses = []
        val_losses = []
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(50):  # Reduced epochs from 100
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
                
                # Clear cache after each batch to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validation with gradient disabled
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                    
                    # Clear cache after each validation batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            # Calculate AUC
            auc_score = roc_auc_score(all_targets, all_preds)
            
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), 'models/ft_transformer.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, AUC: {auc_score:.4f}")
                # Print GPU memory usage
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            
            if patience_counter >= 10:  # Reduced patience from 15
                print("Early stopping triggered")
                break
        
        # Load best model and clear GPU memory
        model.load_state_dict(torch.load('models/ft_transformer.pth'))
        model.eval()
        
        # Final predictions with memory management
        with torch.no_grad():
            # Process in smaller chunks to avoid memory issues
            chunk_size = 1000
            y_pred_proba = []
            
            for i in range(0, len(X_val_tensor), chunk_size):
                chunk = X_val_tensor[i:i+chunk_size].to(device)
                chunk_pred = model(chunk).cpu().numpy()
                y_pred_proba.extend(chunk_pred)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            y_pred_proba = np.array(y_pred_proba)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Clear GPU memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        auc_score = roc_auc_score(self.y_val, y_pred_proba)
        
        # Store results
        self.results['FT-Transformer'] = {
            'type': 'DL',
            'model': model,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(self.y_val, y_pred, output_dict=True),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Create visualizations
        self.create_model_visualizations('FT-Transformer')
        
        print(f"FT-Transformer - AUC: {auc_score:.4f}")
    
    def create_model_visualizations(self, model_name):
        """Create visualizations for a specific model"""
        plt.figure(figsize=(15, 10))
        
        y_true = self.y_val
        y_pred = self.results[model_name]['predictions']
        y_pred_proba = self.results[model_name]['probabilities']
        
        # ROC Curve
        plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Prediction Distribution
        plt.subplot(2, 3, 4)
        plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal', color='skyblue')
        plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', color='salmon')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # Classification Report Heatmap
        plt.subplot(2, 3, 5)
        report = self.results[model_name]['classification_report']
        metrics_data = []
        for class_label in ['0', '1']:
            if class_label in report:
                metrics_data.append([
                    report[class_label]['precision'],
                    report[class_label]['recall'],
                    report[class_label]['f1-score']
                ])
        
        if metrics_data:
            sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=['Precision', 'Recall', 'F1-Score'],
                       yticklabels=['Normal', 'Fraud'])
            plt.title('Classification Metrics')
        
        # Feature Importance (if available)
        plt.subplot(2, 3, 6)
        if hasattr(self.results[model_name]['model'], 'feature_importances_'):
            importances = self.results[model_name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [self.X.columns[i] for i in indices], rotation=45)
            plt.title('Top 10 Feature Importances')
        elif model_name == 'TabNet':
            # TabNet feature importance
            importances = self.results[model_name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [self.X.columns[i] for i in indices], rotation=45)
            plt.title('Top 10 Feature Importances')
        elif model_name == 'FT-Transformer':
            # Training curves for transformer
            losses = self.results[model_name]
            if 'train_losses' in losses:
                plt.plot(losses['train_losses'], label='Train Loss')
                plt.plot(losses['val_losses'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Curves')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower().replace(" ", "_").replace("-", "_")}_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_plots(self):
        """Create model comparison visualizations"""
        print("\nCreating comparison plots...")
        
        # Model Performance Comparison
        plt.figure(figsize=(15, 10))
        
        # AUC Scores Comparison
        plt.subplot(2, 2, 1)
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        colors = ['lightblue' if self.results[name]['type'] == 'ML' else 'lightcoral' 
                 for name in model_names]
        
        bars = plt.bar(range(len(model_names)), auc_scores, color=colors)
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison (AUC)')
        plt.xticks(range(len(model_names)), [name.replace(' ', '\n') for name in model_names], rotation=45)
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # ROC Curves Comparison
        plt.subplot(2, 2, 2)
        for name in model_names:
            y_pred_proba = self.results[name]['probabilities']
            fpr, tpr, _ = roc_curve(self.y_val, y_pred_proba)
            auc = self.results[name]['auc_score']
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Model Type Performance
        plt.subplot(2, 2, 3)
        ml_scores = [score for name, score in zip(model_names, auc_scores) 
                    if self.results[name]['type'] == 'ML']
        dl_scores = [score for name, score in zip(model_names, auc_scores) 
                    if self.results[name]['type'] == 'DL']
        
        plt.boxplot([ml_scores, dl_scores], labels=['Traditional ML', 'Deep Learning'])
        plt.ylabel('AUC Score')
        plt.title('ML vs DL Performance Distribution')
        
        # Performance Summary Table
        plt.subplot(2, 2, 4)
        plt.axis('tight')
        plt.axis('off')
        
        table_data = []
        for name in model_names:
            result = self.results[name]
            precision = result['classification_report']['1']['precision']
            recall = result['classification_report']['1']['recall']
            f1 = result['classification_report']['1']['f1-score']
            table_data.append([name, f"{result['auc_score']:.3f}", 
                             f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'AUC', 'Precision', 'Recall', 'F1-Score'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        plt.title('Performance Summary Table', pad=20)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_summary(self):
        """Save results summary to JSON"""
        summary = {
            'dataset_info': {
                'shape': self.df.shape,
                'fraud_rate': float(self.y.mean()),
                'total_fraud_cases': int(self.y.sum())
            },
            'model_results': {}
        }
        
        for name, result in self.results.items():
            summary['model_results'][name] = {
                'type': result['type'],
                'auc_score': float(result['auc_score']),
                'precision': float(result['classification_report']['1']['precision']),
                'recall': float(result['classification_report']['1']['recall']),
                'f1_score': float(result['classification_report']['1']['f1-score'])
            }
        
        with open('results/model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nResults summary saved!")
    
    def run_complete_pipeline(self):
        """Run the complete ML/DL pipeline"""
        print("="*50)
        print("CREDIT CARD FRAUD DETECTION PIPELINE")
        print("="*50)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create dataset visualizations
        self.create_dataset_visualizations()
        
        # Train ML models
        self.train_ml_models()
        
        # Train DL models
        self.train_tabnet()
        self.train_ft_transformer()
        
        # Create comparison plots
        self.create_comparison_plots()
        
        # Save results summary
        self.save_results_summary()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nResults Summary:")
        for name, result in self.results.items():
            print(f"{name}: AUC = {result['auc_score']:.4f}")

if __name__ == "__main__":
    detector = CreditCardFraudDetection()
    detector.run_complete_pipeline()