import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
from scipy.special import expit
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Neural Network Lab - Perceptron, ADALINE & MADALINE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_diabetes_data():
    """Load Pima Indians Diabetes dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                    'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
    
    try:
        data = pd.read_csv(url, names=column_names)
        return data, True
    except:
        # Fallback: create synthetic data
        np.random.seed(42)
        data = pd.DataFrame({
            'pregnancies': np.random.randint(0, 10, 768),
            'glucose': np.random.randint(50, 200, 768),
            'blood_pressure': np.random.randint(40, 120, 768),
            'skin_thickness': np.random.randint(10, 50, 768),
            'insulin': np.random.randint(0, 300, 768),
            'bmi': np.random.uniform(15, 50, 768),
            'diabetes_pedigree': np.random.uniform(0, 2, 768),
            'age': np.random.randint(20, 80, 768),
            'outcome': np.random.randint(0, 2, 768)
        })
        return data, False

def create_synthetic_data(n_samples, n_features, noise):
    """Create synthetic classification data"""
    return make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_features,
        n_redundant=0, 
        n_clusters_per_class=1,
        flip_y=noise, 
        random_state=42
    )

# ADALINE Implementation
def adaline_fit(X, y, lr=0.01, n_epochs=50):
    """Batch-gradient ADALINE training function"""
    X_b = np.insert(X, 0, 1.0, axis=1)  # Add bias term
    w = np.zeros(X_b.shape[1])
    costs = []
    
    for epoch in range(n_epochs):
        net = X_b.dot(w)
        errors = y - net
        cost = (errors**2).sum() / (2.0 * X.shape[0])
        costs.append(cost)
        w += lr * X_b.T.dot(errors) / X_b.shape[0]
    
    return w, costs

def adaline_net_input(X, w):
    """Calculate net input"""
    return np.insert(X, 0, 1.0, axis=1).dot(w)

def adaline_predict(X, w):
    """Make predictions"""
    return np.where(adaline_net_input(X, w) >= 0.0, 1, 0)

# Activation Functions
def act_fn(z, kind='step'):
    if kind == 'step':
        return np.where(z >= 0, 1, 0)
    if kind == 'linear':
        return z
    if kind == 'sigmoid':
        return expit(z)
    if kind == 'tanh':
        return np.tanh(z)
    if kind == 'relu':
        return np.maximum(0, z)
    raise ValueError("Unknown activation")

def add_bias(X):
    return np.insert(X, 0, 1.0, axis=1)

# Perceptron Implementation
def perceptron_fit(X, y, lr=0.01, epochs=50):
    X_b = add_bias(X)
    w = np.zeros(X_b.shape[1])
    errors_per_epoch = []
    
    for epoch in range(epochs):
        errors = 0
        for xi, target in zip(X_b, y):
            pred = act_fn(np.dot(xi, w), 'step')
            if pred != target:
                w += lr * (target - pred) * xi
                errors += 1
        errors_per_epoch.append(errors)
    
    return w, errors_per_epoch

def perceptron_predict(X, w):
    return act_fn(add_bias(X).dot(w), 'step')

# MADALINE Implementation
def madaline_fit(X, y, lr=0.01, epochs=50, act='tanh', n_hidden=3):
    rng = np.random.RandomState(42)
    X_b = add_bias(X)
    W1 = rng.normal(scale=0.1, size=(X_b.shape[1], n_hidden))
    W2 = rng.normal(scale=0.1, size=(n_hidden + 1, 1))
    costs = []
    
    for epoch in range(epochs):
        # Forward pass
        hidden_raw = X_b.dot(W1)
        hidden_act = act_fn(hidden_raw, act)
        hidden_b = add_bias(hidden_act)
        out_raw = hidden_b.dot(W2)
        out_act = act_fn(out_raw, act)
        
        # Error calculation
        error = y.reshape(-1,1) - out_act
        cost = (error**2).sum() / (2.0 * X.shape[0])
        costs.append(cost)
        
        # Derivatives
        if act == 'linear':
            d = 1
            d_act_hidden = lambda z: np.ones_like(z)
        elif act == 'sigmoid':
            d = out_act * (1 - out_act)
            d_act_hidden = lambda z: act_fn(z,'sigmoid')*(1-act_fn(z,'sigmoid'))
        elif act == 'tanh':
            d = 1 - np.tanh(out_raw)**2
            d_act_hidden = lambda z: 1 - np.tanh(z)**2
        elif act == 'relu':
            d = (out_raw > 0).astype(float)
            d_act_hidden = lambda z: (z > 0).astype(float)
        else:  # step
            d = 1
            d_act_hidden = lambda z: np.ones_like(z)
            
        # Backpropagation
        delta_out = error * d
        W2 += lr * hidden_b.T.dot(delta_out) / X.shape[0]
        delta_hidden = delta_out.dot(W2[1:].T) * d_act_hidden(hidden_raw)
        W1 += lr * X_b.T.dot(delta_hidden) / X.shape[0]
    
    return W1, W2, costs

def madaline_predict(X, W1, W2, act='tanh'):
    hidden = act_fn(add_bias(X).dot(W1), act)
    out = act_fn(add_bias(hidden).dot(W2), act)
    return (out >= 0.5).astype(int).ravel()

# Visualization Functions
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=['No Diabetes', 'Diabetes'],
                    y=['No Diabetes', 'Diabetes'],
                    color_continuous_scale='Blues',
                    title=title)
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=16)
            )
    
    return fig

def plot_decision_boundary_2d(X, y, model_func, title, weights=None, W1=None, W2=None, act='tanh'):
    """Plot decision boundary for 2D data"""
    if X.shape[1] != 2:
        return None
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if model_func == 'adaline':
        Z = adaline_predict(mesh_points, weights)
    elif model_func == 'perceptron':
        Z = perceptron_predict(mesh_points, weights)
    elif model_func == 'madaline':
        Z = madaline_predict(mesh_points, W1, W2, act)
    
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z,
                            colorscale='RdBu', opacity=0.3,
                            showscale=False, line_width=2))
    
    # Add scatter points
    colors = ['red' if label == 0 else 'blue' for label in y]
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], 
                            mode='markers',
                            marker=dict(color=colors, size=8),
                            name='Data Points'))
    
    fig.update_layout(title=title, xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig

def plot_training_curve(costs_or_errors, title, ylabel):
    """Plot training curve"""
    fig = px.line(x=range(len(costs_or_errors)), y=costs_or_errors,
                  title=title, labels={'x': 'Epoch', 'y': ylabel})
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🧠 Neural Network Lab: Perceptron, ADALINE & MADALINE</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("**Lab-2: Implementation of Adaline with Dataset and Simulator**")
    st.markdown("*R Abhijit Srivathsan - 2448044*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Choose program
    program = st.sidebar.selectbox(
        "Select Program",
        ["Program 1: Perceptron", "Program 2: ADALINE", "Program 3: MADALINE Simulator"]
    )
    
    if program == "Program 1: Perceptron":
        st.markdown('<h2 class="sub-header">📊 Program 1: Perceptron</h2>', unsafe_allow_html=True)
        
        # Load data
        data, is_real = load_diabetes_data()
        
        if not is_real:
            st.warning("⚠️ Using synthetic data as fallback (couldn't load real dataset)")
        else:
            st.success("✅ Successfully loaded Pima Indians Diabetes dataset")
        
        # Data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(data.head())
            
        with col2:
            st.subheader("Dataset Statistics")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Features:** {data.shape[1] - 1}")
            st.write(f"**Samples:** {data.shape[0]}")
            
            # Class distribution
            class_dist = data['outcome'].value_counts()
            fig_dist = px.pie(values=class_dist.values, names=['No Diabetes', 'Diabetes'],
                             title="Class Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
        with col3:
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        
        if st.button("🚀 Train Perceptron"):
            # Prepare data
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            with st.spinner("Training Perceptron..."):
                clf = Perceptron(max_iter=max_iter, eta0=learning_rate, random_state=42)
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
            
            # Results
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            with col2:
                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(y_test, y_pred, "Perceptron - Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
    
    elif program == "Program 2: ADALINE":
        st.markdown('<h2 class="sub-header">🔄 Program 2: ADALINE</h2>', unsafe_allow_html=True)
        
        # Choose experiment
        experiment = st.sidebar.selectbox(
            "Select Experiment",
            ["2a: AND Gate", "2b: Diabetes Classification"]
        )
        
        if experiment == "2a: AND Gate":
            st.subheader("🔗 Learning the Logic AND Gate")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                lr_and = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            with col2:
                epochs_and = st.slider("Epochs", 10, 100, 20, 5)
            
            if st.button("🚀 Train ADALINE on AND Gate"):
                # AND gate data
                X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
                y_and = np.array([0,0,0,1])
                
                with st.spinner("Training ADALINE..."):
                    w_and, costs = adaline_fit(X_and, y_and, lr=lr_and, n_epochs=epochs_and)
                    predictions = adaline_predict(X_and, w_and)
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Truth Table Comparison")
                    results_df = pd.DataFrame({
                        'Input 1': X_and[:, 0],
                        'Input 2': X_and[:, 1],
                        'Expected': y_and,
                        'Predicted': predictions,
                        'Correct': y_and == predictions
                    })
                    st.dataframe(results_df)
                    
                    accuracy = np.mean(y_and == predictions)
                    st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    # Training curve
                    fig_cost = plot_training_curve(costs, "ADALINE Training Curve", "Cost")
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                # Decision boundary
                st.subheader("Decision Boundary")
                fig_boundary = plot_decision_boundary_2d(X_and, y_and, 'adaline', 
                                                       "ADALINE Decision Boundary - AND Gate", 
                                                       weights=w_and)
                if fig_boundary:
                    st.plotly_chart(fig_boundary, use_container_width=True)
        
        else:  # Diabetes classification
            st.subheader("🏥 Diabetes Classification with ADALINE")
            
            # Load data
            data, is_real = load_diabetes_data()
            
            # Parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                lr_diabetes = st.slider("Learning Rate", 0.0001, 0.01, 0.0005, 0.0001)
            with col2:
                epochs_diabetes = st.slider("Epochs", 20, 200, 50, 10)
            with col3:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            
            if st.button("🚀 Train ADALINE on Diabetes Data"):
                # Prepare data
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Standardize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                with st.spinner("Training ADALINE..."):
                    w_diabetes, costs = adaline_fit(X_train_scaled, y_train, 
                                                  lr=lr_diabetes, n_epochs=epochs_diabetes)
                    y_pred = adaline_predict(X_test_scaled, w_diabetes)
                
                # Results
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Training curve
                    fig_cost = plot_training_curve(costs, "ADALINE Training Curve", "Cost")
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col2:
                    # Confusion matrix
                    fig_cm = plot_confusion_matrix(y_test, y_pred, "ADALINE - Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
    
    else:  # MADALINE Simulator
        st.markdown('<h2 class="sub-header">🎛️ Program 3: MADALINE Simulator</h2>', unsafe_allow_html=True)
        
        # Configuration
        st.subheader("Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox("Model Type", ["Perceptron", "ADALINE", "MADALINE"])
            dataset_type = st.selectbox("Dataset", ["Synthetic", "Diabetes"])
        
        with col2:
            activation = st.selectbox("Activation Function", 
                                    ["step", "linear", "sigmoid", "tanh", "relu"])
            n_hidden = st.slider("Hidden Neurons (MADALINE only)", 2, 10, 3)
        
        with col3:
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            epochs = st.slider("Epochs", 50, 500, 100, 25)
        
        # Dataset parameters for synthetic data
        if dataset_type == "Synthetic":
            st.subheader("Synthetic Dataset Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_samples = st.slider("Number of Samples", 100, 1000, 600, 50)
            with col2:
                n_features = st.slider("Number of Features", 2, 8, 2)
            with col3:
                noise = st.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)
        
        if st.button("🚀 Train Model"):
            # Prepare dataset
            if dataset_type == "Synthetic":
                X, y = create_synthetic_data(n_samples, n_features, noise)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                data, _ = load_diabetes_data()
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            with st.spinner(f"Training {model_type}..."):
                if model_type == "Perceptron":
                    weights, errors = perceptron_fit(X_train_scaled, y_train, lr=learning_rate, epochs=epochs)
                    y_pred = perceptron_predict(X_test_scaled, weights)
                    training_metric = errors
                    metric_name = "Errors per Epoch"
                    
                elif model_type == "ADALINE":
                    weights, costs = adaline_fit(X_train_scaled, y_train, lr=learning_rate, n_epochs=epochs)
                    y_pred = adaline_predict(X_test_scaled, weights)
                    training_metric = costs
                    metric_name = "Cost"
                    
                else:  # MADALINE
                    W1, W2, costs = madaline_fit(X_train_scaled, y_train, lr=learning_rate, 
                                               epochs=epochs, act=activation, n_hidden=n_hidden)
                    y_pred = madaline_predict(X_test_scaled, W1, W2, act=activation)
                    training_metric = costs
                    metric_name = "Cost"
            
            # Results
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Test Accuracy", f"{accuracy:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Model summary
                st.subheader("Model Summary")
                st.write(f"**Model:** {model_type}")
                st.write(f"**Dataset:** {dataset_type}")
                st.write(f"**Activation:** {activation}")
                if model_type == "MADALINE":
                    st.write(f"**Hidden Neurons:** {n_hidden}")
                st.write(f"**Learning Rate:** {learning_rate}")
                st.write(f"**Epochs:** {epochs}")
            
            with col2:
                # Training curve
                fig_training = plot_training_curve(training_metric, f"{model_type} Training Curve", metric_name)
                st.plotly_chart(fig_training, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_test, y_pred, f"{model_type} - Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Decision boundary for 2D data
            if X_train_scaled.shape[1] == 2:
                st.subheader("Decision Boundary")
                if model_type == "Perceptron":
                    fig_boundary = plot_decision_boundary_2d(X_train_scaled, y_train, 'perceptron',
                                                           f"{model_type} Decision Boundary", weights=weights)
                elif model_type == "ADALINE":
                    fig_boundary = plot_decision_boundary_2d(X_train_scaled, y_train, 'adaline',
                                                           f"{model_type} Decision Boundary", weights=weights)
                else:
                    fig_boundary = plot_decision_boundary_2d(X_train_scaled, y_train, 'madaline',
                                                           f"{model_type} Decision Boundary", 
                                                           W1=W1, W2=W2, act=activation)
                
                if fig_boundary:
                    st.plotly_chart(fig_boundary, use_container_width=True)
            
            # Feature importance (for diabetes dataset)
            if dataset_type == "Diabetes" and model_type in ["Perceptron", "ADALINE"]:
                st.subheader("Feature Importance")
                feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                               'insulin', 'bmi', 'diabetes_pedigree', 'age']
                
                if model_type == "Perceptron":
                    importance = np.abs(weights[1:])  # Exclude bias
                else:
                    importance = np.abs(weights[1:])  # Exclude bias
                
                fig_importance = px.bar(x=feature_names, y=importance,
                                      title=f"{model_type} Feature Importance",
                                      labels={'x': 'Features', 'y': 'Absolute Weight'})
                st.plotly_chart(fig_importance, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Lab-2: Implementation of Adaline with Dataset and Simulator**")
    st.markdown("*Created by R Abhijit Srivathsan - 2448044*")

if __name__ == "__main__":
    main()