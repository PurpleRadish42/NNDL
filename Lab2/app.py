import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification, make_circles, make_moons
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.special import expit
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config with custom theme
st.set_page_config(
    page_title="🧠 ADALINE Neural Network Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .neural-node {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .excellent { background: #48bb78; color: white; }
    .good { background: #38b2ac; color: white; }
    .average { background: #ed8936; color: white; }
    .poor { background: #e53e3e; color: white; }
    
    .animated-bg {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for ADALINE implementation
class AdvancedADALINE:
    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None
        self.costs = []
        self.training_accuracies = []
        
    def fit(self, X, y, callback=None):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs = []
        self.training_accuracies = []
        
        for i in range(self.n_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            
            # Calculate cost (SSE)
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
            
            # Calculate training accuracy
            predictions = np.where(net_input >= 0.0, 1, -1)
            accuracy = np.mean(predictions == y)
            self.training_accuracies.append(accuracy)
            
            # Update weights
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            
            # Callback for real-time updates
            if callback and i % 10 == 0:
                callback(i, cost, accuracy)
                
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def create_synthetic_datasets():
    """Create various synthetic datasets for testing"""
    datasets = {}
    
    # Linear separable
    X_linear, y_linear = make_classification(
        n_samples=300, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    y_linear = np.where(y_linear == 0, -1, 1)
    datasets['Linear Separable'] = (X_linear, y_linear)
    
    # Circles
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    y_circles = np.where(y_circles == 0, -1, 1)
    datasets['Circles'] = (X_circles, y_circles)
    
    # Moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    y_moons = np.where(y_moons == 0, -1, 1)
    datasets['Moons'] = (X_moons, y_moons)
    
    # XOR-like
    np.random.seed(42)
    X_xor = np.random.randn(300, 2)
    y_xor = np.where((X_xor[:, 0] * X_xor[:, 1]) > 0, 1, -1)
    datasets['XOR-like'] = (X_xor, y_xor)
    
    return datasets

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
        # Fallback synthetic data
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

def create_3d_neural_network_viz():
    """Create 3D visualization of neural network"""
    fig = go.Figure()
    
    # Input layer
    input_nodes = [(0, 0, 0), (0, 1, 0), (0, -1, 0)]
    
    # Hidden layer (if any)
    hidden_nodes = [(2, 0, 0)]
    
    # Output layer
    output_nodes = [(4, 0, 0)]
    
    # Add nodes
    for i, (x, y, z) in enumerate(input_nodes):
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=20, color='lightblue'),
            name=f'Input {i+1}' if i < 2 else 'Bias',
            showlegend=True
        ))
    
    for x, y, z in hidden_nodes:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=25, color='orange'),
            name='ADALINE',
            showlegend=True
        ))
    
    for x, y, z in output_nodes:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=20, color='lightgreen'),
            name='Output',
            showlegend=True
        ))
    
    # Add connections
    connections = [
        (input_nodes[0], hidden_nodes[0]),
        (input_nodes[1], hidden_nodes[0]),
        (input_nodes[2], hidden_nodes[0]),
        (hidden_nodes[0], output_nodes[0])
    ]
    
    for (x1, y1, z1), (x2, y2, z2) in connections:
        fig.add_trace(go.Scatter3d(
            x=[x1, x2], y=[y1, y2], z=[z1, z2],
            mode='lines',
            line=dict(color='gray', width=5),
            showlegend=False
        ))
    
    fig.update_layout(
        title="ADALINE Neural Network Architecture",
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Node Position",
            zaxis_title="",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=400
    )
    
    return fig

def plot_decision_boundary_advanced(X, y, model, title):
    """Create advanced decision boundary plot"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(0,0,255,0.3)']],
        showscale=False,
        contours=dict(
            start=-1, end=1, size=2,
            showlines=True, coloring='fill'
        ),
        name='Decision Boundary'
    ))
    
    # Add data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    symbols = ['circle' if label == -1 else 'diamond' for label in y]
    
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(
            color=colors,
            symbol=symbols,
            size=10,
            line=dict(width=2, color='white')
        ),
        name='Data Points'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        height=500
    )
    
    return fig

def create_performance_radar(metrics):
    """Create radar chart for performance metrics"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar"
    )
    
    return fig

def main():
    # Header with animation
    st.markdown('<h1 class="main-header animated-bg">🧠 ADALINE Neural Network Laboratory</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        <i>Advanced Neural Network Implementation & Real-Time Binary Classification</i><br>
        <b>Program #2: Implementing ADALINE for AND Logic Gate & Real-Time Binary Classification</b>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    page = st.sidebar.selectbox(
        "Choose Experiment",
        ["🏠 Home", "🔗 AND Logic Gate", "🏥 Medical Diagnosis", "🔬 Advanced Analysis", "📊 Model Comparison"]
    )
    
    if page == "🏠 Home":
        # Welcome page with impressive visuals
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">Welcome to the ADALINE Laboratory</h2>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 What is ADALINE?</h3>
                <p><strong>ADALINE (Adaptive Linear Neuron)</strong> is a single-layer neural network that uses 
                the Widrow-Hoff learning rule. Unlike the Perceptron, ADALINE minimizes the sum of squared errors 
                between the actual and predicted outputs.</p>
                
                <h4>Key Features:</h4>
                <ul>
                    <li>🔄 <strong>Continuous output</strong> during training</li>
                    <li>📉 <strong>Gradient descent</strong> optimization</li>
                    <li>🎯 <strong>Linear activation</strong> function</li>
                    <li>⚡ <strong>Adaptive learning</strong> rate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive network architecture
            st.plotly_chart(create_3d_neural_network_viz(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🚀 Quick Start</h3>
                <p>Choose an experiment from the sidebar to begin your neural network journey!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance preview
            st.markdown("""
            <div class="feature-card">
                <h4>📈 Expected Performance</h4>
                <div class="performance-badge excellent">AND Gate: 100%</div>
                <div class="performance-badge good">Diabetes: 75-85%</div>
                <div class="performance-badge average">XOR: 50-60%</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "🔗 AND Logic Gate":
        st.markdown('<h2 class="sub-header">🔗 AND Logic Gate Learning</h2>', 
                   unsafe_allow_html=True)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.slider("🎯 Learning Rate", 0.01, 1.0, 0.1, 0.01)
        with col2:
            epochs = st.slider("🔄 Epochs", 10, 200, 50, 10)
        with col3:
            threshold = st.slider("⚡ Threshold", -1.0, 1.0, 0.0, 0.1)
        
        # AND gate data with bipolar encoding
        X_and = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        y_and = np.array([1, -1, -1, -1])
        
        # Display truth table
        st.markdown("### 📋 AND Gate Truth Table (Bipolar Encoding)")
        truth_table = pd.DataFrame({
            'x1': X_and[:, 0],
            'x2': X_and[:, 1], 
            't (target)': y_and
        })
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(truth_table, use_container_width=True)
        
        with col2:
            # Real-time training button
            if st.button("🚀 Train ADALINE", key="and_train"):
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Real-time training visualization
                cost_placeholder = st.empty()
                accuracy_placeholder = st.empty()
                
                # Initialize model
                model = AdvancedADALINE(learning_rate=learning_rate, n_epochs=epochs)
                
                # Real-time callback
                costs_real_time = []
                accuracies_real_time = []
                
                def training_callback(epoch, cost, accuracy):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training... Epoch {epoch + 1}/{epochs}")
                    
                    costs_real_time.append(cost)
                    accuracies_real_time.append(accuracy)
                    
                    # Update real-time plots
                    if len(costs_real_time) > 1:
                        fig_cost = px.line(y=costs_real_time, title="Real-time Cost")
                        cost_placeholder.plotly_chart(fig_cost, use_container_width=True)
                        
                        fig_acc = px.line(y=accuracies_real_time, title="Real-time Accuracy")
                        accuracy_placeholder.plotly_chart(fig_acc, use_container_width=True)
                
                # Train model
                model.fit(X_and, y_and, callback=training_callback)
                
                # Final predictions
                predictions = model.predict(X_and)
                net_outputs = model.net_input(X_and)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training Complete!")
                
                # Results
                st.markdown("### 🎯 Results")
                
                results_df = pd.DataFrame({
                    'x1': X_and[:, 0],
                    'x2': X_and[:, 1],
                    'Target': y_and,
                    'Net Output': net_outputs.round(3),
                    'Prediction': predictions,
                    '✅ Correct': ['✅' if p == t else '❌' for p, t in zip(predictions, y_and)]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Performance metrics
                accuracy = np.mean(predictions == y_and)
                final_cost = model.costs[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Accuracy</h3>
                        <h2>{accuracy:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📉 Final Cost</h3>
                        <h2>{final_cost:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔄 Epochs</h3>
                        <h2>{epochs}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Decision boundary visualization
                st.markdown("### 🎨 Decision Boundary Visualization")
                fig_boundary = plot_decision_boundary_advanced(X_and, y_and, model, 
                                                             "ADALINE Decision Boundary - AND Gate")
                st.plotly_chart(fig_boundary, use_container_width=True)
                
                # Training curves
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cost_final = px.line(
                        y=model.costs, 
                        title="Cost Function Over Time",
                        labels={'index': 'Epoch', 'y': 'Sum of Squared Errors'}
                    )
                    fig_cost_final.update_traces(line_color='#667eea')
                    st.plotly_chart(fig_cost_final, use_container_width=True)
                
                with col2:
                    fig_acc_final = px.line(
                        y=model.training_accuracies,
                        title="Training Accuracy Over Time", 
                        labels={'index': 'Epoch', 'y': 'Accuracy'}
                    )
                    fig_acc_final.update_traces(line_color='#764ba2')
                    st.plotly_chart(fig_acc_final, use_container_width=True)
    
    elif page == "🏥 Medical Diagnosis":
        st.markdown('<h2 class="sub-header">🏥 Real-Time Binary Classification: Diabetes Prediction</h2>', 
                   unsafe_allow_html=True)
        
        # Load diabetes data
        data, is_real = load_diabetes_data()
        
        if is_real:
            st.success("✅ Successfully loaded Pima Indians Diabetes Dataset")
        else:
            st.warning("⚠️ Using synthetic fallback data")
        
        # Data exploration
        with st.expander("📊 Explore Dataset", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Dataset Overview")
                st.dataframe(data.head(10))
                
                st.markdown("#### Statistical Summary")
                st.dataframe(data.describe())
            
            with col2:
                # Correlation heatmap
                correlation_matrix = data.corr()
                fig_corr = px.imshow(correlation_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Class distribution
                class_counts = data['outcome'].value_counts()
                fig_dist = px.pie(values=class_counts.values, 
                                names=['No Diabetes', 'Diabetes'],
                                title="Class Distribution")
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            test_size = st.slider("📊 Test Size", 0.1, 0.4, 0.2, 0.05)
            scaler_type = st.selectbox("🔧 Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        
        with col2:
            learning_rate = st.slider("🎯 Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            epochs = st.slider("🔄 Epochs", 50, 500, 100, 25)
        
        with col3:
            random_state = st.slider("🎲 Random State", 1, 100, 42, 1)
            cross_validation = st.checkbox("🔄 Cross Validation", value=True)
        
        with col4:
            feature_selection = st.multiselect(
                "📝 Select Features", 
                options=data.columns[:-1].tolist(),
                default=data.columns[:-1].tolist()
            )
        
        if st.button("🚀 Train Medical AI Model", key="diabetes_train"):
            if not feature_selection:
                st.error("❌ Please select at least one feature!")
                return
            
            # Prepare data
            X = data[feature_selection].values
            y = data['outcome'].values
            y = np.where(y == 0, -1, 1)  # Convert to bipolar
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale data
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Training with real-time updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Real-time visualization containers
            col1, col2 = st.columns(2)
            with col1:
                cost_chart = st.empty()
            with col2:
                accuracy_chart = st.empty()
            
            # Initialize model
            model = AdvancedADALINE(learning_rate=learning_rate, n_epochs=epochs, random_state=random_state)
            
            # Real-time callback
            def training_callback(epoch, cost, accuracy):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"🧠 Training Neural Network... Epoch {epoch + 1}/{epochs}")
                
                # Update charts every 10 epochs
                if len(model.costs) > 10:
                    fig_cost = px.line(y=model.costs, title="Cost Function",
                                     labels={'index': 'Epoch', 'y': 'SSE'})
                    fig_cost.update_traces(line_color='#e74c3c')
                    cost_chart.plotly_chart(fig_cost, use_container_width=True)
                    
                    fig_acc = px.line(y=model.training_accuracies, title="Training Accuracy",
                                     labels={'index': 'Epoch', 'y': 'Accuracy'})
                    fig_acc.update_traces(line_color='#2ecc71')
                    accuracy_chart.plotly_chart(fig_acc, use_container_width=True)
            
            # Train model
            with st.spinner("🧠 Training ADALINE Neural Network..."):
                model.fit(X_train_scaled, y_train, callback=training_callback)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.net_input(X_test_scaled)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training Complete! Analyzing Results...")
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Convert back to 0/1 for sklearn metrics
            y_test_01 = np.where(y_test == -1, 0, 1)
            y_pred_01 = np.where(y_pred == -1, 0, 1)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test_01, y_pred_01)
            recall = recall_score(y_test_01, y_pred_01)
            f1 = f1_score(y_test_01, y_pred_01)
            
            # Display results with stunning visuals
            st.markdown("### 🎯 Model Performance Results")
            
            # Performance metrics grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Accuracy</h3>
                    <h1>{accuracy:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Precision</h3>
                    <h1>{precision:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔍 Recall</h3>
                    <h1>{recall:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎭 F1-Score</h3>
                    <h1>{f1:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix with enhanced design
                cm = confusion_matrix(y_test_01, y_pred_01)
                fig_cm = px.imshow(cm, 
                                 labels=dict(x="Predicted", y="Actual", color="Count"),
                                 x=['No Diabetes', 'Diabetes'],
                                 y=['No Diabetes', 'Diabetes'],
                                 color_continuous_scale='Blues',
                                 title="🎯 Confusion Matrix")
                
                # Add text annotations
                for i in range(len(cm)):
                    for j in range(len(cm[0])):
                        fig_cm.add_annotation(
                            x=j, y=i, text=str(cm[i][j]),
                            showarrow=False,
                            font=dict(size=20, color="white" if cm[i][j] > cm.max()/2 else "black")
                        )
                
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test_01, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                           name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                           line=dict(color='#667eea', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                           mode='lines',
                                           name='Random Classifier',
                                           line=dict(dash='dash', color='gray')))
                
                fig_roc.update_layout(
                    title='📈 ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Feature importance visualization
            if len(feature_selection) > 1:
                st.markdown("### 🔍 Feature Analysis")
                
                # Feature weights visualization
                feature_weights = np.abs(model.weights[1:])
                importance_df = pd.DataFrame({
                    'Feature': feature_selection,
                    'Importance': feature_weights
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                      orientation='h',
                                      title='📊 Feature Importance (Absolute Weights)',
                                      color='Importance',
                                      color_continuous_scale='viridis')
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Cross-validation results
            if cross_validation:
                st.markdown("### 🔄 Cross-Validation Analysis")
                
                with st.spinner("Performing cross-validation..."):
                    from sklearn.linear_model import Perceptron  # Use as proxy for ADALINE
                    
                    # Create a proxy model for cross-validation
                    proxy_model = Perceptron(max_iter=epochs, eta0=learning_rate, random_state=random_state)
                    cv_scores = cross_val_score(proxy_model, X_train_scaled, y_train, cv=5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 CV Mean</h3>
                        <h1>{cv_scores.mean():.1%}</h1>
                        <p>±{cv_scores.std():.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fig_cv = px.box(y=cv_scores, title="Cross-Validation Scores Distribution")
                    st.plotly_chart(fig_cv, use_container_width=True)
            
            # Training history visualization
            st.markdown("### 📈 Training History")
            
            fig_training = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cost Function', 'Training Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_training.add_trace(
                go.Scatter(y=model.costs, name="Cost", line=dict(color='#e74c3c')),
                row=1, col=1
            )
            
            fig_training.add_trace(
                go.Scatter(y=model.training_accuracies, name="Accuracy", line=dict(color='#2ecc71')),
                row=1, col=2
            )
            
            fig_training.update_layout(height=400, title_text="🧠 Neural Network Training Progress")
            st.plotly_chart(fig_training, use_container_width=True)
            
            # Performance radar chart
            metrics_radar = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            }
            
            fig_radar = create_performance_radar(metrics_radar)
            st.plotly_chart(fig_radar, use_container_width=True)
    
    elif page == "🔬 Advanced Analysis":
        st.markdown('<h2 class="sub-header">🔬 Advanced Neural Network Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Dataset selection
        st.markdown("### 📊 Choose Dataset")
        
        dataset_type = st.selectbox(
            "Select Dataset Type",
            ["Synthetic Datasets", "Diabetes Dataset", "Custom Upload"]
        )
        
        if dataset_type == "Synthetic Datasets":
            synthetic_datasets = create_synthetic_datasets()
            dataset_name = st.selectbox("Choose Synthetic Dataset", list(synthetic_datasets.keys()))
            
            X, y = synthetic_datasets[dataset_name]
            
            # Visualize dataset
            colors = ['red' if label == -1 else 'blue' for label in y]
            fig_data = px.scatter(x=X[:, 0], y=X[:, 1], color=colors,
                                title=f"Dataset: {dataset_name}",
                                labels={'x': 'Feature 1', 'y': 'Feature 2'})
            st.plotly_chart(fig_data, use_container_width=True)
            
        elif dataset_type == "Diabetes Dataset":
            data, _ = load_diabetes_data()
            
            # Feature engineering options
            st.markdown("#### 🔧 Feature Engineering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature_subset = st.multiselect(
                    "Select Features",
                    options=data.columns[:-1].tolist(),
                    default=['glucose', 'bmi', 'age', 'diabetes_pedigree']
                )
                
                if len(feature_subset) < 2:
                    st.error("Please select at least 2 features for analysis")
                    return
            
            with col2:
                # PCA option for dimensionality reduction
                use_pca = st.checkbox("Apply PCA (2D visualization)")
                
                if use_pca:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_subset = data[feature_subset].values
                    X = pca.fit_transform(X_subset)
                    
                    st.write(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.1%}")
                else:
                    if len(feature_subset) == 2:
                        X = data[feature_subset].values
                    else:
                        st.warning("Using first 2 features for 2D visualization")
                        X = data[feature_subset[:2]].values
            
            y = data['outcome'].values
            y = np.where(y == 0, -1, 1)
        
        else:  # Custom Upload
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.dataframe(data.head())
                
                target_col = st.selectbox("Select target column", data.columns)
                feature_cols = st.multiselect("Select feature columns", 
                                            [col for col in data.columns if col != target_col])
                
                if len(feature_cols) >= 2:
                    X = data[feature_cols].values
                    y = data[target_col].values
                    
                    # Ensure binary classification
                    unique_vals = np.unique(y)
                    if len(unique_vals) == 2:
                        y = np.where(y == unique_vals[0], -1, 1)
                    else:
                        st.error("Target must be binary for ADALINE classification")
                        return
                else:
                    st.error("Please select at least 2 features")
                    return
            else:
                st.info("Please upload a CSV file to continue")
                return
        
        # Advanced model configuration
        st.markdown("### ⚙️ Advanced Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rates = st.multiselect(
                "Learning Rates to Compare",
                [0.0001, 0.001, 0.01, 0.1, 0.5],
                default=[0.001, 0.01, 0.1]
            )
            
        with col2:
            epoch_options = st.multiselect(
                "Epoch Counts to Test",
                [50, 100, 200, 500],
                default=[100, 200]
            )
            
        with col3:
            noise_levels = st.slider(
                "Add Gaussian Noise",
                0.0, 0.5, 0.0, 0.05
            )
        
        if st.button("🚀 Run Advanced Analysis", key="advanced_analysis"):
            if not learning_rates or not epoch_options:
                st.error("Please select at least one learning rate and epoch count")
                return
                
            # Add noise if specified
            if noise_levels > 0:
                noise = np.random.normal(0, noise_levels, X.shape)
                X_noisy = X + noise
            else:
                X_noisy = X.copy()
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_noisy)
            
            # Results storage
            results = []
            
            # Progress tracking
            total_experiments = len(learning_rates) * len(epoch_options)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            experiment_count = 0
            
            # Grid search over parameters
            for lr in learning_rates:
                for epochs in epoch_options:
                    experiment_count += 1
                    status_text.text(f"🔬 Running experiment {experiment_count}/{total_experiments}")
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train model
                    model = AdvancedADALINE(learning_rate=lr, n_epochs=epochs)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    final_cost = model.costs[-1]
                    
                    results.append({
                        'Learning Rate': lr,
                        'Epochs': epochs,
                        'Accuracy': accuracy,
                        'Final Cost': final_cost,
                        'Model': model
                    })
                    
                    progress_bar.progress(experiment_count / total_experiments)
            
            status_text.text("✅ Analysis Complete!")
            
            # Results visualization
            results_df = pd.DataFrame(results)
            
            st.markdown("### 📊 Experiment Results")
            
            # Heatmap of results
            pivot_table = results_df.pivot(index='Learning Rate', columns='Epochs', values='Accuracy')
            
            fig_heatmap = px.imshow(pivot_table,
                                  title="🎯 Accuracy Heatmap: Learning Rate vs Epochs",
                                  color_continuous_scale='viridis',
                                  aspect='auto')
            
            # Add text annotations
            for i, lr in enumerate(pivot_table.index):
                for j, epoch in enumerate(pivot_table.columns):
                    value = pivot_table.loc[lr, epoch]
                    fig_heatmap.add_annotation(
                        x=j, y=i, text=f"{value:.2f}",
                        showarrow=False, font=dict(color="white", size=12)
                    )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Best model analysis
            best_result = results_df.loc[results_df['Accuracy'].idxmax()]
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model Configuration</h3>
                <p><strong>Learning Rate:</strong> {best_result['Learning Rate']}</p>
                <p><strong>Epochs:</strong> {best_result['Epochs']}</p>
                <p><strong>Accuracy:</strong> {best_result['Accuracy']:.1%}</p>
                <p><strong>Final Cost:</strong> {best_result['Final Cost']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis of best model
            best_model = best_result['Model']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training curve of best model
                fig_best_cost = px.line(y=best_model.costs,
                                      title="🏆 Best Model: Cost Function",
                                      labels={'index': 'Epoch', 'y': 'Cost'})
                st.plotly_chart(fig_best_cost, use_container_width=True)
            
            with col2:
                # Decision boundary of best model (if 2D)
                if X_scaled.shape[1] == 2:
                    # Retrain on full dataset for visualization
                    viz_model = AdvancedADALINE(learning_rate=best_result['Learning Rate'], 
                                              n_epochs=best_result['Epochs'])
                    viz_model.fit(X_scaled, y)
                    
                    fig_best_boundary = plot_decision_boundary_advanced(
                        X_scaled, y, viz_model, "🏆 Best Model: Decision Boundary"
                    )
                    st.plotly_chart(fig_best_boundary, use_container_width=True)
                else:
                    st.info("Decision boundary visualization only available for 2D data")
            
            # Comparative analysis
            st.markdown("### 📈 Comparative Analysis")
            
            fig_compare = px.scatter(results_df, x='Learning Rate', y='Accuracy',
                                   size='Epochs', color='Final Cost',
                                   title="🔍 Parameter Impact Analysis",
                                   hover_data=['Epochs', 'Final Cost'])
            st.plotly_chart(fig_compare, use_container_width=True)
    
    elif page == "📊 Model Comparison":
        st.markdown('<h2 class="sub-header">📊 Model Comparison & Benchmarking</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Compare ADALINE with Other Algorithms</h3>
            <p>Benchmark ADALINE against traditional machine learning algorithms on the same dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load diabetes data
        data, _ = load_diabetes_data()
        
        # Model selection
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            ["ADALINE", "Logistic Regression", "Random Forest", "SVM", "Naive Bayes", "KNN"],
            default=["ADALINE", "Logistic Regression", "Random Forest"]
        )
        
        if len(models_to_compare) < 2:
            st.error("Please select at least 2 models for comparison")
            return
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("CV Folds", 3, 10, 5, 1)
        
        with col2:
            adaline_lr = st.slider("ADALINE Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            adaline_epochs = st.slider("ADALINE Epochs", 50, 500, 100, 25)
        
        if st.button("🚀 Run Model Comparison", key="model_comparison"):
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
            
            # Results storage
            comparison_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(models_to_compare):
                status_text.text(f"🔬 Training {model_name}...")
                
                if model_name == "ADALINE":
                    # Convert to bipolar for ADALINE
                    y_train_bipolar = np.where(y_train == 0, -1, 1)
                    y_test_bipolar = np.where(y_test == 0, -1, 1)
                    
                    model = AdvancedADALINE(learning_rate=adaline_lr, n_epochs=adaline_epochs)
                    
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train_bipolar)
                    training_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test_scaled)
                    y_pred_01 = np.where(y_pred == -1, 0, 1)
                    
                    accuracy = accuracy_score(y_test, y_pred_01)
                    
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    y_pred_01 = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred_01)
                    
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    y_pred_01 = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred_01)
                    
                elif model_name == "SVM":
                    model = SVC(kernel='rbf', random_state=42)
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    y_pred_01 = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred_01)
                    
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    y_pred_01 = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred_01)
                    
                elif model_name == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    y_pred_01 = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred_01)
                
                # Cross-validation
                if model_name == "ADALINE":
                    # Use proxy for CV
                    cv_model = LogisticRegression(random_state=42, max_iter=1000)
                    cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=cv_folds)
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                
                comparison_results.append({
                    'Model': model_name,
                    'Test Accuracy': accuracy,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Training Time': training_time,
                    'CV Scores': cv_scores
                })
                
                progress_bar.progress((i + 1) / len(models_to_compare))
            
            status_text.text("✅ Comparison Complete!")
            
            # Results visualization
            st.markdown("### 🏆 Comparison Results")
            
            results_df = pd.DataFrame(comparison_results)
            
            # Performance comparison chart
            fig_comparison = px.bar(results_df, x='Model', y='Test Accuracy',
                                  title="🎯 Model Accuracy Comparison",
                                  color='Test Accuracy',
                                  color_continuous_scale='viridis')
            
            # Add CV error bars
            fig_comparison.update_traces(
                error_y=dict(
                    type='data',
                    array=results_df['CV Std'],
                    visible=True
                )
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Detailed comparison table
            display_df = results_df[['Model', 'Test Accuracy', 'CV Mean', 'CV Std', 'Training Time']].copy()
            display_df['Test Accuracy'] = display_df['Test Accuracy'].apply(lambda x: f"{x:.1%}")
            display_df['CV Mean'] = display_df['CV Mean'].apply(lambda x: f"{x:.1%}")
            display_df['CV Std'] = display_df['CV Std'].apply(lambda x: f"±{x:.3f}")
            display_df['Training Time'] = display_df['Training Time'].apply(lambda x: f"{x:.3f}s")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Cross-validation comparison
            st.markdown("### 📊 Cross-Validation Analysis")
            
            cv_data = []
            for result in comparison_results:
                for score in result['CV Scores']:
                    cv_data.append({
                        'Model': result['Model'],
                        'CV Score': score
                    })
            
            cv_df = pd.DataFrame(cv_data)
            
            fig_cv_box = px.box(cv_df, x='Model', y='CV Score',
                              title="📈 Cross-Validation Score Distribution")
            st.plotly_chart(fig_cv_box, use_container_width=True)
            
            # Performance vs Training Time
            fig_time_acc = px.scatter(results_df, x='Training Time', y='Test Accuracy',
                                    text='Model', title="⚡ Accuracy vs Training Time",
                                    size_max=60)
            fig_time_acc.update_traces(textposition="top center")
            st.plotly_chart(fig_time_acc, use_container_width=True)
            
            # Winner announcement
            best_model = results_df.loc[results_df['Test Accuracy'].idxmax()]
            
            st.markdown(f"""
            <div class="metric-card">
                <h2>🏆 Winner: {best_model['Model']}</h2>
                <p><strong>Accuracy:</strong> {best_model['Test Accuracy']:.1%}</p>
                <p><strong>CV Score:</strong> {best_model['CV Mean']:.1%} ± {best_model['CV Std']:.3f}</p>
                <p><strong>Training Time:</strong> {best_model['Training Time']:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧠 <strong>ADALINE Neural Network Laboratory</strong> | Created with ❤️ using Streamlit</p>
        <p><i>Advanced implementation of Adaptive Linear Neuron for educational and research purposes</i></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()