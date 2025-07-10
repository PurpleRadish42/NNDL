import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import json
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="3D Robot Navigation Game - MADALINE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for game-like styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00ff00, #00cc00);
        color: black;
        font-weight: bold;
        border: 2px solid #00ff00;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
        transition: all 0.3s;
        font-family: 'Orbitron', monospace;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 0, 0.8);
    }
    
    .game-title {
        font-family: 'Orbitron', monospace;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00ff00, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        margin-bottom: 20px;
    }
    
    .instruction-box {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid #00ff00;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        font-family: 'Orbitron', monospace;
    }
    
    .sensor-display {
        background: rgba(0, 0, 0, 0.9);
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-weight: bold;
        color: #00ffff;
        box-shadow: inset 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .forward {
        background: linear-gradient(45deg, #00ff00, #00cc00);
        color: black;
        box-shadow: 0 0 30px rgba(0, 255, 0, 0.8);
    }
    
    .turn-left {
        background: linear-gradient(45deg, #00ffff, #0099ff);
        color: black;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.8);
    }
    
    .turn-right {
        background: linear-gradient(45deg, #ff00ff, #cc00cc);
        color: black;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.8);
    }
    
    .score-display {
        font-family: 'Orbitron', monospace;
        font-size: 24px;
        color: #00ff00;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.8);
    }
    
    .health-bar {
        width: 100%;
        height: 30px;
        background: rgba(255, 0, 0, 0.3);
        border: 2px solid #ff0000;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
    }
    
    .health-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ff6600);
        transition: width 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'madaline' not in st.session_state:
    st.session_state.madaline = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'robot_position' not in st.session_state:
    st.session_state.robot_position = [5, 5, 0]  # x, y, z
if 'maze' not in st.session_state:
    st.session_state.maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,0,1,0,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,1],
        [1,1,1,0,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1]
    ])
if 'robot_orientation' not in st.session_state:
    st.session_state.robot_orientation = 0  # 0: North, 1: East, 2: South, 3: West
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'health' not in st.session_state:
    st.session_state.health = 100
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'collectibles' not in st.session_state:
    # Place collectibles randomly on empty spaces
    st.session_state.collectibles = []
    for _ in range(5):
        while True:
            x, y = np.random.randint(1, 10), np.random.randint(1, 10)
            if st.session_state.maze[y, x] == 0 and [x, y] != [5, 5]:
                st.session_state.collectibles.append([x, y])
                break

# JavaScript for keyboard controls
def get_keyboard_control_html():
    return """
    <script>
    const doc = window.parent.document;
    
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown' || 
            e.key === 'ArrowLeft' || e.key === 'ArrowRight' ||
            e.key === 'w' || e.key === 'a' || e.key === 's' || e.key === 'd' ||
            e.key === 'W' || e.key === 'A' || e.key === 'S' || e.key === 'D') {
            e.preventDefault();
            
            let buttonText = '';
            switch(e.key.toLowerCase()) {
                case 'arrowup':
                case 'w':
                    buttonText = '⬆️ Forward';
                    break;
                case 'arrowleft':
                case 'a':
                    buttonText = '⬅️ Turn Left';
                    break;
                case 'arrowright':
                case 'd':
                    buttonText = '➡️ Turn Right';
                    break;
                case 'arrowdown':
                case 's':
                    buttonText = '⬇️ Backward';
                    break;
            }
            
            const buttons = doc.querySelectorAll('button');
            for (let button of buttons) {
                if (button.textContent.includes(buttonText)) {
                    button.click();
                    break;
                }
            }
        }
    });
    </script>
    """

class MADALINE:
    def __init__(self, n_inputs=2, n_adalines=4, learning_rate=0.1):
        self.n_inputs = n_inputs
        self.n_adalines = n_adalines
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights = np.random.randn(n_adalines, n_inputs) * 0.1
        self.bias = np.random.randn(n_adalines) * 0.1
        self.output_weights = np.random.randn(3, n_adalines) * 0.1
        self.output_bias = np.random.randn(3) * 0.1
        
    def adaline_output(self, x):
        net_input = np.dot(self.weights, x) + self.bias
        return np.where(net_input >= 0, 1, -1)
    
    def predict(self, x):
        adaline_outputs = self.adaline_output(x)
        final_outputs = np.dot(self.output_weights, adaline_outputs) + self.output_bias
        return np.argmax(final_outputs)
    
    def train(self, X, y, epochs=100):
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            total_error = 0
            correct = 0
            
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                
                adaline_outputs = self.adaline_output(x)
                prediction = self.predict(x)
                
                error = 1 if prediction != target else 0
                total_error += error
                
                if prediction == target:
                    correct += 1
                
                if error != 0:
                    target_vec = np.zeros(3)
                    target_vec[target] = 1
                    
                    output_errors = target_vec - np.eye(3)[prediction]
                    
                    for j in range(3):
                        self.output_weights[j] += self.learning_rate * output_errors[j] * adaline_outputs
                        self.output_bias[j] += self.learning_rate * output_errors[j]
                    
                    for j in range(self.n_adalines):
                        error_signal = np.sum(output_errors * self.output_weights[:, j])
                        if error_signal != 0:
                            self.weights[j] += self.learning_rate * error_signal * x
                            self.bias[j] += self.learning_rate * error_signal
            
            accuracy = correct / len(X)
            history['loss'].append(total_error / len(X))
            history['accuracy'].append(accuracy)
            
        return history

def generate_training_data():
    X = np.array([
        [0, 0],  # No obstacles -> Forward
        [0, 1],  # Right obstacle -> Turn Left
        [1, 0],  # Left obstacle -> Turn Right
        [1, 1],  # Both obstacles -> Turn Right
    ])
    y = np.array([0, 1, 2, 2])
    return X, y

def create_3d_maze():
    """Create a 3D visualization of the maze"""
    fig = go.Figure()
    
    # Create floor
    floor_x, floor_y = np.meshgrid(range(len(st.session_state.maze[0])), range(len(st.session_state.maze)))
    floor_z = np.zeros_like(floor_x)
    
    # Add floor with checkerboard pattern
    floor_colors = np.zeros_like(floor_x)
    for i in range(len(st.session_state.maze)):
        for j in range(len(st.session_state.maze[0])):
            if st.session_state.maze[i, j] == 0:
                floor_colors[i, j] = (i + j) % 2
    
    fig.add_trace(go.Surface(
        x=floor_x,
        y=floor_y,
        z=floor_z,
        surfacecolor=floor_colors,
        colorscale=[[0, '#1a1a1a'], [0.5, '#2a2a2a'], [1, '#3a3a3a']],
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.5)
    ))
    
    # Create walls
    wall_height = 2
    for i in range(len(st.session_state.maze)):
        for j in range(len(st.session_state.maze[0])):
            if st.session_state.maze[i, j] == 1:
                # Create a 3D cube for each wall
                x = [j, j+1, j+1, j, j, j+1, j+1, j]
                y = [i, i, i+1, i+1, i, i, i+1, i+1]
                z = [0, 0, 0, 0, wall_height, wall_height, wall_height, wall_height]
                
                # Create cube faces
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[0, 0, 0, 0, 4, 4, 1, 2],
                    j=[1, 2, 4, 3, 5, 6, 5, 6],
                    k=[2, 3, 5, 7, 6, 7, 2, 3],
                    color='darkblue',
                    opacity=0.8,
                    lighting=dict(ambient=0.7, diffuse=0.9, specular=0.5),
                    showlegend=False
                ))
    
    # Add robot
    robot_x, robot_y, robot_z = st.session_state.robot_position
    
    # Robot body (sphere)
    fig.add_trace(go.Scatter3d(
        x=[robot_x + 0.5],
        y=[robot_y + 0.5],
        z=[robot_z + 0.5],
        mode='markers',
        marker=dict(
            size=15,
            color='lime',
            symbol='circle',
            line=dict(color='darkgreen', width=2)
        ),
        showlegend=False
    ))
    
    # Robot direction indicator
    direction_map = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
    dx, dy = direction_map[st.session_state.robot_orientation]
    
    fig.add_trace(go.Cone(
        x=[robot_x + 0.5],
        y=[robot_y + 0.5],
        z=[robot_z + 0.8],
        u=[dx],
        v=[dy],
        w=[0],
        sizemode="absolute",
        sizeref=0.3,
        colorscale=[[0, 'yellow'], [1, 'orange']],
        showscale=False
    ))
    
    # Add collectibles
    for collectible in st.session_state.collectibles:
        fig.add_trace(go.Scatter3d(
            x=[collectible[0] + 0.5],
            y=[collectible[1] + 0.5],
            z=[1],
            mode='markers',
            marker=dict(
                size=10,
                color='gold',
                symbol='diamond',
                line=dict(color='orange', width=2)
            ),
            showlegend=False
        ))
    
    # Camera settings for game-like view
    camera = dict(
        eye=dict(x=robot_x + 0.5 - 3*np.sin(np.radians(st.session_state.robot_orientation * 90)), 
                 y=robot_y + 0.5 - 3*np.cos(np.radians(st.session_state.robot_orientation * 90)), 
                 z=3),
        center=dict(x=robot_x + 0.5, y=robot_y + 0.5, z=0.5),
        up=dict(x=0, y=0, z=1)
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, visible=False),
            bgcolor='black',
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

def get_sensor_readings():
    y, x = st.session_state.robot_position[:2]
    orientation = st.session_state.robot_orientation
    
    if orientation == 0:  # North
        left_check = (y, x-1)
        right_check = (y, x+1)
    elif orientation == 1:  # East
        left_check = (y-1, x)
        right_check = (y+1, x)
    elif orientation == 2:  # South
        left_check = (y, x+1)
        right_check = (y, x-1)
    else:  # West
        left_check = (y+1, x)
        right_check = (y-1, x)
    
    left_sensor = 1 if (left_check[0] < 0 or left_check[0] >= len(st.session_state.maze) or 
                       left_check[1] < 0 or left_check[1] >= len(st.session_state.maze[0]) or
                       st.session_state.maze[left_check[0]][left_check[1]] == 1) else 0
    
    right_sensor = 1 if (right_check[0] < 0 or right_check[0] >= len(st.session_state.maze) or 
                        right_check[1] < 0 or right_check[1] >= len(st.session_state.maze[0]) or
                        st.session_state.maze[right_check[0]][right_check[1]] == 1) else 0
    
    return left_sensor, right_sensor

def move_robot(action):
    old_pos = st.session_state.robot_position[:2].copy()
    
    if action == 0:  # Forward
        y, x = st.session_state.robot_position[:2]
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = directions[st.session_state.robot_orientation]
        new_y, new_x = y + dy, x + dx
        
        if (0 <= new_y < len(st.session_state.maze) and 
            0 <= new_x < len(st.session_state.maze[0]) and
            st.session_state.maze[new_y][new_x] == 0):
            st.session_state.robot_position = [new_x, new_y, 0]
            st.session_state.score += 10
            
            # Check for collectibles
            for i, collectible in enumerate(st.session_state.collectibles):
                if collectible == [new_x, new_y]:
                    st.session_state.collectibles.pop(i)
                    st.session_state.score += 100
                    st.session_state.health = min(100, st.session_state.health + 20)
                    break
        else:
            st.session_state.health -= 5
    elif action == 1:  # Turn Left
        st.session_state.robot_orientation = (st.session_state.robot_orientation - 1) % 4
        st.session_state.score += 2
    elif action == 2:  # Turn Right
        st.session_state.robot_orientation = (st.session_state.robot_orientation + 1) % 4
        st.session_state.score += 2
    elif action == 3:  # Backward
        y, x = st.session_state.robot_position[:2]
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = directions[st.session_state.robot_orientation]
        new_y, new_x = y - dy, x - dx
        
        if (0 <= new_y < len(st.session_state.maze) and 
            0 <= new_x < len(st.session_state.maze[0]) and
            st.session_state.maze[new_y][new_x] == 0):
            st.session_state.robot_position = [new_x, new_y, 0]
            st.session_state.score += 5

# Main app
st.markdown('<h1 class="game-title">🤖 NEURAL MAZE NAVIGATOR 3D 🎮</h1>', unsafe_allow_html=True)

# Game instructions
if not st.session_state.game_started:
    st.markdown("""
    <div class="instruction-box">
        <h2 style="color: #00ff00; text-align: center;">🎮 GAME INSTRUCTIONS 🎮</h2>
        <hr style="border-color: #00ff00;">
        
        <h3 style="color: #00ffff;">📖 Story:</h3>
        <p style="color: #ffffff;">You are controlling an AI-powered robot navigating through a mysterious 3D maze. 
        Your robot uses MADALINE neural network to make intelligent decisions based on sensor readings!</p>
        
        <h3 style="color: #00ffff;">🎯 Objective:</h3>
        <ul style="color: #ffffff;">
            <li>Navigate through the maze using your neural network</li>
            <li>Collect golden diamonds for bonus points (+100 points)</li>
            <li>Avoid hitting walls (damages health -5 HP)</li>
            <li>Train your AI to make better decisions</li>
        </ul>
        
        <h3 style="color: #00ffff;">🕹️ Controls:</h3>
        <table style="width: 100%; color: #ffffff;">
            <tr>
                <td><b>W / ↑</b></td><td>Move Forward</td>
                <td><b>A / ←</b></td><td>Turn Left</td>
            </tr>
            <tr>
                <td><b>S / ↓</b></td><td>Move Backward</td>
                <td><b>D / →</b></td><td>Turn Right</td>
            </tr>
        </table>
        
        <h3 style="color: #00ffff;">💡 Tips:</h3>
        <ul style="color: #ffffff;">
            <li>Train the MADALINE network first for intelligent navigation</li>
            <li>Watch the sensor indicators - they show obstacles</li>
            <li>Collect diamonds to restore health</li>
            <li>Use Auto-Navigate for AI-controlled movement</li>
        </ul>
        
        <h3 style="color: #00ffff;">🏆 Scoring:</h3>
        <ul style="color: #ffffff;">
            <li>Forward movement: +10 points</li>
            <li>Turning: +2 points</li>
            <li>Backward movement: +5 points</li>
            <li>Collecting diamonds: +100 points</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 START GAME 🚀", key="start_game"):
            st.session_state.game_started = True
            st.rerun()

else:
    # Inject keyboard controls
    components.html(get_keyboard_control_html(), height=0)
    
    # Game HUD
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="score-display">SCORE: {st.session_state.score}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="score-display">DIAMONDS: {len(st.session_state.collectibles)}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="sensor-display">HEALTH</div>', unsafe_allow_html=True)
        health_percentage = st.session_state.health / 100
        st.markdown(f'''
        <div class="health-bar">
            <div class="health-fill" style="width: {health_percentage * 100}%;"></div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        if st.button("📋 Instructions"):
            st.session_state.game_started = False
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("🎮 CONTROL PANEL")
        
        # Training section
        st.subheader("🧠 Neural Network Training")
        
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            n_adalines = st.slider("ADALINES", 2, 10, 4)
        with col2:
            epochs = st.slider("Epochs", 10, 500, 100, 10)
        
        if st.button("🚀 TRAIN AI"):
            with st.spinner("Training Neural Network..."):
                X, y = generate_training_data()
                st.session_state.madaline = MADALINE(
                    n_inputs=2, 
                    n_adalines=n_adalines, 
                    learning_rate=learning_rate
                )
                history = st.session_state.madaline.train(X, y, epochs)
                st.session_state.training_history = history
                st.success("✅ AI Training Complete!")
        
        # Manual control
        st.subheader("🕹️ Manual Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬆️ Forward", key="forward_btn"):
                move_robot(0)
            if st.button("⬅️ Turn Left", key="left_btn"):
                move_robot(1)
        with col2:
            if st.button("⬇️ Backward", key="backward_btn"):
                move_robot(3)
            if st.button("➡️ Turn Right", key="right_btn"):
                move_robot(2)
        
        if st.button("🔄 RESET GAME"):
            st.session_state.robot_position = [5, 5, 0]
            st.session_state.robot_orientation = 0
            st.session_state.score = 0
            st.session_state.health = 100
            st.session_state.collectibles = []
            for _ in range(5):
                while True:
                    x, y = np.random.randint(1, 10), np.random.randint(1, 10)
                    if st.session_state.maze[y, x] == 0 and [x, y] != [5, 5]:
                        st.session_state.collectibles.append([x, y])
                        break
            st.rerun()
    
    # Main game area
    tab1, tab2, tab3 = st.tabs(["🎮 GAME", "📊 AI STATS", "🧪 TEST AI"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🌐 3D MAZE WORLD")
            maze_fig = create_3d_maze()
            st.plotly_chart(maze_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.subheader("📡 SENSORS")
            
            if st.session_state.madaline:
                left_sensor, right_sensor = get_sensor_readings()
                
                # Sensor visualization
                st.markdown(f'''
                <div class="sensor-display">
                    LEFT SENSOR<br>
                    <span style="font-size: 30px; color: {"#ff0000" if left_sensor else "#00ff00"};">
                        {"⚠️ WALL" if left_sensor else "✅ CLEAR"}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="sensor-display" style="margin-top: 10px;">
                    RIGHT SENSOR<br>
                    <span style="font-size: 30px; color: {"#ff0000" if right_sensor else "#00ff00"};">
                        {"⚠️ WALL" if right_sensor else "✅ CLEAR"}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
                
                # AI Prediction
                sensor_input = np.array([left_sensor, right_sensor])
                prediction = st.session_state.madaline.predict(sensor_input)
                
                action_map = {0: "FORWARD", 1: "TURN LEFT", 2: "TURN RIGHT"}
                action_class_map = {0: "forward", 1: "turn-left", 2: "turn-right"}
                
                action_text = action_map[prediction]
                action_class = action_class_map[prediction]
                
                st.markdown(f'<div class="prediction-box {action_class}">AI SAYS: {action_text}</div>', 
                           unsafe_allow_html=True)
                
                # Auto-move button
                if st.button("🤖 EXECUTE AI"):
                    move_robot(prediction)
                    st.session_state.prediction_history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'left': left_sensor,
                        'right': right_sensor,
                        'action': action_text,
                        'position': st.session_state.robot_position[:2].copy()
                    })
                    st.rerun()
                
                # Auto-navigation
                auto_nav = st.checkbox("🔄 AUTO-PILOT")
                if auto_nav:
                    time.sleep(0.5)
                    move_robot(prediction)
                    st.session_state.prediction_history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'left': left_sensor,
                        'right': right_sensor,
                        'action': action_text,
                        'position': st.session_state.robot_position[:2].copy()
                    })
                    st.rerun()
            else:
                st.warning("⚠️ TRAIN AI FIRST!")
    
    with tab2:
        if st.session_state.training_history:
            st.subheader("📊 AI Training Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss curve with game styling
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=st.session_state.training_history['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='#ff0066', width=3),
                    marker=dict(size=8, color='#ff0066')
                ))
                fig_loss.update_layout(
                    title="Training Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0.8)',
                    plot_bgcolor='rgba(0,0,0,0.9)',
                    font=dict(color='#00ff00', family='Orbitron'),
                    xaxis=dict(gridcolor='rgba(0,255,0,0.2)'),
                    yaxis=dict(gridcolor='rgba(0,255,0,0.2)')
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # Accuracy curve
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=st.session_state.training_history['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#00ff00', width=3),
                    marker=dict(size=8, color='#00ff00')
                ))
                fig_acc.update_layout(
                    title="Training Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0.8)',
                    plot_bgcolor='rgba(0,0,0,0.9)',
                    font=dict(color='#00ff00', family='Orbitron'),
                    xaxis=dict(gridcolor='rgba(0,255,0,0.2)'),
                    yaxis=dict(gridcolor='rgba(0,255,0,0.2)')
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                <div class="sensor-display">
                    FINAL LOSS<br>
                    <span style="font-size: 24px; color: #ff0066;">
                        {st.session_state.training_history['loss'][-1]:.4f}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="sensor-display">
                    ACCURACY<br>
                    <span style="font-size: 24px; color: #00ff00;">
                        {st.session_state.training_history['accuracy'][-1]:.0%}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="sensor-display">
                    EPOCHS<br>
                    <span style="font-size: 24px; color: #00ffff;">
                        {len(st.session_state.training_history['loss'])}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("📊 Train the AI to see analytics")
    
    with tab3:
        st.subheader("🧪 AI Testing Lab")
        
        if st.session_state.madaline:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### Test Sensor Inputs")
                
                test_col1, test_col2 = st.columns(2)
                with test_col1:
                    left_test = st.selectbox("Left Sensor", [0, 1], 
                                            format_func=lambda x: "Clear" if x == 0 else "Wall Detected")
                with test_col2:
                    right_test = st.selectbox("Right Sensor", [0, 1],
                                             format_func=lambda x: "Clear" if x == 0 else "Wall Detected")
                
                if st.button("🔮 TEST AI PREDICTION"):
                    test_input = np.array([left_test, right_test])
                    test_prediction = st.session_state.madaline.predict(test_input)
                    
                    action_map = {0: "FORWARD", 1: "TURN LEFT", 2: "TURN RIGHT"}
                    action_class_map = {0: "forward", 1: "turn-left", 2: "turn-right"}
                    
                    test_action = action_map[test_prediction]
                    test_class = action_class_map[test_prediction]
                    
                    st.markdown(f'<div class="prediction-box {test_class}">AI DECISION: {test_action}</div>', 
                               unsafe_allow_html=True)
            
            # Truth table with game styling
            st.markdown("### 🧠 AI Decision Matrix")
            
            truth_data = []
            for left in [0, 1]:
                for right in [0, 1]:
                    input_vec = np.array([left, right])
                    pred = st.session_state.madaline.predict(input_vec)
                    action_symbols = ['⬆️ FORWARD', '⬅️ LEFT', '➡️ RIGHT']
                    truth_data.append({
                        'Left Sensor': '✅ Clear' if left == 0 else '🚫 Wall',
                        'Right Sensor': '✅ Clear' if right == 0 else '🚫 Wall',
                        'AI Decision': action_symbols[pred]
                    })
            
            truth_df = pd.DataFrame(truth_data)
            
            # Style the dataframe
            st.markdown("""
            <style>
            .dataframe {
                font-family: 'Orbitron', monospace;
                color: #00ff00;
                background-color: rgba(0, 0, 0, 0.8);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(truth_df, use_container_width=True, hide_index=True)
            
            # Navigation history
            if st.session_state.prediction_history:
                st.markdown("### 📜 Recent AI Decisions")
                history_df = pd.DataFrame(st.session_state.prediction_history[-5:])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Train the AI first to access the testing lab!")
    
    # Game Over condition
    if st.session_state.health <= 0:
        st.markdown("""
        <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    background: rgba(255, 0, 0, 0.9); padding: 40px; border-radius: 20px; 
                    border: 3px solid #ff0000; text-align: center; z-index: 1000;">
            <h1 style="color: white; font-family: 'Orbitron', monospace;">GAME OVER</h1>
            <h2 style="color: white; font-family: 'Orbitron', monospace;">Final Score: {}</h2>
            <p style="color: white;">Press Reset Game to play again!</p>
        </div>
        """.format(st.session_state.score), unsafe_allow_html=True)
    
    # Victory condition
    if len(st.session_state.collectibles) == 0:
        st.markdown("""
        <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    background: linear-gradient(45deg, #00ff00, #00ffff); padding: 40px; 
                    border-radius: 20px; border: 3px solid #00ff00; text-align: center; z-index: 1000;">
            <h1 style="color: black; font-family: 'Orbitron', monospace;">🏆 VICTORY! 🏆</h1>
            <h2 style="color: black; font-family: 'Orbitron', monospace;">Final Score: {}</h2>
            <p style="color: black;">All diamonds collected! You are a maze master!</p>
        </div>
        """.format(st.session_state.score), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-family: 'Orbitron', monospace; color: #00ff00;">
    🤖 <b>NEURAL MAZE NAVIGATOR 3D</b> | Powered by MADALINE AI | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
                