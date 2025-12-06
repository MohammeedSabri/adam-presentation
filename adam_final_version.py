import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import base64
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="The Adam Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (GLASSMORPHISM) ---
def load_css(img_b64):
    st.markdown(f"""
    <style>
        /* Main Background */
        .stApp {{
            background-image: url(data:image/webp;base64,{img_b64});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Typography - Force White */
        h1, h2, h3, h4, h5, p, span, li, .stMarkdown, label {{
            color: #FFFFFF !important;
            font-family: 'Helvetica Neue', sans-serif;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
        }}
        
        /* Glassmorphism Cards */
        .glass-card {{
            background: rgba(16, 24, 40, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        
        /* Custom Buttons */
        .stButton>button {{
            background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            height: 3em;
            font-weight: bold;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: rgba(10, 10, 10, 0.9);
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADER ---
def get_base64_image(image_file):
    try:
        with open(image_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# --- 4. DATA CACHING (ML Part) ---
@st.cache_data
def generate_dummy_data(n=400, seed=42):
    """Generates synthetic 3D clusters."""
    np.random.seed(seed)
    # Create 3 distinct clusters
    c1 = np.random.normal(loc=[1, 1, 1], scale=0.8, size=(n//3, 3))
    c2 = np.random.normal(loc=[4, 4, 1], scale=0.8, size=(n//3, 3))
    c3 = np.random.normal(loc=[2, 5, 5], scale=0.8, size=(n//3, 3))
    
    data = np.vstack([c1, c2, c3])
    labels = np.concatenate([['Cluster A']*(n//3), ['Cluster B']*(n//3), ['Cluster C']*(n//3)])
    
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    df['category'] = labels
    return df

@st.cache_resource
def train_model(df, model_type):
    X = df[['x', 'y', 'z']].values
    y_map = {l: i for i, l in enumerate(df['category'].unique())}
    y = df['category'].map(y_map).values
    
    if model_type == "KNN":
        clf = KNeighborsClassifier(n_neighbors=7)
    else:
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
    
    clf.fit(X, y)
    return clf, y_map

# --- LOAD BACKGROUND ---
bg_b64 = get_base64_image('background.webp')
if bg_b64:
    load_css(bg_b64)
else:
    st.warning("⚠️ 'background.webp' not found. Using default dark mode.")

# --- 5. NAVIGATION ---
SLIDES = {
    "01": "🚀 Introduction",
    "02": "📜 History",
    "03": "💡 The Intuition (Deep Dive)",
    "04": "🧮 The Mathematics",
    "05": "🎛️ Interactive Adam: Skewed Valley",
    "06": "🎛️ Interactive Adam v2: The Rosenbrock Valley",
    "07": "💾 The Algorithm (MATLAB Implementation)",
    "08": "🤖 Bonus: 3D Data Clustering",
    "09": "🎯 Conclusion",
    "10": "📚 References",
    "11": "🌐 Terminology"
}

st.sidebar.markdown("## 🧭 Navigation")
selection = st.sidebar.radio("Go to:", list(SLIDES.values()))
curr_key = list(SLIDES.keys())[list(SLIDES.values()).index(selection)]

# --- 6. SLIDE CONTENT ---

# === SLIDE 1: INTRO ===
if curr_key == "01":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("The Adam Optimizer")
    st.markdown("### <span style='color:#4facfe'>Ada</span>ptive <span style='color:#4facfe'>M</span>oment Estimation", unsafe_allow_html=True)
    st.markdown("---")
    st.write("""
    **Welcome.** This is a deep dive into the engine that powers modern AI.
    
    * **The Problem:** Standard Gradient Descent is "memoryless". It slows down on plateaus and oscillates in ravines.
    * **The Solution:** Adam gives the optimizer **Velocity (Momentum)** and **Adaptive Brakes (RMSProp)**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

#===== slide 2 : history ======

# === SLIDE 2: HISTORY ===
elif curr_key == "02":
    with st.container():
        st.markdown('<div class="slide-container">', unsafe_allow_html=True)

        st.header("📜 History & Development of Adam")
        st.markdown("---")
        
        st.subheader("💡 The Original Adam (2015)")
        st.markdown("""
        - **When:** Introduced in **2015** (ICLR) by **Diederik P. Kingma** and **Jimmy Ba** (University of Amsterdam and University of Toronto/Google DeepMind).
        - **Title:** "Adam: A Method for Stochastic Optimization" (arXiv:1412.6980).
        - **Developed For:** Training **Deep Neural Networks** with large parameter counts and high-dimensional spaces.
        - **Initial Impact:** Quickly became the **default optimizer** because it combined the benefits of Momentum and RMSProp, working efficiently and robustly "out of the box."
        """)
        
        st.markdown("---")
        
        st.subheader("🚧 Key Evolutionary Stages")
        st.markdown("""
        - **2018: AMSGrad**
            - **Researchers:** Reddi et al.
            - **Discovery:** Showed that the original Adam could, in rare cases, fail to converge due to increasing effective step sizes.
            - **Fix:** AMSGrad was proposed to ensure the adaptive learning rate component is **non-decreasing**, guaranteeing convergence.
        - **2019: AdamW (The Standard Today) **
            - **Researchers:** Loshchilov & Hutter.
            - **Discovery:** Identified a flaw where applying **L2 regularization** (Weight Decay) inside the Adam update rule was incorrect. The adaptive learning rate effectively diminished the regularization's power.
            - **Fix:** **AdamW** (Adam with Weight decay fix) decouples the L2 regularization, applying it as a separate term.
            - **Result:** AdamW leads to **better generalization** and is the current standard for training many modern large models (like Transformers).
        """)
        
        st.success("Adam and its variants remain foundational, with AdamW being the most commonly used version today.")

        st.markdown('</div>', unsafe_allow_html=True)

# === SLIDE 3: INTUITION (VIDEO INSIGHTS) ===
elif curr_key == "03":
    st.title("The Intuition: Why Adam?")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Adam combines two distinct mechanisms to solve specific failures of Gradient Descent.")
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏎️ Momentum", "⚖️ RMSProp", "🔧 Bias Correction"])
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("1. Momentum: The Heavy Ball")
        st.info("Solves: The Ravine & Local Minima Problem")
        st.write("""
        * **Concept:** Imagine a heavy ball rolling down a hill. It builds up **Velocity**.
        * **Acceleration:** On flat surfaces (plateaus), gradients are small, but momentum keeps the ball moving fast.
        * **Dampening:** In narrow ravines where gradients flip-flop (zig-zag), momentum averages them out, creating a smooth path down the center.
        * **Math:** Exponential Moving Average (EWMA) of gradients.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("2. RMSProp: The Adaptive Brakes")
        st.info("Solves: The Sensitive vs. Stubborn Parameter Problem")
        st.write("""
        * **Concept:** Some parameters (weights) have steep slopes (sensitive), others have flat slopes (stubborn).
        * **The Fix:** RMSProp gives each parameter its **own** learning rate.
        * **Mechanism:** It tracks the "magnitude" of recent gradients.
            * If gradients are **large**, it divides the step size (hits the brakes).
            * If gradients are **small**, it increases the step size (hits the gas).
        * **Why 'Forgetful'?** It uses a moving average so it adapts to the *current* terrain, not the past history.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("3. Bias Correction: The Cold Start")
        st.info("Solves: The Slow Start Problem")
        st.write("""
        * **The Issue:** Adam initializes its memory (Momentum & RMSProp) to 0.
        * **Result:** The first few steps are biased towards 0 (too small/slow).
        * **The Fix:** We artificially boost the estimates at the start using $1/(1-\\beta^t)$.
        * **Effect:** This correction fades away as time ($t$) goes on, leaving the pure algorithm running.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# === SLIDE 3: MATHEMATICS - Interactive Evolution of Optimizers ===
elif curr_key == "04":
    st.title("The Evolution of Gradient Descent: From Memoryless to Adaptive Momentum")
    
    # Add a progress bar to show the evolutionary journey
    st.markdown("### 🚀 Optimization Algorithm Journey")
    concepts = ['Standard GD', 'Momentum', 'AdaGrad', 'Adam']
    concept_index = {'Standard Gradient Descent': 0, 
                     'Gradient Descent with Momentum': 1,
                     'AdaGrad (Adaptive Learning Rate)': 2,
                     'Adam Optimizer': 3}
    
    # Use a radio button for interactive concept switching
    concept = st.radio(
        "Select the Optimization Concept to explore:",
        ('Standard Gradient Descent', 'Gradient Descent with Momentum', 
         'AdaGrad (Adaptive Learning Rate)', 'Adam Optimizer'),
        index=0,
        horizontal=True
    )
    
    # Show progress bar
    progress_val = concept_index[concept] / (len(concepts) - 1)
    st.progress(progress_val, text=f"Evolution Progress: {concepts[concept_index[concept]]}")
    
    st.markdown("---")
    
    # --- 1. Standard Gradient Descent ---
    if concept == 'Standard Gradient Descent':
        st.header("1. Standard Gradient Descent: The Memoryless Step")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(
                """
                ### **The Core Problem: Gradient Descent is Memoryless**  
                At every step it calculates the gradient and takes a new step. That's it.
                
                **As it gets closer to minimum**:  
                - The gradient shrinks  
                - Steps get smaller and smaller  
                - **Result: Painful Deceleration!**
                
                *Like walking blindly toward a target with no sense of previous steps*
                """
            )
        
        with col2:
            st.markdown("#### Visual Analogy")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Gradient_descent.gif/400px-Gradient_descent.gif", 
                    caption="Standard GD: Small steps near minimum")
        
        st.markdown("---")
        
        # Algorithm and Example in columns
        col_algo, col_example = st.columns(2)
        
        with col_algo:
            st.markdown("#### 📝 **Algorithm: Standard GD**")
            st.latex(r"\theta_t = \theta_{t-1} - \alpha \cdot \nabla f(\theta_{t-1})")
            st.code("""
# Pseudo-code
For each iteration:
    gradient = compute_gradient(θ)
    θ = θ - α * gradient  # Memoryless update
            """)
            
            st.markdown("**Where:**")
            st.markdown("- $\\theta_t$: New step")
            st.markdown("- $\\theta_{t-1}$: Current step")  
            st.markdown("- $\\alpha=0.1$: Learning rate (fixed)")
            st.markdown("- $\\nabla f$: Gradient of the function")
        
        with col_example:
            st.markdown("#### 🔢 **Demonstration:** $f(θ) = θ^2$")
            st.latex(r"f(\theta)=\theta^2 \quad \nabla f(\theta)=2\theta")
            
            # Interactive example
            start_theta = st.slider("Starting θ(0)", -20.0, 20.0, 10.0, 0.1)
            alpha = st.slider("Learning Rate α", 0.01, 0.5, 0.1, 0.01)
            
            if st.button("Calculate First Steps", key="gd_calc"):
                # Manual calculations
                grad0 = 2 * start_theta
                theta1 = start_theta - alpha * grad0
                grad1 = 2 * theta1
                theta2 = theta1 - alpha * grad1
                
                st.markdown(f"""
                **Starting at θ(0) = {start_theta} with α = {alpha}**
                
                **Iteration 1:**
                - Gradient = 2 × {start_theta} = **{grad0:.1f}**
                - θ(1) = {start_theta} - {alpha} × {grad0:.1f} = **{theta1:.2f}**
                - Descent: {abs(start_theta - theta1):.2f}
                
                **Iteration 2:**
                - Gradient = 2 × {theta1:.2f} = **{grad1:.2f}**
                - θ(2) = {theta1:.2f} - {alpha} × {grad1:.2f} = **{theta2:.2f}**
                - Descent: {abs(theta1 - theta2):.2f} ⬇️
                
                ⚠️ **Problem:** Descent shrinks from **{abs(start_theta - theta1):.2f}** to **{abs(theta1 - theta2):.2f}**!
                """)
        
        # Problem Highlight
        with st.expander("📉 **The Deceleration Problem - Detailed Analysis**"):
            st.markdown("""
            **Why this happens:**
            1. Near minimum: $\\nabla f(\\theta) \\rightarrow 0$
            2. Update: $\\theta_t = \\theta_{t-1} - \\alpha \\cdot \\text{(tiny gradient)}$
            3. Steps become infinitesimally small
            4. Convergence takes forever!
            
            **Mathematically:**
            $\\theta_{t} - \\theta_{t-1} = -\\alpha \\cdot \\nabla f(\\theta_{t-1})$
            
            As $\\theta \\rightarrow \\theta^*$ (optimum), $\\nabla f(\\theta) \\rightarrow 0$, so steps $\\rightarrow 0$
            """)
            
            # Comparison table
            st.markdown("#### 📊 Comparison at t=4")
            st.markdown("""
            | Algorithm | θ(4) | Progress |
            |-----------|------|----------|
            | Standard GD | 3.277 | Slow! |
            | With Momentum | 0.084 | Much faster! |
            """)

    # --- 2. Gradient Descent with Momentum ---
    elif concept == 'Gradient Descent with Momentum':
        st.header("2. Gradient Descent with Momentum: Adding Velocity")
        
        st.markdown(
            """
            ### **Solution to Deceleration: Give the Algorithm Memory!**  
            *Imagine a ball rolling downhill - it builds up momentum and doesn't slow down.*
            
            **Key Insight:** Add **velocity** term that accumulates past gradients.
            """
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Physics Analogy**")
            st.markdown("""
            **Without Momentum:**
            - Like walking in molasses
            - Each step independent
            - No carry-over energy
            
            **With Momentum:**
            - Like a rolling ball
            - Past motion influences present
            - Energy accumulates
            """)
            
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*_91D77EBB6ESV-9lb6xTjA.gif", 
                    caption="Momentum helps overcome local minima")
        
        with col2:
            st.markdown("#### 📝 **Algorithm: GD with Momentum**")
            st.latex(r"v_t = \beta \cdot v_{t-1} + \nabla f(\theta_{t-1})")
            st.latex(r"\theta_t = \theta_{t-1} - \alpha \cdot v_t")
            
            st.code("""
# Pseudo-code
velocity = 0
for each iteration:
    gradient = compute_gradient(θ)
    velocity = β * velocity + gradient  # Momentum term
    θ = θ - α * velocity  # Update with momentum
            """)
            
            st.markdown("**Where:**")
            st.markdown("- $v_t$: Velocity (accumulated momentum)")
            st.markdown("- $\\beta$: Momentum coefficient (e.g., 0.5, 0.9)")
            st.markdown("- $\\alpha$: Learning rate")
        
        st.markdown("---")
        
        # Interactive Example
        st.markdown("#### 🔢 **Demonstration: Controlled Acceleration**")
        
        col_params, col_calc = st.columns([1, 2])
        
        with col_params:
            start_theta = st.slider("Starting θ(0)", -20.0, 20.0, 10.0, 0.1, key="mom_start")
            alpha = st.slider("Learning Rate α", 0.01, 0.5, 0.1, 0.01, key="mom_alpha")
            beta = st.slider("Momentum β", 0.0, 0.99, 0.5, 0.01, key="mom_beta")
            
            if st.button("Calculate with Momentum", key="mom_calc"):
                # Store calculations
                calculations = []
                theta = start_theta
                velocity = 0
                
                for i in range(3):  # First 3 iterations
                    gradient = 2 * theta
                    velocity = beta * velocity + gradient
                    new_theta = theta - alpha * velocity
                    descent = abs(theta - new_theta)
                    
                    calculations.append({
                        'iter': i+1,
                        'theta': theta,
                        'gradient': gradient,
                        'velocity': velocity,
                        'new_theta': new_theta,
                        'descent': descent
                    })
                    
                    theta = new_theta
        
        with col_calc:
            if 'calculations' in locals():
                st.markdown(f"**Starting at θ(0) = {start_theta}, α = {alpha}, β = {beta}**")
                
                for calc in calculations:
                    st.markdown(f"""
                    **Iteration {calc['iter']}:**
                    - Gradient = 2 × {calc['theta']:.1f} = **{calc['gradient']:.1f}**
                    - Velocity = {beta} × {calc.get('prev_vel', 0):.1f} + {calc['gradient']:.1f} = **{calc['velocity']:.1f}**
                    - θ({calc['iter']}) = {calc['theta']:.1f} - {alpha} × {calc['velocity']:.1f} = **{calc['new_theta']:.2f}**
                    - Descent: **{calc['descent']:.2f}**
                    """)
                    
                    if calc['iter'] > 1:
                        prev_desc = calculations[calc['iter']-2]['descent']
                        if calc['descent'] > prev_desc:
                            st.markdown(f"🚀 **Acceleration:** {prev_desc:.2f} → {calc['descent']:.2f} (+{calc['descent']-prev_desc:.2f})")
                
                # Show comparison
                st.markdown("---")
                st.markdown("""
                #### 🏆 **Comparison Results**
                
                | Metric | Standard GD | GD with Momentum |
                |--------|-------------|------------------|
                | θ(1) | 8.0 | 8.0 |
                | θ(2) | 6.4 | **5.4** |
                | Descent (t=1→2) | 1.6 | **2.6** |
                | θ(4) | 3.277 | **0.084** |
                
                ✅ **Controlled Acceleration Achieved!**
                """)

    # --- 3. AdaGrad (Adaptive Learning Rate) ---
    elif concept == 'AdaGrad (Adaptive Learning Rate)':
        st.header("3. AdaGrad: The Birth of Per-Parameter Adaptation")
        
        st.markdown(
            """
            ### **The New Handicap: Unbalanced Functions**
            
            **Problem:** Single learning rate fails for functions with different sensitivities.
            """
        )
        
        # Unbalanced function example
        col_func, col_problem = st.columns(2)
        
        with col_func:
            st.markdown("#### ⚖️ **The Unbalanced Function**")
            st.latex(r"f(x,y) = 50x^2 + y^2")
            st.latex(r"\nabla f = [100x,\ 2y]")
            
            st.markdown("""
            **Parameter Sensitivities:**
            - **x is "sensitive":** Gradient = $100x$ (50× stronger!)
            - **y is "stubborn":** Gradient = $2y$ (50× weaker!)
            
            *Same learning rate cannot serve both!*
            """)
        
        with col_problem:
            st.markdown("#### ❌ **Why Standard GD Fails Here**")
            
            st.code("""
# Try standard GD with α=0.01
Start: θ(0) = [1.5, 10]

Iteration 1:
gradient = [100*1.5, 2*10] = [150, 20]
update = 0.01 * [150, 20] = [1.5, 0.2]
θ(1) = [0.0, 9.8]  # x overshot to 0!

Iteration 3:
θ(3) = [0.0, 9.22]  # y barely moved!
            """)
            
            st.markdown("""
            **Observation:**
            - α=0.01 perfect for x → reaches 0 in one step
            - Same α crippling small for y → barely moves
            """)
        
        st.markdown("---")
        
        # AdaGrad Solution
        st.markdown("### 🎯 **AdaGrad Solution: Unique Learning Rates**")
        st.markdown("Give each parameter its own learning rate that adapts over time.")
        
        col_algo, col_mech = st.columns(2)
        
        with col_algo:
            st.markdown("#### 📝 **AdaGrad Algorithm**")
            st.latex(r"g_{t}^2 = \sum_{i=1}^{t} (\nabla f_i)^2 \quad \text{(running total of squared gradients)}")
            st.latex(r"\text{adaptive\_lr} = \frac{\alpha}{\sqrt{g_{t}^2 + \epsilon}}")
            st.latex(r"\theta_t = \theta_{t-1} - \text{adaptive\_lr} \cdot \nabla f(\theta_{t-1})")
            
            st.code("""
# Pseudo-code
g_squared = [0, 0, ...]  # One per parameter
for each iteration:
    gradient = compute_gradient(θ)
    g_squared += gradient^2  # Accumulate
    adaptive_lr = α / (sqrt(g_squared) + ε)
    θ = θ - adaptive_lr * gradient
            """)
        
        with col_mech:
            st.markdown("#### 🔧 **How It Works**")
            
            # Interactive demonstration
            st.markdown("**For our unbalanced function:**")
            
            if st.button("Show AdaGrad in Action", key="adagrad_demo"):
                # Simplified calculation
                st.markdown("""
                **Starting:** θ(0) = [1.5, 10], α = 1.5 (aggressive!)
                
                **Iteration 1:**
                - gradient = [150, 20]
                - g_squared = [22500, 400]
                - adaptive_lr_x = 1.5 / sqrt(22500) = **0.01**
                - adaptive_lr_y = 1.5 / sqrt(400) = **0.075**
                - update_x = 0.01 × 150 = 1.5
                - update_y = 0.075 × 20 = 1.5
                - θ(1) = [0.0, 8.5] ✅
                
                **Result at t=4:**
                - Standard GD: θ(4) = [0.0, 9.22]
                - **AdaGrad: θ(4) = [0.0, 6.7]** 🎉
                """)
        
        # The Fatal Flaw
        with st.expander("⚠️ **The Fatal Flaw: Dying Learning Rate Problem**", expanded=True):
            st.markdown("""
            ### 💀 **Why AdaGrad Eventually Fails**
            
            **The Problem:** $g_{t}^2$ only accumulates, never decreases!
            
            **Mathematically:**
            $g_{t}^2 = g_{t-1}^2 + (\nabla f_t)^2$
            
            As $t \\rightarrow \\infty$, $g_{t}^2 \\rightarrow \\infty$
            
            Then: adaptive_lr = $\\frac{\\alpha}{\\sqrt{\\infty + \\epsilon}} \\rightarrow 0$
            
            **Consequence:** All learning rates shrink to zero → training stops prematurely!
            
            **Visualization:**
            ```
            Time:  |---|-----|--------|----------------|-------------------------|
            g²:    | 1 | 10  | 100    | 1000           | 10000                   |
            LR:    | 1 | 0.3 | 0.1    | 0.03           | 0.01 → 0.001 → 0.0001...|
            ```
            
            **Solution:** RMSprop (and later Adam) fix this by using moving averages instead of sums!
            """)

    # --- 4. Adam Optimizer ---
    elif concept == 'Adam Optimizer':
        st.header("4. Adam: Adaptive Moment Estimation - The Ultimate Unifier")
        
        st.markdown(
            """
            ### 🏆 **Adam = Momentum + RMSprop + Bias Correction**
            
            **The Best of Both Worlds:**
            1. **Momentum** for direction/speed
            2. **RMSprop** for per-parameter adaptation (without dying rates)
            3. **Bias Correction** for better early steps
            """
        )
        
        # Adam Overview
        col_overview, col_visual = st.columns([2, 1])
        
        with col_overview:
            st.markdown("#### 🎯 **The Complete Package**")
            st.markdown("""
            **Adam fixes all previous problems:**
            ✅ No more deceleration (Momentum)  
            ✅ No single learning rate (Per-parameter adaptation)  
            ✅ No dying learning rates (RMSprop instead of sum)  
            ✅ No cold start (Bias correction)
            """)
            
            st.metric("Default Hyperparameters", "Well-tuned out of the box!")
            st.markdown("- $\\beta_1 = 0.9$ (Momentum memory)")
            st.markdown("- $\\beta_2 = 0.999$ (Squared gradient memory)")
            st.markdown("- $\\alpha = 0.001$ (Learning rate)")
            st.markdown("- $\\epsilon = 10^{-8}$ (Numerical stability)")
        
        with col_visual:
            st.image("https://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif",
                    caption="Adam navigating complex landscapes")
        
        st.markdown("---")
        
        # The Two Engines
        st.markdown("### ⚙️ **The Two Engines of Adam**")
        
        col_engine1, col_engine2 = st.columns(2)
        
        with col_engine1:
            st.markdown("#### 1. **The Direction Engine (Momentum)**")
            st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t")
            st.markdown("**Exponentially Weighted Moving Average (EWMA) of gradients**")
            
            # EWMA explanation
            with st.expander("📊 **Understanding EWMA**"):
                st.markdown("""
                **Formula:**
                $\\text{average}_t = \\beta \\cdot \\text{average}_{t-1} + (1-\\beta) \\cdot \\text{new\\_value}_t$
                
                **Memory Control:**
                - $\\beta=0.9$: Long memory (reacts slowly)
                - $\\beta=0.1$: Short memory (reacts quickly)
                
                **For Adam:**
                - $\\beta_1=0.9$: Stable direction tracking
                - Tracks the "average gradient direction"
                """)
            
            st.markdown("**Purpose:** Smooths the gradient path, maintains velocity")
        
        with col_engine2:
            st.markdown("#### 2. **The Adaptive Rate Engine (RMSprop)**")
            st.latex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2")
            st.markdown("**EWMA of squared gradients**")
            
            with st.expander("🔧 **How It Fixes AdaGrad**"):
                st.markdown("""
                **AdaGrad Problem:**
                $g_t^2 = \\sum_{i=1}^t (\\nabla f_i)^2$ → grows infinitely
                
                **RMSprop Solution:**
                $v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$ → moving average
                
                **Key Difference:**
                - Old gradients get discounted by $\\beta_2^t$
                - No infinite accumulation
                - Learning rates don't die!
                """)
            
            st.markdown("**Purpose:** Per-parameter learning rates that adapt but don't vanish")
        
        st.markdown("---")
        
        # Bias Correction
        st.markdown("### 🎯 **The Secret Sauce: Bias Correction**")
        
        col_bias, col_why = st.columns(2)
        
        with col_bias:
            st.markdown("#### **The Problem**")
            st.markdown("""
            **Initialization issue:**
            - $m_0 = 0$, $v_0 = 0$
            - Early steps are underestimated
            - Slow start to training
            """)
            
            st.latex(r"E[m_t] = E[g_t] \cdot (1 - \beta_1^t)")
            st.caption("Expected value shows bias toward zero early on")
        
        with col_why:
            st.markdown("#### **The Solution**")
            st.latex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}")
            st.latex(r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}")
            
            st.markdown("**What this does:**")
            st.markdown("""
            1. **Early training (t small):**
               - $1-\\beta^t \\approx 0.1$
               - Division amplifies $m_t$, $v_t$
               - Bigger steps, faster start
            
            2. **Later training (t large):**
               - $1-\\beta^t \\approx 1$
               - No correction needed
               - Correction turns itself off!
            """)
        
        st.markdown("---")
        
        # The Complete Algorithm
        st.markdown("### 🚀 **The Complete Adam Algorithm**")
        
        # Show in a nice code block
        adam_code = """
# ADAM: Adaptive Moment Estimation
# ================================

# 1. INITIALIZATION
m = 0      # First moment vector (momentum)
v = 0      # Second moment vector (adaptive rate)
t = 0      # Timestep counter

# 2. HYPERPARAMETERS
α = 0.001  # Learning rate
β₁ = 0.9   # Momentum decay
β₂ = 0.999 # Squared gradient decay
ε = 1e-8   # Numerical stability

# 3. TRAINING LOOP
for each iteration:
    t = t + 1
    g = compute_gradient(θ)           # Current gradient
    
    # Update biased moment estimates
    m = β₁ * m + (1 - β₁) * g        # Momentum (Direction)
    v = β₂ * v + (1 - β₂) * g²       # Adaptive rate (Magnitude)
    
    # Bias correction
    m̂ = m / (1 - β₁ᵗ)                # Correct momentum bias
    v̂ = v / (1 - β₂ᵗ)                # Correct adaptive rate bias
    
    # The final update
    θ = θ - α * m̂ / (√(v̂) + ε)       # Adam update rule
"""
        
        col_code, col_interpret = st.columns([1, 1])
        
        with col_code:
            st.code(adam_code, language="python")
        
        with col_interpret:
            st.markdown("#### 🎯 **Intuitive Interpretation**")
            st.markdown("""
            **The Update Rule:**
            $\\theta = \\theta - \\alpha \\times \\frac{\\text{momentum}}{\\text{volatility}}$
            
            **Breaking it down:**
            
            1. **Numerator (m̂):**
               - Smoothed gradient direction
               - Momentum from past steps
               - "Where should we go?"
            
            2. **Denominator (√(v̂)):**
               - Adaptive scaling factor
               - Per-parameter adjustment
               - "How big of a step?"
            
            3. **Together:**
               - Sensitive parameters: Small v̂ → bigger steps
               - Stubborn parameters: Large v̂ → smaller steps
               - Consistent momentum across all
            """)
            
            # Final summary
            st.success("""
            **✅ Adam Achieves:**
            - Fast convergence (Momentum)
            - Per-parameter adaptation (AdaGrad/RMSprop)
            - No dying learning rates (RMSprop)
            - Good early training (Bias correction)
            - Default parameters that usually work
            """)
        
        # Comparison Table
        with st.expander("📊 **Algorithm Comparison Summary**"):
            st.markdown("""
            | Algorithm | Key Idea | Solves | Problem |
            |-----------|----------|--------|---------|
            | **Standard GD** | Fixed step | - | Deceleration |
            | **Momentum** | Add velocity | Deceleration | Single learning rate |
            | **AdaGrad** | Per-parameter LR | Single LR | Dying learning rates |
            | **RMSprop** | Moving average | Dying LR | No momentum |
            | **Adam** | Momentum + RMSprop | All above | - |
            """)
    
    # Navigation helper
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        if concept != 'Standard Gradient Descent':
            if st.button("← Previous: Standard GD"):
                st.session_state.concept_index = max(0, concept_index[concept] - 1)
    
    with col_nav2:
        st.markdown(f"**Currently viewing:** {concept}")
    
    with col_nav3:
        if concept != 'Adam Optimizer':
            if st.button(f"Next: {concepts[concept_index[concept] + 1]} →"):
                st.session_state.concept_index = min(3, concept_index[concept] + 1)

# Note: To make navigation work, you might need to add to your session state
if 'concept_index' not in st.session_state:
    st.session_state.concept_index = 0



# === SLIDE 4: INTERACTIVE SKEWED VALLEY (WITH ANIMATION) ===
elif curr_key == "05":
    st.title("🎛️ Interactive: Watch Adam Converge")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Objective:** Minimize the Skewed Valley function: $f(x,y) = 2x^2 + y^2 - xy - 3x - y + 4$")
    st.info("This anisotropic (skewed) surface is difficult for standard algorithms. Watch how Adam navigates it.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar Controls (Menu from Photo)
    with st.sidebar:
        st.markdown("---")
        st.header("🧪 Simulation Lab")
        st.markdown("**Start Position**")
        start_x = st.slider("X₀", -2.0, 4.0, -1.5, 0.1)
        start_y = st.slider("Y₀", -2.0, 4.0, 3.0, 0.1)
        
        st.markdown("**Hyperparameters**")
        lr = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01)
        iterations = st.slider("Iterations", 10, 200, 70)

        st.markdown("**Decay Rates**")
        beta1 = st.slider("Beta 1 (Momentum)", 0.0, 0.999, 0.9, 0.001)
        beta2 = st.slider("Beta 2 (RMSProp)", 0.0, 0.999, 0.999, 0.001)
        
        st.markdown("---")
        st.markdown("**🎞️ Animation Control**")
        # Animation Slider
        replay_step = st.slider("Replay Step", 1, iterations, iterations)
        show_arrows = st.checkbox("Show Flashes (Arrows)", value=True)

    # --- SIMULATION LOGIC ---
    # Skewed Valley Function
    def func(x, y): return 2*x**2 + y**2 - x*y - 3*x - y + 4
    # Gradient of Skewed Valley
    def grad(x, y): return np.array([4*x - y - 3, 2*y - x - 1])

    params = np.array([start_x, start_y])
    m = np.zeros(2)
    v = np.zeros(2)
    epsilon = 1e-8
    
    # History storage [x, y, z, u, v]
    history = []
    
    # Store initial state
    z_init = func(params[0], params[1])
    history.append([params[0], params[1], z_init, 0, 0])

    for t in range(1, iterations + 1):
        g = grad(params[0], params[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Calculate Update Step
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Store state (Arrow points in direction of negative step)
        z_curr = func(params[0], params[1])
        # We store the update step as the vector (u, v) for the arrow
        history.append([params[0], params[1], z_curr, -step[0], -step[1]])
        
        # Apply Update
        params = params - step

    history = np.array(history)

    # --- PLOTTING ---
    # Create Grid for Background Surface
    x_range = np.linspace(-2, 4, 60)
    y_range = np.linspace(-2, 4, 60)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # Slicing for Animation
    view_hist = history[:replay_step]

    fig = go.Figure()
    
    # Surface
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Loss Surface', showscale=False))
    
    # Path (Animated by slider)
    fig.add_trace(go.Scatter3d(
        x=view_hist[:,0], y=view_hist[:,1], z=view_hist[:,2], 
        mode='lines+markers', 
        line=dict(color='red', width=5), 
        marker=dict(size=3, color='yellow'), 
        name='Adam Trajectory'
    ))

    # Flashes (Arrows) - Only show if checkbox checked
    if show_arrows and len(view_hist) > 1:
        # Show arrows for the last 15 steps (to prevent clutter), or all if short path
        start_arrow = max(1, replay_step - 15)
        vecs = view_hist[start_arrow:]
        
        fig.add_trace(go.Cone(
            x=vecs[:,0], y=vecs[:,1], z=vecs[:,2],
            u=vecs[:,3], v=vecs[:,4], w=np.zeros(len(vecs)), # Flat arrows on Z
            sizemode="scaled", sizeref=0.5, anchor="tail",
            colorscale='Reds', showscale=False, name='Flash'
        ))

    # Global Min Marker
    fig.add_trace(go.Scatter3d(x=[1], y=[1], z=[func(1,1)], mode='markers', marker=dict(size=6, color='white', symbol='x'), name='Global Min (1,1)'))

    fig.update_layout(
        title='Adam Optimization Trajectory',
        height=600,
        template="plotly_dark",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Loss', aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===slide: % The rosenbrock valley
elif curr_key == "06":
    st.header("🎛️ Interactive Adam: The Rosenbrock Valley")
    st.markdown("""
    The **Rosenbrock function** (or Banana function) is a famous test because the global minimum is inside a long, narrow, parabolic valley. 
    It is very hard for standard Gradient Descent to navigate, but Adam handles it well.
    """)
    
    # --- Sidebar Controls for this slide ---
    with st.sidebar:
        st.markdown("### Simulation Settings")
        lr = st.slider("Learning Rate", 0.001, 1.0, 0.5, 0.01)
        beta1 = st.slider("Beta 1 (Momentum)", 0.0, 0.99, 0.9)
        beta2 = st.slider("Beta 2 (RMSProp)", 0.0, 0.999, 0.99)
        iterations = st.slider("Iterations", 50, 500, 300)

    # Layout
    col_viz, col_controls = st.columns([3, 1])

    # --- SIMULATION LOGIC: ROSENBROCK ---
    # f(x,y) = (1-x)^2 + 100(y-x^2)^2. Min is at (1,1)
    def rosenbrock(x, y): return (1 - x)**2 + 100 * (y - x**2)**2
    # Gradient of Rosenbrock
    def grad_rosenbrock(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])

    with col_controls:
        st.markdown("**Start Position**")
        start_x = st.number_input("Start X", value=-1.5, step=0.1)
        start_y = st.number_input("Start Y", value=-1.0, step=0.1)
        st.write(f"Global Min is at (1, 1)")

    # Run Optimizer
    params = np.array([start_x, start_y])
    m = np.zeros(2)
    v = np.zeros(2)
    epsilon = 1e-8
    
    path_x, path_y, path_z = [params[0]], [params[1]], [rosenbrock(params[0], params[1])]

    for t in range(1, iterations + 1):
        g = grad_rosenbrock(params[0], params[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update
        params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        path_x.append(params[0])
        path_y.append(params[1])
        path_z.append(rosenbrock(params[0], params[1]))

    # --- PLOTTING ---
    with col_viz:
        # Create Grid
        x_range = np.linspace(-2.5, 2.5, 100)
        y_range = np.linspace(-2, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = rosenbrock(X, Y)
        # Clip Z for better visualization (Rosenbrock shoots up fast)
        Z = np.clip(Z, 0, 2000) 

        fig = go.Figure()
        
        # Surface with contours
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis', opacity=0.7, 
            showscale=False,
            contours = {
                "z": {"show": True, "start": 0, "end": 1000, "size": 50, "color":"white"}
            }
        ))
        
        # The Optimizer Path
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='lines',
            line=dict(color='red', width=6),
            name='Adam Path'
        ))
        
        # Start and End points
        fig.add_trace(go.Scatter3d(
            x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
            mode='markers', marker=dict(size=5, color='blue'), name='Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
            mode='markers', marker=dict(size=8, color='white', symbol='diamond'), name='End'
        ))

        fig.update_layout(
            title='Adam navigating the Rosenbrock Valley',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Loss (Clipped)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


# === SLIDE 7: MATLAB ALGORITHM ===
elif curr_key == "07":
    st.header("💾 The Algorithm (MATLAB Implementation)")
    st.write("A standard implementation logic for educational purposes.")
    
    matlab_code = """
function [theta] = adam_optimizer(grad_func, theta_init, alpha, beta1, beta2, epsilon, max_iters)
    m = zeros(size(theta_init));
    v = zeros(size(theta_init));
    theta = theta_init;
    
    for t = 1:max_iters
        g = grad_func(theta);
        
        % Update biased first moment estimate (Momentum)
        m = beta1 * m + (1 - beta1) * g;
        
        % Update biased second raw moment estimate (RMSProp)
        v = beta2 * v + (1 - beta2) * (g.^2);
        
        % Compute bias-corrected estimates
        m_hat = m / (1 - beta1^t);
        v_hat = v / (1 - beta2^t);
        
        % Update parameters
        theta = theta - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
    end
end
    """
    st.code(matlab_code, language='matlab')


# === SLIDE 8: ML (YOUR REQUESTED 3D KNN/RF CODE) ===
elif curr_key == "08":
    st.header("🤖 Bonus: 3D Data Clustering")
    st.write("Demonstration: generate synthetic 3D clusters, train a classifier, and visualize the decision boundary that separates them.")

    # --- 1. Sidebar Controls (Moved all settings here) ---
    with st.sidebar:
        st.header("⚙️ Clustering Settings")
        st.markdown("---")
        
        # Classifier Choice
        clf_type = st.selectbox("Classifier", ["KNN", "Random Forest"], key="clf_type_3d")
        
        st.markdown("**Hyperparameters**")
        if clf_type == "KNN":
            n_neighbors = st.slider("K (Neighbors)", 1, 15, 7, key="knn_k")
        else:
            n_estimators = st.slider("N (Trees)", 10, 200, 100, key="rf_n")
            max_depth = st.slider("Max Depth", 1, 10, 5, key="rf_depth")
        
        st.markdown("---")
        
        # Visualization Control
        resolution = st.select_slider("Grid Density", options=[10, 15, 20], value=10, key="res_3d")
        st.caption("Higher density = smoother boundary but slower.")
        
        st.markdown("---")
        
        # Training Button (Use session state for explicit control)
        if st.button("🚀 Train & Visualize"):
            st.session_state.train_trigger_3d = True
            st.rerun() # Rerun immediately after button press
        
        if 'train_trigger_3d' not in st.session_state:
            st.session_state.train_trigger_3d = False

    # --- 2. Data Preparation ---
    df = generate_dummy_data()
    X = df[['x', 'y', 'z']].values
    label_map = {lab: i for i, lab in enumerate(df['category'].unique())}
    y_encoded = df['category'].map(label_map).values

    # Check if the model has been trained/triggered
    if st.session_state.train_trigger_3d:
        
        # --- 3. Model Training ---
        with st.spinner(f"Training {clf_type} model..."):
            if clf_type == "KNN":
                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
            clf.fit(X, y_encoded)
        
        acc = accuracy_score(y_encoded, clf.predict(X))
        st.success(f"Model Trained! Training Accuracy: **{acc:.1%}**")

        # --- 4. Generate Decision Grid ---
        with st.spinner(f"Calculating decision boundary at resolution {resolution}x{resolution}x{resolution}..."):
            margin = 1.0
            x_min, x_max = X[:,0].min()-margin, X[:,0].max()+margin
            y_min, y_max = X[:,1].min()-margin, X[:,1].max()+margin
            z_min, z_max = X[:,2].min()-margin, X[:,2].max()+margin

            # Create the 3D grid
            gx = np.linspace(x_min, x_max, resolution)
            gy = np.linspace(y_min, y_max, resolution)
            gz = np.linspace(z_min, z_max, resolution)
            
            grid_pts = np.array(np.meshgrid(gx, gy, gz)).reshape(3, -1).T
            grid_preds = clf.predict(grid_pts)

        # --- 5. Visualization ---
        fig = go.Figure()

        # Define consistent colors
        colors = ['#EF553B', '#00CC96', '#636EFB'] 
        symbols = {'Cluster A': 'circle', 'Cluster B': 'diamond', 'Cluster C': 'square'}
        
        # Plot Decision Regions (Low Opacity)
        for cls_name, cls_idx in label_map.items():
            mask = grid_preds == cls_idx
            pts = grid_pts[mask]
            if len(pts) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode='markers',
                    marker=dict(size=3, color=colors[cls_idx], opacity=0.1),
                    name=f"Region: {cls_name}",
                    showlegend=False
                ))

        # Plot Actual Data Points (High Opacity)
        for cat in df['category'].unique():
            subset = df[df['category'] == cat]
            idx = label_map[cat]
            fig.add_trace(go.Scatter3d(
                x=subset['x'], y=subset['y'], z=subset['z'],
                mode='markers',
                marker=dict(size=6, symbol=symbols[cat], color=colors[idx], line=dict(width=1, color='white'), opacity=1.0),
                name=cat
            ))

        fig.update_layout(
            title=f"3D Clusters & Decision Boundaries ({clf_type})",
            height=700,
            template="plotly_dark",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Initial message before training
        st.info("Adjust the settings in the sidebar and click '🚀 Train & Visualize' to generate the 3D decision boundary plot.")
        st.markdown('<div class="glass-card">Visualizing how an optimized classifier draws decision boundaries in 3D space.</div>', unsafe_allow_html=True)
        st.image("https://placehold.co/700x350/236B8E/FFF?text=3D+Classification+Decision+Boundary", caption="Visualization will appear here after training.")


# === SLIDE 9: CONCLUSION ===
elif curr_key == "09":
    st.header("Conclusion")
    st.write("""
    * **Adam** is the "Swiss Army Knife" of Deep Learning optimization.
    * It combines the speed of **Momentum** with the adaptability of **RMSProp**.
    * While variants like AdamW exist for specific needs, the core Adam algorithm remains the gold standard for starting any new AI project.
    """)
    st.balloons()

# === SLIDE 10: REFERENCES ===
elif curr_key == "10":
    st.header("References")
    st.markdown("""
    1. **Kingma, D., & Ba, J. (2015).** *Adam: A Method for Stochastic Optimization*. ICLR.
    2. **Reddi, S. J., et al. (2018).** *On the Convergence of Adam and Beyond*. ICLR.
    3. **Loshchilov, I., & Hutter, F. (2019).** *Decoupled Weight Decay Regularization*. ICLR.
    4. **Ruder, S. (2016).** *An overview of gradient descent optimization algorithms*.
    """)

# === SLIDE 11: TERMINOLOGY ===
elif curr_key == "11":
    st.title("🌐 Terminology")
    
    data = [
        ["Adam Optimizer", "Adaptive Moment Estimation", "Combines Momentum and RMSProp."],
        ["Momentum", "Velocity / Ball rolling", "Accelerates in consistent directions."],
        ["RMSProp", "Adaptive Brakes / Friction", "Slows down on steep slopes, speeds up on flat ones."],
        ["Bias Correction", "Warm-up", "Fixes the initial slow start (zero-bias)."],
        ["Epoch", "Training Cycle", "One pass through the entire dataset."]
    ]
    df = pd.DataFrame(data, columns=["Term", "Analogy", "Definition"])
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.table(df)
    st.markdown('</div>', unsafe_allow_html=True)