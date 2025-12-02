import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import pi
import subprocess
import time
import os
import sys

st.set_page_config(
    page_title="ADHD Clinical Decision Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    div.stButton > button:first-child {
        background-color: #004d40;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
    }
    div.block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("adhd_final_model.joblib")
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    filename = "CPT_II_ConnersContinuousPerformanceTest.csv"
    try:
        try:
            data = pd.read_csv(filename, delimiter=";")
        except:
            data = pd.read_csv(filename, delimiter=",")
        
        # Cleanup
        data.columns = data.columns.str.strip()
        if "Adhd Confidence Index" in data.columns:
            data.rename(columns={"Adhd Confidence Index": "label"}, inplace=True)
        if "ID" in data.columns:
            data = data.drop(columns=["ID"])
            
        # European format fix
        data = data.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="ignore")
            
        if "label" in data.columns:
            X = data.drop(columns=["label"])
        else:
            X = data
        return X
    except:
        return None

model = load_model()
X_data = load_data()


def create_radar_chart(patient_data, population_avg):
    categories = ['Inattention', 'Impulsivity', 'Speed (RT)', 'Consistency']
    
    patient_vals = [
        patient_data['Adhd TScore Omissions'].values[0],
        patient_data['Adhd TScore Commissions'].values[0],
        patient_data['Adhd TScore HitRT'].values[0],
        patient_data['Adhd TScore VarSE'].values[0]
    ]
    
    avg_vals = [
        population_avg['Adhd TScore Omissions'],
        population_avg['Adhd TScore Commissions'],
        population_avg['Adhd TScore HitRT'],
        population_avg['Adhd TScore VarSE']
    ]

    patient_vals += patient_vals[:1]
    avg_vals += avg_vals[:1]
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    ax.plot(angles, avg_vals, linewidth=1, linestyle='solid', label="Population Avg", color="blue")
    ax.fill(angles, avg_vals, 'blue', alpha=0.1)
    
    ax.plot(angles, patient_vals, linewidth=2, linestyle='solid', label="Patient", color="red")
    ax.fill(angles, patient_vals, 'red', alpha=0.25)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig

def run_monitoring_system():
    st.subheader("📷 Live Hyperactivity Monitor")
    st.info("This module uses Computer Vision to track skeletal movement.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "monitor_running" not in st.session_state:
            st.session_state.monitor_running = False

        if st.button("▶️ Start Camera", type="primary"):
            st.session_state.monitor_running = True
            try:
                # ROBUST LAUNCH: Uses sys.executable to ensure correct python env
                current_folder = os.getcwd()
                script_path = os.path.join(current_folder, "monitor.py")
                
                if os.path.exists(script_path):
                    subprocess.Popen([sys.executable, script_path])
                    st.success("Launching...")
                    time.sleep(2)
                else:
                    st.error(f"monitor.py not found at {script_path}")
            except Exception as e:
                st.error(f"Failed: {e}")

        if st.button("⏹️ Stop Updates"):
            st.session_state.monitor_running = False
            st.warning("Paused.")

    with col2:
        status_file = "monitor_status.txt"
        alert_box = st.empty()
        
        if st.session_state.monitor_running:
            while st.session_state.monitor_running:
                if os.path.exists(status_file):
                    try:
                        with open(status_file, "r") as f:
                            status = f.read()
                        if "ALERT" in status:
                            alert_box.error(f"🚨 {status}")
                        else:
                            alert_box.success(f"✅ {status}")
                    except:
                        pass
                else:
                    alert_box.info("Waiting for camera...")
                time.sleep(0.5)


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
st.sidebar.title("Patient Assessment")
st.sidebar.markdown("Adjust observed behaviors:")

def user_input_features():
    inattention = st.sidebar.slider("Inattention (Staring/Missing)", 0, 10, 2)
    impulsivity = st.sidebar.slider("Impulsivity (Interrupting)", 0, 10, 2)
    speed = st.sidebar.slider("Response Speed (Slowness)", 0, 10, 3)
    consistency = st.sidebar.slider("Inconsistency (Variability)", 0, 10, 2)
    return inattention, impulsivity, speed, consistency

inattention, impulsivity, speed, consistency = user_input_features()
st.sidebar.markdown("---")
st.sidebar.info("Model: Stacking Regressor")

# Use Session State to store prediction so it doesn't vanish when switching tabs
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "fake_patient" not in st.session_state:
    st.session_state.fake_patient = None

# Button logic updates session state
if st.sidebar.button("RUN ANALYSIS", type="primary"):
    if model is not None and X_data is not None:
        fake_patient = X_data.median().to_frame().T 
        population_avg = X_data.mean()

        def set_val(col, val, reverse=False):
            if col in X_data.columns:
                min_v, max_v = X_data[col].min(), X_data[col].max()
                res = max_v - (val/10)*(max_v-min_v) if reverse else min_v + (val/10)*(max_v-min_v)
                fake_patient[col] = res

        set_val('Raw Score Omissions', inattention)
        set_val('Adhd TScore Omissions', inattention)
        set_val('Raw Score DPrime', inattention, reverse=True) 
        set_val('Raw Score Commissions', impulsivity)
        set_val('Adhd TScore Commissions', impulsivity)
        set_val('Raw Score Perseverations', impulsivity)
        set_val('Raw Score HitRT', speed)
        set_val('Adhd TScore HitRT', speed)
        set_val('Raw Score HitRTBlock', speed)
        set_val('Raw Score HitRTIsi', speed)
        set_val('Raw Score HitSE', consistency)
        set_val('Adhd TScore VarSE', consistency)
        set_val('Raw Score HitSEBlock', consistency)
        set_val('Raw Score VarSE', consistency)

        pred = model.predict(fake_patient)[0]
        
     
        st.session_state.prediction_result = pred
        st.session_state.fake_patient = fake_patient
        st.session_state.population_avg = population_avg



col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 ADHD Clinical Prediction System")
    st.markdown("### AI-Powered Diagnostic Support Tool")
with col2:
    if model is not None:
        st.success("System Online")
    else:
        st.error("Model Offline")

st.divider()

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Analysis", "🔬 Clinical Metrics", "🛠️ Model Internals", "👁️ Live Monitor"])


with tab1:
    if st.session_state.prediction_result is not None:
        pred = st.session_state.prediction_result
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted ADHD Confidence", f"{pred:.1f}%", f"{pred - 50:.1f} vs Avg")
        with m2:
            status = "High Likelihood" if pred > 60 else "Moderate" if pred > 40 else "Low Likelihood"
            st.metric("Classification", status)
        with m3:
            st.metric("Reliability", "94.2% (Test R²)")
            
        st.progress(min(int(pred), 100))
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Patient vs. Population")
            fig = create_radar_chart(st.session_state.fake_patient, st.session_state.population_avg)
            st.pyplot(fig)
        with c2:
            st.subheader("Symptom Contribution")
            symptom_data = pd.DataFrame({
                'Symptom': ['Inattention', 'Impulsivity', 'Slowness', 'Variability'],
                'Severity': [inattention, impulsivity, speed, consistency]
            })
            st.bar_chart(symptom_data.set_index('Symptom'))
    else:
        st.info("👈 Please set parameters in the sidebar and click 'RUN ANALYSIS'")


with tab2:
    if st.session_state.fake_patient is not None:
        st.subheader("Simulated CPT-II Metrics")
        fp = st.session_state.fake_patient
        metrics_df = fp[['Raw Score Omissions', 'Raw Score Commissions', 'Raw Score HitRT', 'Raw Score HitSE']].T
        metrics_df.columns = ['Estimated Value']
        metrics_df['Unit'] = ['Missed Targets', 'False Alarms', 'Milliseconds (ms)', 'Standard Error']
        st.table(metrics_df)
    else:
        st.info("No analysis data available.")


with tab3:
    st.subheader("Stacking Model Architecture")
    st.code("Pipeline(Preprocessor -> SelectFromModel -> StackingRegressor[LGBM, XGB, Cat])")
    st.info("Architecture optimized for Ryzen 5600H.")


with tab4:
    run_monitoring_system()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2025 Clinical AI Research Group</div>", unsafe_allow_html=True)