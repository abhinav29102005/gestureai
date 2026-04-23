import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os

# --- UI CONFIG ---
st.set_page_config(page_title="GesturAI Industrial Suite", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;900&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background: #0b0e14; color: #ffffff; }
    .card { background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
    .stat-val { font-size: 2.2rem; font-weight: 900; color: #00f2fe; }
    .stat-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 2px; }
    .theory-section { margin-bottom: 40px; padding: 20px; border-left: 4px solid #58a6ff; background: #0d1117; }
    </style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'frame_idx' not in st.session_state: st.session_state.frame_idx = 0
if 'active' not in st.session_state: st.session_state.active = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("MASTER CONTROL")
    res_files = [f for f in os.listdir("results") if f.endswith(".json")]
    subjects = sorted([f.split("_")[1].replace(".json", "") for f in res_files])
    subject = st.selectbox("SUBJECT ID", subjects)
    latency = st.slider("VISUAL SPEED (DELAY MS)", 1, 500, 20)
    
    st.divider()
    if st.button("START ANALYSIS", use_container_width=True, type="primary"):
        st.session_state.active = True
    if st.button("TERMINATE", use_container_width=True):
        st.session_state.active = False
    if st.button("RESET TIMELINE", use_container_width=True):
        st.session_state.frame_idx = 0; st.session_state.active = False

# --- DATA LOADING ---
@st.cache_data
def load_sub_data(sub_id):
    with open(f"results/sub_{sub_id}.json", "r") as f: return json.load(f)

full_data = load_sub_data(subject)

# --- HEADER ---
st.title("GESTURAI INDUSTRIAL MASTER SUITE")

# --- MASTER TABS ---
tab_analysis, tab_subject, tab_theory, tab_global = st.tabs(["LIVE ANALYTICS", "SUBJECT PROFILE", "THEORY AND SCIENCE", "GLOBAL INTELLIGENCE"])

with tab_analysis:
    if st.session_state.active:
        c1, c2, c3 = st.columns(3)
        act_card = c1.empty(); conf_card = c2.empty(); win_card = c3.empty()
        
        l_col, r_col = st.columns([2, 1])
        with l_col: sig_view = st.empty(); gauge_view = st.empty()
        with r_col: attn_view = st.empty(); xai_view = st.empty()

        for i in range(st.session_state.frame_idx, len(full_data)):
            if not st.session_state.active: break
            frame = full_data[i]
            
            act_card.markdown(f"<div class='card'><div class='stat-label'>Gesture</div><div class='stat-val' style='color:#7ee787'>{frame['activity']}</div></div>", unsafe_allow_html=True)
            conf_card.markdown(f"<div class='card'><div class='stat-label'>Confidence</div><div class='stat-val'>{frame['confidence']*100:.1f}%</div></div>", unsafe_allow_html=True)
            win_card.markdown(f"<div class='card'><div class='stat-label'>Window Index</div><div class='stat-val'>{i}</div></div>", unsafe_allow_html=True)
            
            df_win = pd.DataFrame(frame['raw_window'], columns=['AX','AY','AZ','AR','GX','GY','GZ','GR'])
            sig_view.plotly_chart(px.line(df_win, template="plotly_dark", height=350, title="8-AXIS SENSOR FUSION").update_layout(margin=dict(l=0,r=0,t=40,b=0)), use_container_width=True, key=f"s_{i}")
            gauge_view.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=frame['confidence']*100, gauge={'bar':{'color':"#00f2fe"}, 'axis':{'range':[0,100]}})).update_layout(height=200, template="plotly_dark"), use_container_width=True, key=f"g_{i}")
            attn_view.plotly_chart(px.imshow(np.array(frame['attention']).reshape(1,-1), color_continuous_scale='Magma', template="plotly_dark", title="ATTENTION MAP").update_layout(height=180, coloraxis_showscale=False), use_container_width=True, key=f"a_{i}")
            xai_view.plotly_chart(px.bar(x=df_win.columns, y=frame['local_imp'], template="plotly_dark", title="SENSOR ATTRIBUTION", color_discrete_sequence=['#58a6ff']).update_layout(height=300), use_container_width=True, key=f"x_{i}")
            
            st.session_state.frame_idx = i
            time.sleep(latency / 1000.0)
    else:
        st.info("System Ready. Click START ANALYSIS in sidebar to launch simulation.")

with tab_subject:
    st.header(f"SUBJECT PROFILE: {subject}")
    df_s = pd.DataFrame(full_data)
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.pie(df_s, names="activity", title="ACTIVITY PROFILE", template="plotly_dark", hole=0.4), use_container_width=True)
    with col2: st.plotly_chart(px.histogram(df_s, x="activity", y="confidence", histfunc="avg", title="GESTURE CLARITY", template="plotly_dark"), use_container_width=True)
    st.subheader("FULL SESSION GESTURE CHRONOLOGY")
    st.plotly_chart(px.scatter(df_s, x=df_s.index, y="activity", color="confidence", title="TEMPORAL SEQUENCE", template="plotly_dark"), use_container_width=True)

with tab_theory:
    st.header("THE SCIENCE BEHIND GESTURAI")
    
    st.subheader("1. Resultant Magnitude (Physics Layer)")
    st.latex(r"a_{res} = \sqrt{a_x^2 + a_y^2 + a_z^2}")
    st.markdown("""
    **Human Explanation**: We combine the three physical axes (X, Y, Z) into a single intensity value. 
    This allows the model to recognize activities like Jogging regardless of how the phone is oriented in your pocket. 
    It makes the system Rotation-Invariant.
    """)
    st.divider()

    st.subheader("2. Temporal Attention (Explainability Layer)")
    st.latex(r"\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}")
    st.markdown("""
    **Human Explanation**: In every 10-second window, some moments are silence and some are action. 
    The Attention Spotlight automatically scans the signal and ignores background noise, 
    focusing the model's energy only on the specific milliseconds where the gesture actually occurs.
    """)
    st.divider()

    st.subheader("3. Multi-Scale Convolutions (Deep Learning Layer)")
    st.markdown("""
    **Human Explanation**: Our model uses three parallel convolution branches with different kernel sizes (3, 5, and 11). 
    This is like looking at the data through three different magnifying glasses:
    - One for tiny high-frequency jitters (typing).
    - One for medium-scale movements (walking).
    - One for large, slow cycles (stairs).
    """)

with tab_global:
    st.header("GLOBAL SYSTEM INTELLIGENCE")
    g1, g2 = st.columns(2)
    with g1: st.image("outputs/training_metrics.png", caption="Model Convergence (Loss vs Accuracy)")
    with g2: st.image("outputs/shap_final.png", caption="Global Sensor Importance (Aggregated SHAP)")
    st.divider()
    st.image("outputs/class_distribution.png", caption="3.6M Record Industrial Dataset Distribution")
