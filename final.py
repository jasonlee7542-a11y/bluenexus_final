import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BlueNEXUS | Autonomous Engine", layout="wide", initial_sidebar_state="expanded")

# --- INDUSTRIAL THEME CSS ---
st.markdown("""
    <style>
    .main { background-color: #0B132B; color: #FFFFFF; }
    .sidebar .sidebar-content { background-color: #1C2541; }
    h1, h2, h3 { color: #48CAE4 !important; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1C2541; padding: 15px; border-radius: 8px; border: 1px dashed #48CAE4; }
    .stAlert { background-color: #1C2541; border: 1px solid #FF3366; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🌊 BlueNEXUS // Autonomous Telemetry Engine")
st.markdown("**Infrastructure:** Plant A & Plant B | **Mode:** Unsupervised SPC & Predictive Diagnostics")

# --- AUTHENTICATION ---
password = st.text_input("🔒 Enter secure access password:", type="password")
if password != "blueNEXUS!":
    st.warning("⚠️ Restricted System: Enter credentials to initialize telemetry streams.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Command Center")
plant = st.sidebar.selectbox("Select Target Facility", ["Plant A", "Plant B"])
file_name = "plant_A_equipment_daily.csv" if plant == "Plant A" else "plant_B_equipment_daily.csv"

try:
    df = pd.read_csv(file_name)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        x_axis_col = 'date'
    else:
        x_axis_col = df.columns[0] 
        
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # --- DYNAMIC FEATURE ENGINEERING (THE CHEM-ENG EDGE) ---
    with st.expander("⚙️ Process Engineering: Calculate Specific Energy (SEC)"):
        st.markdown("Map sensor tags to calculate $SEC = \\frac{Power}{Flow Rate}$.")
        col_f, col_p = st.columns(2)
        flow_col = col_f.selectbox("Select Flow Rate Sensor:", ["None"] + numeric_cols)
        pow_col = col_p.selectbox("Select Power Usage Sensor:", ["None"] + numeric_cols)
        
        if flow_col != "None" and pow_col != "None":
            df['Specific_Energy'] = df[pow_col] / (df[flow_col] + 1e-5) # Prevent division by zero
            if 'Specific_Energy' not in numeric_cols:
                numeric_cols.append('Specific_Energy')
            st.success("✅ 'Specific_Energy' engineered successfully. Available for analysis.")

    # --- SECTION 1: TIME-SERIES ANOMALY DETECTION (ROLLING IQR) ---
    st.divider()
    st.subheader("🚨 Predictive Maintenance: Dynamic Anomaly Detection")
    st.markdown("Unsupervised tracking of operational drift using physics-informed Rolling IQR.")
    
    target_var = st.selectbox("Select Telemetry Stream for Anomaly Tracking:", numeric_cols, index=len(numeric_cols)-1)
    roll_window = st.slider("Rolling Baseline Window (Days/Periods)", min_value=3, max_value=30, value=14)
    
    if target_var:
        # Calculate Rolling SPC Bounds
        df['Rolling_Median'] = df[target_var].rolling(window=roll_window).median()
        df['Rolling_Q1'] = df[target_var].rolling(window=roll_window).quantile(0.25)
        df['Rolling_Q3'] = df[target_var].rolling(window=roll_window).quantile(0.75)
        df['Rolling_IQR'] = df['Rolling_Q3'] - df['Rolling_Q1']
        
        df['UCL'] = df['Rolling_Q3'] + (1.5 * df['Rolling_IQR']) # Upper Control Limit
        df['LCL'] = df['Rolling_Q1'] - (1.5 * df['Rolling_IQR']) # Lower Control Limit
        
        # Flag Anomalies
        df['Is_Anomaly'] = (df[target_var] > df['UCL']) | (df[target_var] < df['LCL'])
        anomalies = df[df['Is_Anomaly']]
        
        fig_anom = go.Figure()
        
        # Add Control Limits (Shaded Area)
        fig_anom.add_trace(go.Scatter(
            x=pd.concat([df[x_axis_col], df[x_axis_col][::-1]]),
            y=pd.concat([df['UCL'], df['LCL'][::-1]]),
            fill='toself', fillcolor='rgba(72, 202, 228, 0.1)',
            line=dict(color='rgba(255,255,255,0)'), name='Safe Operating Envelope'
        ))
        
        # Add Actual Telemetry Line
        fig_anom.add_trace(go.Scatter(x=df[x_axis_col], y=df[target_var], mode='lines', line=dict(color='#48CAE4', width=2), name=target_var))
        
        # Add Rolling Median
        fig_anom.add_trace(go.Scatter(x=df[x_axis_col], y=df['Rolling_Median'], mode='lines', line=dict(color='#FFFFFF', width=1, dash='dash'), name='Rolling Median'))
        
        # Add Anomaly Markers
        fig_anom.add_trace(go.Scatter(
            x=anomalies[x_axis_col], y=anomalies[target_var],
            mode='markers', marker=dict(color='#FF3366', size=8, symbol='x'), name='Detected Anomaly'
        ))
        
        fig_anom.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", title=f"Real-Time Anomaly Tracking: {target_var}")
        st.plotly_chart(fig_anom, use_container_width=True, theme=None)
        
        # Executive Summary Alert Log
        st.markdown(f"**Anomaly Log:** Detected **{len(anomalies)}** threshold breaches in the active dataset.")
        if not anomalies.empty:
            st.dataframe(anomalies[[x_axis_col, target_var, 'UCL', 'LCL']].tail(5), use_container_width=True)

    # --- SECTION 2: MULTI-VARIABLE DIAGNOSTIC OVERLAY ---
    st.divider()
    st.subheader("📊 Diagnostic Overlay (Shape & Trend Correlation)")
    
    selected_cols = []
    col1, col2, col3 = st.columns(3)
    for i, col_name in enumerate(numeric_cols):
        with [col1, col2, col3][i % 3]:
            if st.checkbox(col_name, key=f"chk_{col_name}", value=(i < 2)):
                selected_cols.append(col_name)
                
    if len(selected_cols) > 0:
        plot_df = df.copy()
        if st.checkbox("Normalize variables (0 - 100% Min-Max Scale)", value=True):
            for col in selected_cols:
                min_v, max_v = plot_df[col].min(), plot_df[col].max()
                plot_df[col] = ((plot_df[col] - min_v) / (max_v - min_v) * 100) if max_v != min_v else 0
        fig_line = px.line(plot_df, x=x_axis_col, y=selected_cols)
        fig_line.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

except FileNotFoundError:
    st.error(f"⚠️ Critical Error: '{file_name}' telemetry file missing.")