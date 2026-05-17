"""
Corporate Financial Risk Dashboard — Fraud Detection System
Fully portable (CSV-based), real AI predictions, deployment-ready.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Risk Assurance — Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME CONFIGURATION ──────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

def toggle_theme():
    st.session_state.theme = 'Light' if st.session_state.theme == 'Dark' else 'Dark'

# Add the theme toggle button to the top of the sidebar
st.sidebar.button(f"☀️ Switch to Light Theme" if st.session_state.theme == 'Dark' else "🌙 Switch to Dark Theme", on_click=toggle_theme, use_container_width=True)

is_dark = st.session_state.theme == 'Dark'

# Dynamic Colors
bg_color = "rgba(15,12,41,0.5)" if is_dark else "rgba(240,242,246,0.5)"
text_color = "rgba(255,255,255,0.8)" if is_dark else "rgba(0,0,0,0.8)"
header_bg = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)" if is_dark else "linear-gradient(135deg, #f5f7fa, #c3cfe2)"
header_text = "#fff" if is_dark else "#333"
kpi_bg = "linear-gradient(145deg, rgba(30,30,60,0.8), rgba(20,20,40,0.9))" if is_dark else "linear-gradient(145deg, #ffffff, #f0f2f6)"
kpi_label = "rgba(255,255,255,0.55)" if is_dark else "rgba(0,0,0,0.6)"
kpi_value = "#fff" if is_dark else "#111"
sidebar_bg = "linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%)" if is_dark else "linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)"
sidebar_text = "#fff" if is_dark else "#222"
model_stat_bg = "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.05)"
model_stat_border = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"
model_stat_label = "rgba(255,255,255,0.6)" if is_dark else "rgba(0,0,0,0.6)"

# ─── CUSTOM CSS ──────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, .stApp {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.5rem; }}

/* Header */
.header-container {{
    background: {header_bg};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}}
.header-container h1 {{
    color: {header_text};
    font-weight: 700;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
}}
.header-container p {{
    color: {header_text};
    opacity: 0.7;
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}}
.header-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #00c9ff, #92fe9d);
    color: #0a0a0a;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.6rem;
}}

/* KPI Metric Cards */
div[data-testid="stMetric"] {{
    background: {kpi_bg};
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    backdrop-filter: blur(10px);
}}
div[data-testid="stMetric"] label {{
    color: {kpi_label} !important;
    font-weight: 500;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {kpi_value} !important;
    font-weight: 700;
    font-size: 1.8rem !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid rgba(128,128,128,0.2);
}}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {{
    color: {sidebar_text} !important;
}}

/* Plotly charts background */
.stPlotlyChart {{ border-radius: 14px; overflow: hidden; }}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(128,128,128,0.2);
}}

/* Fraud alert card */
.fraud-alert {{
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 24px rgba(255,65,108,0.3);
    animation: pulse 2s ease-in-out infinite;
}}
.legit-alert {{
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 24px rgba(56,239,125,0.2);
}}
@keyframes pulse {{
    0%, 100% {{ box-shadow: 0 4px 24px rgba(255,65,108,0.3); }}
    50% {{ box-shadow: 0 4px 36px rgba(255,65,108,0.6); }}
}}

/* Model stat cards */
.model-stat {{
    background: {model_stat_bg};
    border: 1px solid {model_stat_border};
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
}}
.model-stat .label {{ color: {model_stat_label}; font-size: 0.82rem; }}
.model-stat .value {{ color: #38ef7d; font-weight: 700; font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)

# ─── LOAD RESOURCES ──────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('model_fraud.pkl')
    scaler = joblib.load('scaler_fraud.pkl')
    features = joblib.load('feature_columns.pkl') if os.path.exists('feature_columns.pkl') else None
    metrics = {}
    if os.path.exists('training_metrics.json'):
        with open('training_metrics.json', 'r') as f:
            metrics = json.load(f)
    return model, scaler, features, metrics

@st.cache_data
def load_data():
    df = pd.read_csv('processed_fraud_data.csv')
    return df

model, scaler, feature_columns, metrics = load_model()
df = load_data()

# ─── HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>🛡️ Corporate Financial Risk Dashboard</h1>
    <p>Transaction Anomaly Detection System — Powered by Backpropagation Neural Network</p>
    <div class="header-badge">🟢 AI MODEL ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: AI PREDICTION ─────────────────────────────
st.sidebar.markdown("## 🔍 Forensic AI Simulator")
st.sidebar.markdown("Test a transaction against the trained neural network:")

input_step = st.sidebar.number_input("Step (Time Unit)", min_value=1, value=1, step=1)
input_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=100.0)
input_type = st.sidebar.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT'])
input_old_bal_orig = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
input_new_bal_orig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=4000.0)
input_old_bal_dest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
input_new_bal_dest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=1000.0)

if st.sidebar.button("🚀 Run AI Detection", use_container_width=True):
    if feature_columns is not None:
        # Build feature vector matching training pipeline
        error_bal_orig = input_new_bal_orig + input_amount - input_old_bal_orig
        error_bal_dest = input_old_bal_dest + input_amount - input_new_bal_dest
        amount_bal_ratio = input_amount / input_old_bal_orig if input_old_bal_orig > 0 else 0.0
        is_orig_zeroed = 1 if input_new_bal_orig == 0 else 0
        is_dest_empty_start = 1 if input_old_bal_dest == 0 else 0

        input_dict = {
            'step': input_step,
            'amount': input_amount,
            'old_balance_orig': input_old_bal_orig,
            'new_balance_orig': input_new_bal_orig,
            'old_balance_dest': input_old_bal_dest,
            'new_balance_dest': input_new_bal_dest,
            'error_bal_orig': error_bal_orig,
            'error_bal_dest': error_bal_dest,
            'amount_bal_ratio': amount_bal_ratio,
            'is_orig_zeroed': is_orig_zeroed,
            'is_dest_empty_start': is_dest_empty_start,
        }

        # One-hot encode transaction type
        for col in feature_columns:
            if col.startswith('trans_type_'):
                type_name = col.replace('trans_type_', '')
                input_dict[col] = 1 if input_type == type_name else 0

        input_df = pd.DataFrame([input_dict])
        # Ensure column order matches training
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0]

        fraud_prob = probability[1] * 100
        legit_prob = probability[0] * 100

        if prediction == 1:
            st.sidebar.markdown(f"""
            <div class="fraud-alert">
                <h2 style="margin:0;">🚨 FRAUD DETECTED</h2>
                <p style="margin:0.5rem 0 0 0; font-size:1.3rem; font-weight:700;">{fraud_prob:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"""
            <div class="legit-alert">
                <h2 style="margin:0;">✅ LEGITIMATE</h2>
                <p style="margin:0.5rem 0 0 0; font-size:1.3rem; font-weight:700;">{legit_prob:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Key Risk Factors:**")
        if error_bal_orig != 0:
            st.sidebar.warning(f"Balance error (sender): ${error_bal_orig:,.2f}")
        if is_orig_zeroed:
            st.sidebar.warning("Account drained to zero")
        if amount_bal_ratio > 1:
            st.sidebar.warning(f"Amount exceeds balance ({amount_bal_ratio:.1f}x)")
        if error_bal_orig == 0 and not is_orig_zeroed and amount_bal_ratio <= 1:
            st.sidebar.info("No significant risk factors detected")
    else:
        st.sidebar.error("⚠️ Feature columns not found. Please retrain the model.")

# Model performance card in sidebar
if metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    for label, key in [("Accuracy", "accuracy"), ("F1 Score", "f1_score"),
                       ("Precision", "precision"), ("Recall", "recall"),
                       ("ROC-AUC", "roc_auc")]:
        val = metrics.get(key, 0)
        color = "#38ef7d" if val >= 0.9 else "#ffd700" if val >= 0.7 else "#ff4b2b"
        st.sidebar.markdown(f"""
        <div class="model-stat">
            <span class="label">{label}</span>
            <span class="value" style="color:{color}">{val:.4f}</span>
        </div>""", unsafe_allow_html=True)
    cv = metrics.get('cv_f1_mean', 0)
    cv_std = metrics.get('cv_f1_std', 0)
    st.sidebar.markdown(f"""
    <div class="model-stat">
        <span class="label">CV F1 (5-fold)</span>
        <span class="value">{cv:.4f} ± {cv_std:.4f}</span>
    </div>""", unsafe_allow_html=True)

# ─── KPI METRICS ─────────────────────────────────────────
total = len(df)
fraud_count = df[df['is_fraud'] == 1].shape[0]
fraud_rate = (fraud_count / total) * 100
money_at_risk = df[df['is_fraud'] == 1]['amount'].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions Monitored", f"{total:,}")
c2.metric("Fraud Detected", f"{fraud_count:,}")
c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
c4.metric("Potential Loss Prevented", f"${money_at_risk:,.0f}")

st.markdown("")

# ─── ROW 1: Scatter + Pie ────────────────────────────────
chart_colors = {0: '#38ef7d', 1: '#ff416c'}
plot_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor=bg_color,
    font=dict(family='Inter', color=text_color, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=text_color))
)


col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("#### 💰 Anomaly Scatter — Amount vs Sender Balance")
    df_plot = df.copy()
    df_plot['Fraud'] = df_plot['is_fraud'].map({0: 'Legit', 1: 'Fraud'})
    fig_scatter = px.scatter(
        df_plot, x='old_balance_orig', y='amount', color='Fraud',
        color_discrete_map={'Legit': '#38ef7d', 'Fraud': '#ff416c'},
        opacity=0.6,
        hover_data=['error_bal_orig', 'amount_bal_ratio'],
        labels={'old_balance_orig': 'Sender Original Balance ($)', 'amount': 'Transaction Amount ($)'}
    )
    fig_scatter.update_layout(**plot_layout, height=420)
    fig_scatter.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    st.markdown("#### 📊 Fraud by Transaction Type")
    type_cols = [c for c in df.columns if c.startswith('trans_type_')]
    if type_cols:
        fraud_df = df[df['is_fraud'] == 1]
        type_counts = {}
        for col in type_cols:
            name = col.replace('trans_type_', '')
            type_counts[name] = fraud_df[col].sum()
        # The dropped first dummy category
        other_count = len(fraud_df) - sum(type_counts.values())
        if other_count > 0:
            type_counts['OTHER'] = other_count
        fig_pie = px.pie(
            names=list(type_counts.keys()), values=list(type_counts.values()),
            hole=0.45,
            color_discrete_sequence=['#ff416c', '#ff8c42', '#ffd700', '#38ef7d', '#00c9ff']
        )
        fig_pie.update_layout(**plot_layout, height=420, showlegend=True)
        fig_pie.update_traces(textfont_color='white', textinfo='label+percent')
        st.plotly_chart(fig_pie, use_container_width=True)

# ─── ROW 2: Trend + Balance Error ────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 📈 Fraud Trend Over Time")
    trend = df.groupby('step').agg(
        total=('is_fraud', 'count'),
        fraud=('is_fraud', 'sum')
    ).reset_index()
    # Bin into larger time steps for readability
    trend['step_bin'] = pd.cut(trend['step'], bins=30, labels=False)
    trend_binned = trend.groupby('step_bin').agg(
        fraud=('fraud', 'sum'), total=('total', 'sum')
    ).reset_index()
    trend_binned['fraud_rate'] = (trend_binned['fraud'] / trend_binned['total'] * 100).round(2)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=trend_binned['step_bin'], y=trend_binned['total'],
        name='Total Txns', marker_color='rgba(56,239,125,0.3)',
        yaxis='y'
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_binned['step_bin'], y=trend_binned['fraud'],
        name='Fraud Count', line=dict(color='#ff416c', width=3),
        mode='lines+markers', yaxis='y2'
    ))
    trend_layout = {k: v for k, v in plot_layout.items() if k != 'legend'}
    fig_trend.update_layout(
        **trend_layout, height=380,
        yaxis=dict(title='Total Transactions', showgrid=False),
        yaxis2=dict(title='Fraud Count', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', font=dict(color='rgba(255,255,255,0.7)'))
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_d:
    st.markdown("#### 🔥 Risk Score Distribution")
    if 'amount_bal_ratio' in df.columns:
        df_risk = df.copy()
        df_risk['risk_label'] = df_risk['is_fraud'].map({0: 'Legit', 1: 'Fraud'})
        # Cap ratio for visualization
        df_risk['ratio_capped'] = df_risk['amount_bal_ratio'].clip(upper=10)
        fig_risk = px.histogram(
            df_risk, x='ratio_capped', color='risk_label', nbins=50,
            color_discrete_map={'Legit': '#38ef7d', 'Fraud': '#ff416c'},
            barmode='overlay', opacity=0.7,
            labels={'ratio_capped': 'Amount / Balance Ratio (capped at 10x)'}
        )
        fig_risk.update_layout(**plot_layout, height=380)
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("amount_bal_ratio feature not available. Retrain to enable this chart.")

# ─── ALERTS TABLE ─────────────────────────────────────────
st.markdown("#### ⚠️ High Priority Alerts — Recent Fraudulent Transactions")
fraud_alerts = df[df['is_fraud'] == 1].copy()
display_cols = ['step', 'amount', 'old_balance_orig', 'new_balance_orig',
                'error_bal_orig', 'amount_bal_ratio', 'is_orig_zeroed']
display_cols = [c for c in display_cols if c in fraud_alerts.columns]
fraud_display = fraud_alerts[display_cols].head(15)

# Style the amount column
st.dataframe(
    fraud_display.style.format({
        'amount': '${:,.2f}',
        'old_balance_orig': '${:,.2f}',
        'new_balance_orig': '${:,.2f}',
        'error_bal_orig': '${:,.2f}',
        'amount_bal_ratio': '{:.2f}x'
    }).background_gradient(subset=['amount'], cmap='Reds'),
    use_container_width=True, height=400
)

# ─── CONFUSION MATRIX (if metrics available) ─────────────
if metrics and 'confusion_matrix' in metrics:
    st.markdown("#### 🎯 Model Confusion Matrix")
    cm = metrics['confusion_matrix']
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[cm['true_negative'], cm['false_positive']],
           [cm['false_negative'], cm['true_positive']]],
        x=['Predicted Legit', 'Predicted Fraud'],
        y=['Actual Legit', 'Actual Fraud'],
        text=[[f"TN: {cm['true_negative']:,}", f"FP: {cm['false_positive']:,}"],
              [f"FN: {cm['false_negative']:,}", f"TP: {cm['true_positive']:,}"]],
        texttemplate='%{text}',
        colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#ff416c']],
        showscale=False
    ))
    fig_cm.update_layout(
        **plot_layout, height=350,
        xaxis=dict(title='Predicted'), yaxis=dict(title='Actual', autorange='reversed')
    )
    fig_cm.update_traces(textfont=dict(size=16, color='white'))
    st.plotly_chart(fig_cm, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Developed by **Athar Iftikhar Akhsan** | "
    "Tech Stack: Python, Scikit-Learn, SMOTE, Streamlit, Plotly | "
    "Project for Forensic Technology & Risk Assurance"
)