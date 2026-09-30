import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Churn Intelligence Engine", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Customer Churn & Retention Engine")
st.caption("Real-Time Account Risk Scoring Platform")

@st.cache_resource
def train_and_cache_model():
    np.random.seed(42)
    n = 1200
    tenure = np.random.randint(1, 72, size=n)
    monthly = np.random.uniform(20, 120, size=n)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n, p=[0.6, 0.25, 0.15])
    payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n)
    tickets = np.random.poisson(1.5, size=n)
    
    # Calculate probability
    prob = 0.3 * (contract == 'Month-to-month') + 0.08 * tickets - 0.005 * tenure
    prob = np.clip(prob, 0.05, 0.95)
    churn = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': np.round(monthly, 2),
        'total_charges': np.round(monthly * tenure, 2),
        'contract_type': contract,
        'payment_method': payment,
        'support_tickets': tickets,
        'churn': churn
    })
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['tenure_months', 'monthly_charges', 'total_charges', 'support_tickets']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['contract_type', 'payment_method'])
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    
    model.fit(X, y)
    return model, df

model, df = train_and_cache_model()

# KPI Banner
c1, c2, c3 = st.columns(3)
c1.metric("Monitored Accounts", f"{len(df):,}")
c2.metric("Average Monthly Bill", f"${df['monthly_charges'].mean():.2f}")
c3.metric("Baseline Churn Rate", f"{(df['churn'].mean() * 100):.1f}%")

st.markdown("---")

# Sidebar
st.sidebar.header("📋 Customer Profile")
t_months = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
m_charge = st.sidebar.number_input("Monthly Charges ($)", 20.0, 150.0, 65.0)
s_tickets = st.sidebar.slider("Support Tickets Raised", 0, 10, 2)
c_type = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
p_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

input_data = pd.DataFrame([{
    'tenure_months': t_months,
    'monthly_charges': m_charge,
    'total_charges': t_months * m_charge,
    'contract_type': c_type,
    'payment_method': p_method,
    'support_tickets': s_tickets
}])

# Inference
proba = model.predict_proba(input_data)[0][1]
pct = proba * 100

st.subheader("🎯 Real-Time Prediction")
col_res, col_chart = st.columns([1, 2])

with col_res:
    if proba >= 0.60:
        st.error(f"### High Risk: {pct:.1f}%")
        st.write("🚨 Immediate customer retention outreach recommended.")
    elif proba >= 0.35:
        st.warning(f"### Medium Risk: {pct:.1f}%")
        st.write("⚠️ Proactive follow-up required.")
    else:
        st.success(f"### Low Risk: {pct:.1f}%")
        st.write("✅ Account is healthy.")

with col_chart:
    st.subheader("Historical Churn by Contract")
    chart_data = df.groupby('contract_type')['churn'].mean() * 100
    st.bar_chart(chart_data)
