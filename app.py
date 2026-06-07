import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import plotly.express as px

st.set_page_config(
    page_title = "Dynamic Pricing AI",
    layout="wide"
)
st.title(
    "💰 Dynamic Pricing Intelligence Engine"
)
model = PPO.load(
    "pricing_agent_v35"
)
inventory = st.slider(
    "Inventory",
    0,
    1000,
    500
)
competitor_price = st.slider(
    "Competitor Price",
    10,
    200,
    50
)
demand = st.slider(
    "Demand",
    0,
    300,
    120
)
state = np.array(
    [
        inventory,
        50,
        demand,
        30,

        inventory,
        80,
        demand,
        30,

        inventory,
        35,
        demand,
        30,

        inventory,
        120,
        demand,
        30,

        inventory,
        65,
        demand,
        30,
    ],
    dtype=np.float32
)

action, _ = model.predict(
    state,
    deterministic=True
)

recommended_prices = [
    int(a * 5 + 10)
    for a in action
]

products = ["P1","P2","P3","P4","P5"]
df = pd.DataFrame({
    "Product": products,
    "Recommended Price":
    recommended_prices
})
st.subheader("Recommended Prices")
st.dataframe(
    df,
    use_container_width=True
)

st.subheader("📊 Business Metrics")
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.metric(
        "Avarage Price",
        f"${sum(recommended_prices)/
            len(recommended_prices):.2f}"
    )
with col2:
    st.metric(
        "Inventory",
        inventory
    )    
with col3:
    st.metric(
        "Competitor Price",
        competitor_price
    )    
with col4:
    st.metric(
        "Forecast Deamand",
        int(demand * 1.15)
    )    

future_days = list(
    range(1,31)
)
forecast = []
for day in future_days:
    forecast.append(
        np.random.randint(
            1000,
            5000
        )
    )
forecast_df = pd.DataFrame({
    "Day": future_days,
    "Revenue": forecast
})    
fig = px.line(
    forecast_df,
    x="Day",
    y="Revenue",
    title="30-Day Revenue Forecast"
)
st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader(
    "🤖 AI Recommendation"
)
if inventory < 100:
    st.warning(
        "Low inventory detected. Increase price."
    )
if competitor_price < 40:
    st.info(
        "Competitor undercutting. Consider discounts."
    )    
if demand > 200:
    st.success(
        "High demand. Opportunity for premium pricing."
    ) 

