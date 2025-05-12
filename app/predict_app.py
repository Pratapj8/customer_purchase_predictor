import streamlit as st
from model_utils import load_model, predict_purchase

from model_utils import load_model

st.title("🛍️ Customer Purchase Predictor")

# Load model
theta, X_mean, X_std = load_model()

# Input form
with st.form("customer_form"):
    time_on_site = st.number_input("Time on site (minutes)", min_value=0.0, value=6.0)
    page_views = st.number_input("Number of page views", min_value=0, value=10)
    previous_purchases = st.number_input("Number of previous purchases", min_value=0, value=1)
    ad_clicks = st.number_input("Number of ad clicks", min_value=0, value=1)
    is_returning = st.selectbox("Is returning customer?", [0, 1])
    submitted = st.form_submit_button("Predict")

if submitted:
    features = [time_on_site, page_views, previous_purchases, ad_clicks, is_returning]
    prediction = predict_purchase(features, theta, X_mean, X_std)
    st.success(f"💰 Predicted Purchase Amount: ${prediction:.2f}")
