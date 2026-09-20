import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(
    repo_id="Jganesh045/superkart-sales-model",
    filename="best_superkart_sales_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("SuperKart Sales Forecast")
st.write("""
This application predicts the expected **total sales revenue** of a product at a given store
based on the product's characteristics and the store's profile.
Please enter the details below to get a sales forecast.
""")

# User input
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_type = st.selectbox("Product Type", [
    "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Snack Foods",
    "Meat", "Household", "Hard Drinks", "Fruits and Vegetables", "Breads", "Soft Drinks",
    "Breakfast", "Others", "Starchy Foods", "Seafood"
])
store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
])

product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
product_allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=1.0,
                                          value=0.05, step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP (USD)", min_value=0.0, max_value=500.0, value=150.0, step=0.1)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2005)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}])

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Product-Store Sales Total: **${prediction:,.2f} USD**")
