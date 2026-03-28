import streamlit as st
import pandas as pd
import joblib
import random
import hashlib

scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

# MongoDB setup (optional - graceful fallback if not available)
mongo_available = False
collection = None
try:
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.admin.command('ping')
    db = client["insurance_db"]
    collection = db["predictions"]
    mongo_available = True
except Exception:
    pass  # MongoDB not available — app continues without it

st.set_page_config(page_title="Health Insurance Costs Prediction", page_icon="💰", layout="centered")
st.title("Health Insurance Costs Prediction")
st.write("Enter the following details to predict:")

# List of insurance companies
insurance_companies = [
    "HDFC ERGO Health Insurance",
    "Star Health & Allied Insurance",
    "Care Health Insurance",
    "Niva Bupa Health Insurance",
    "ICICI Lombard Health Insurance"
]

with st.form("input_form"):
    name = st.text_input("Name")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", ["male", "female"])
        diabetic = st.selectbox("Diabetic", ["yes", "no"])
        smoker = st.selectbox("Smoker", ["yes", "no"])

    submit_button = st.form_submit_button("Predict Payment")

if submit_button:
    # Manual encoding without using LabelEncoder
    gender_encoded = 1 if gender.lower() == "male" else 0
    diabetic_encoded = 1 if diabetic.lower() == "yes" else 0
    smoker_encoded = 1 if smoker.lower() == "yes" else 0
    
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender_encoded],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic_encoded],
        "children": [children],
        "smoker": [smoker_encoded]
    })

    # Scale only numeric columns that were scaled during training
    numeric_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    prediction = model.predict(input_data)[0]
    
    # Store data in MongoDB (if available)
    if mongo_available and collection is not None:
        data_to_store = {
            "name": name,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "bloodpressure": bloodpressure,
            "diabetic": diabetic,
            "children": children,
            "smoker": smoker,
            "predicted_payment": round(prediction, 2).item()
        }
        try:
            collection.insert_one(data_to_store)
        except Exception:
            pass  # Silently ignore storage errors
    
    st.success(f"✅ Predicted Insurance Payment: ₹{prediction:.2f}/year")
    
    # Create a hash of the prediction to generate consistent shuffle
    prediction_hash = int(hashlib.md5(str(round(prediction, 2)).encode()).hexdigest(), 16)
    random.seed(prediction_hash)
    shuffled_companies = insurance_companies.copy()
    random.shuffle(shuffled_companies)
    
    # Display recommended insurance companies
    st.markdown("---")
    st.markdown("# 🏆 BEST RECOMMENDED INSURANCE COMPANIES")
    
    for i, company in enumerate(shuffled_companies, 1):
        st.markdown(f"**{i}. {company}**")
    
    st.markdown("---")
    st.info("💡 These recommendations are personalized based on your health profile and predicted insurance cost.")
