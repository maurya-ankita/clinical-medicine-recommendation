import streamlit as st
import requests

# Set page layout and visual tab title
st.set_page_config(page_title="CDSS Dashboard", layout="centered")

st.title("🏥 Clinical Recommendation System Dashboard")
st.markdown("""
Fusing **Sentence-BERT (SBERT)** text semantics with a regularized 
**Neural Collaborative Filtering (NeuMF)** network to predict multi-label treatment plans.
""")
st.write("---")

# User Input Form Fields
patient_id = st.number_input("Enter Patient Identifier ID Number:", min_value=0, max_value=10000, value=12, step=1)
clinical_note = st.text_area(
    label="Doctor's Unstructured Clinical Summary / Symptom Notes:", 
    placeholder="Type clinical observations here (e.g., Patient presents with elevated HbA1c levels, increased thirst, and high blood sugar...)",
    height=150
)
max_recommendations = st.slider("Select Target Top-K Medicines to Predict:", min_value=1, max_value=5, value=3)

st.write("---")

# Execution Button
if st.button("⚡ Run Diagnostic Engine Inference", use_container_width=True):
    if not clinical_note.strip():
        st.warning("⚠️ Please provide a clinical symptom description first.")
    else:
        # 1. Package the payload exactly matching our FastAPI Pydantic schema
        payload = {
            "patient_id": int(patient_id),
            "condition_description": clinical_note,
            "top_k": int(max_recommendations)
        }
        
        # 2. Fire an HTTP POST request over to our active Uvicorn local server port
        try:
            api_url = "http://127.0.0.1:8000/predict"
            
            with st.spinner("Processing clinical note semantics via SBERT & NeuMF Core..."):
                response = requests.post(api_url, json=payload)
                
            if response.status_code == 200:
                data = response.json()
                
                st.success("🎉 Inference Analysis Successful!")
                st.subheader("📋 Top Recommended Treatments:")
                
                # 3. Iterate through recommendations and display interactive visual metrics
                for drug, confidence in zip(data["recommended_treatment"], data["confidence_metrics"]):
                    # Display metrics cleanly as full visual cards
                    st.metric(
                        label=f"Prescription Option: {drug}", 
                        value=f"{round(confidence * 100, 2)}% Confidence Match"
                    )
            else:
                st.error(f"❌ API Server Error (Status Code: {response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Connection Refused! Make sure your FastAPI backend server is actively running on http://127.0.0.1:8000")