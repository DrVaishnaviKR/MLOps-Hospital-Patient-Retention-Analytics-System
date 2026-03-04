import os

import streamlit as st
import requests

st.set_page_config(
    page_title="Patient Churn Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Patient Churn Prediction")
st.markdown("Enter patient details below to predict churn probability.")

# API URL - uses env var on Render, falls back to localhost for local dev
DEFAULT_API_URL = os.environ.get(
    "BACKEND_URL", "https://churn-backend-k07b.onrender.com"
)
API_URL = st.sidebar.text_input(
    "FastAPI URL",
    value=DEFAULT_API_URL
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app predicts whether a patient is likely to churn "
    "based on demographics, visit history, and satisfaction scores."
)

# Input form
with st.form("prediction_form"):
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        state = st.text_input("State", value="California")

    with col2:
        tenure = st.number_input("Tenure (Months)", min_value=0, value=24)
        specialty = st.selectbox(
            "Specialty",
            ["Cardiology", "Dermatology", "General Practice",
             "Neurology", "Orthopedics", "Pediatrics",
             "Oncology", "Psychiatry"]
        )
        insurance = st.selectbox(
            "Insurance Type",
            ["Private", "Medicare", "Medicaid", "Self-Pay"]
        )

    with col3:
        visits = st.number_input(
            "Visits Last Year", min_value=0, value=5
        )
        missed = st.number_input(
            "Missed Appointments", min_value=0, value=1
        )
        days_since = st.number_input(
            "Days Since Last Visit", min_value=0, value=30
        )

    st.subheader("Satisfaction Scores")
    col4, col5 = st.columns(2)

    with col4:
        overall_sat = st.slider(
            "Overall Satisfaction", 1.0, 5.0, 3.5, 0.1
        )
        wait_sat = st.slider(
            "Wait Time Satisfaction", 1.0, 5.0, 3.5, 0.1
        )
        staff_sat = st.slider(
            "Staff Satisfaction", 1.0, 5.0, 3.5, 0.1
        )

    with col5:
        provider_rating = st.slider(
            "Provider Rating", 1.0, 5.0, 3.5, 0.1
        )
        avg_cost = st.number_input(
            "Avg Out-of-Pocket Cost ($)", min_value=0.0, value=150.0
        )
        distance = st.number_input(
            "Distance to Facility (Miles)", min_value=0.0, value=10.0
        )

    st.subheader("Other Details")
    col6, col7 = st.columns(2)

    with col6:
        billing_issues = st.selectbox(
            "Billing Issues (0=No, 1=Yes)", [0, 1]
        )
        portal_usage = st.selectbox(
            "Portal Usage (0=No, 1=Yes)", [0, 1]
        )

    with col7:
        referrals = st.number_input(
            "Referrals Made", min_value=0, value=0
        )
        last_interaction = st.text_input(
            "Last Interaction Date (DD-MM-YYYY)", value="15-01-2025"
        )

    submitted = st.form_submit_button(
        "🔮 Predict Churn", use_container_width=True
    )

if submitted:
    payload = {
        "Age": age,
        "Gender": gender,
        "State": state,
        "Tenure_Months": tenure,
        "Specialty": specialty,
        "Insurance_Type": insurance,
        "Visits_Last_Year": visits,
        "Missed_Appointments": missed,
        "Days_Since_Last_Visit": days_since,
        "Last_Interaction_Date": last_interaction,
        "Overall_Satisfaction": overall_sat,
        "Wait_Time_Satisfaction": wait_sat,
        "Staff_Satisfaction": staff_sat,
        "Provider_Rating": provider_rating,
        "Avg_Out_Of_Pocket_Cost": avg_cost,
        "Billing_Issues": billing_issues,
        "Portal_Usage": portal_usage,
        "Referrals_Made": referrals,
        "Distance_To_Facility_Miles": distance,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["churn_probability"]

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(
                    f"⚠️ **HIGH CHURN RISK** — "
                    f"Churn Probability: **{probability:.1%}**"
                )
                st.markdown(
                    "This patient is likely to churn. "
                    "Consider proactive engagement strategies."
                )
            else:
                st.success(
                    f"✅ **LOW CHURN RISK** — "
                    f"Churn Probability: **{probability:.1%}**"
                )
                st.markdown(
                    "This patient is likely to stay. "
                    "Continue current engagement practices."
                )

            # Show raw response
            with st.expander("View Raw API Response"):
                st.json(result)
        else:
            st.error(
                f"API Error: {response.status_code} - {response.text}"
            )
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. "
            "Make sure the FastAPI server is running."
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")
