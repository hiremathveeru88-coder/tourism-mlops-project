
import os
import streamlit as st
import pandas as pd
import joblib


# Load the model committed by the pipeline
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)


st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts the likelihood of a customer purchasing
the tourism package based on customer and travel-related information.

Enter the customer details below to get a prediction.
""")


# Customer inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=200,
    value=15,
    step=1
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Free Lancer",
        "Small Business",
        "Large Business"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Divorced",
        "Single",
        "Unmarried"
    ]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    value=25000,
    step=1000
)


# Create input dataframe

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])


# Prediction

if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "Customer is likely to purchase the tourism package"

        st.subheader("Prediction Result:")
        st.success(f"**{result}**")

    else:
        result = "Customer is unlikely to purchase the tourism package"

        st.subheader("Prediction Result:")
        st.warning(f"**{result}**")

    st.write(
        f"Purchase probability: **{probability:.2%}**"
    )
