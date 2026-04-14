import streamlit as st
import pickle
import numpy as np

# Load model
with open('Logistic Regression/model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival:")

# Inputs
name = st.text_input("Passenger Name")
ticket = st.text_input("Ticket Number")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
selected_sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.5)
selected_embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Prediction
if st.button("Predict"):
    try:
        # Dummy PassengerId
        passenger_id = 1

        # One-hot encoding for Sex
        sex_female = 1 if selected_sex == "female" else 0
        sex_male = 1 if selected_sex == "male" else 0

        # One-hot encoding for Embarked
        embarked_C = 1 if selected_embarked == "C" else 0
        embarked_Q = 1 if selected_embarked == "Q" else 0
        embarked_S = 1 if selected_embarked == "S" else 0

        # Final input (11 features)
        input_features = np.array([[
            passenger_id,
            pclass,
            age,
            sibsp,
            parch,
            fare,
            sex_female,
            sex_male,
            embarked_C,
            embarked_Q,
            embarked_S
        ]])

        # Prediction
        prediction = model.predict(input_features)[0]

        result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"

        st.success(f"Prediction for {name or 'Passenger'} (Ticket: {ticket}): {result}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")