import streamlit as st
import pickle
import numpy as np
from mappings import sex as sex_mapping, embarked as embarked_mapping

# Load model
with open('model.pkl', 'rb') as p:
    model = pickle.load(p)

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details to predict survival:")

# Input fields
name = st.text_input("Passenger Name")
ticket = st.text_input("Ticket Number")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
selected_sex = st.selectbox("Sex", list(sex_mapping.keys()))
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.5)
cabin = st.text_input("Cabin (Optional)", value="")  # Ignored in model
selected_embarked = st.selectbox("Port of Embarkation (Embarked)", list(embarked_mapping.keys()))

# Prepare input for model
if st.button("Predict"):
    try:
        input_features = np.array([[
            pclass,
            sex_mapping[selected_sex],
            age,
            sibsp,
            parch,
            fare,
            embarked_mapping[selected_embarked]
        ]])

        prediction = model.predict(input_features)[0]
        result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
        st.success(f"Prediction for {name or 'Passenger'} (Ticket: {ticket}): **{result}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
