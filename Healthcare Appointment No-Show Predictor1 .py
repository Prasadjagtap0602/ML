import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("🏥 Healthcare Appointment No-Show Predictor")
st.write("Predict whether a patient will miss a medical appointment.")

data = pd.DataFrame({
    "Age": np.random.randint(0, 100, 200),
    "Gender": np.random.choice(["Male", "Female", "Other"], 200),
    "AppointmentType": np.random.choice(["General Checkup", "Specialist", "Follow-up"], 200),
    "DaysUntilAppointment": np.random.randint(0, 30, 200),
    "PreviousNoShows": np.random.randint(0, 5, 200),
    "SMSReminder": np.random.choice(["Yes", "No"], 200),
    "NoShow": np.random.choice(["Yes", "No"], 200)
})

label_cols = ["Gender", "AppointmentType", "SMSReminder", "NoShow"]
encoders = {}
for col in label_cols:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

X = data.drop("NoShow", axis=1)
y = data["NoShow"]

model = RandomForestClassifier()
model.fit(X, y)

st.header("📥 Enter Patient Details")

age = st.slider("Patient Age", 0, 100, 35)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
appt_type = st.selectbox("Appointment Type", ["General Checkup", "Specialist", "Follow-up"])
days_until = st.slider("Days Until Appointment", 0, 30, 7)
prev_noshows = st.slider("Previous No-Shows", 0, 5, 1)
sms_sent = st.radio("SMS Reminder Sent?", ["Yes", "No"])

if st.button("Predict No-Show"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [encoders["Gender"].transform([gender])[0]],
        "AppointmentType": [encoders["AppointmentType"].transform([appt_type])[0]],
        "DaysUntilAppointment": [days_until],
        "PreviousNoShows": [prev_noshows],
        "SMSReminder": [encoders["SMSReminder"].transform([sms_sent])[0]]
    })

    pred = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]
    no_show_percentage = round(pred_proba[1] * 100, 2)

    result_text = "YES — Patient likely to miss" if pred == 1 else "NO — Patient likely to attend"

    st.subheader("📊 Prediction Result")
    st.write(f"**No-Show Prediction:** {result_text}")
    st.write(f"**Probability of No-Show:** {no_show_percentage}%")
    st.info("This demo uses synthetic data. Use real clinic data for accurate predictions.")
