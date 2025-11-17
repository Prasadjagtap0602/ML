streamlit run app.py
import streamlit as st

st.title("🏥 Healthcare Appointment No-Show Predictor")
st.write("A simple rule-based model to predict if a patient may miss their appointment.")

age = st.slider("Patient Age", 0, 100, 35)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
appt_type = st.selectbox("Appointment Type", ["General Checkup", "Specialist", "Follow-up"])
days_until = st.slider("Days Until Appointment", 0, 30, 7)
previous_no_shows = st.slider("Previous No-Shows", 0, 5, 1)
sms_sent = st.radio("SMS Reminder Sent?", ["Yes", "No"])

def predict_no_show(age, gender, appt_type, days, prev, sms):
    score = 0
    if days > 10:
        score += 2
    score += prev * 2
    if sms == "Yes":
        score -= 1
    if age < 25:
        score += 1
    probability = min(max((score * 12), 5), 95)
    prediction = "Yes" if probability >= 50 else "No"
    return prediction, probability

if st.button("Predict No-Show"):
    prediction, probability = predict_no_show(
        age, gender, appt_type, days_until, previous_no_shows, sms_sent
    )
    st.subheader("📊 Prediction Result")
    st.write(f"**Will the patient miss the appointment?** {prediction}")
    st.write(f"**Likelihood of No-Show:** {probability}%")

    if prediction == "Yes":
        st.error("⚠ The patient may miss the appointment.")
    else:
        st.success("✔ The patient will likely attend.")
