import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
num_samples = 1000

data = {
    "YearsExperience": np.random.randint(0, 21, num_samples),
    "EducationLevel": np.random.choice(["High School", "Bachelor", "Master", "PhD"], num_samples),
    "SkillMatch": np.random.randint(1, 6, num_samples),
    "Certifications": np.random.randint(0, 6, num_samples),
    "SimilarRolesBefore": np.random.choice(["Yes", "No"], num_samples),
    "CoverLetterIncluded": np.random.choice(["Yes", "No"], num_samples),
}

df = pd.DataFrame(data)

def compute_success(row):
    score = 0
    score += row["YearsExperience"] * 2
    score += row["SkillMatch"] * 3
    score += row["Certifications"] * 2
    if row["EducationLevel"] == "Bachelor":
        score += 5
    elif row["EducationLevel"] == "Master":
        score += 10
    elif row["EducationLevel"] == "PhD":
        score += 15
    if row["SimilarRolesBefore"] == "Yes":
        score += 5
    if row["CoverLetterIncluded"] == "Yes":
        score += 3
    return 1 if score >= 35 else 0

df["Shortlisted"] = df.apply(compute_success, axis=1)

label_cols = ["EducationLevel", "SimilarRolesBefore", "CoverLetterIncluded"]
encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Shortlisted", axis=1)
y = df["Shortlisted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_shortlist(years_exp, education, skill_match, certs, similar_roles, cover_letter):
    edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    yes_no_map = {"No": 0, "Yes": 1}
    sample = pd.DataFrame([{
        "YearsExperience": years_exp,
        "EducationLevel": edu_map[education],
        "SkillMatch": skill_match,
        "Certifications": certs,
        "SimilarRolesBefore": yes_no_map[similar_roles],
        "CoverLetterIncluded": yes_no_map[cover_letter],
    }])
    prob = model.predict_proba(sample)[0][1]
    prediction = "Yes" if prob >= 0.5 else "No"
    return prediction, round(prob * 100, 2)

print("\nExample Prediction:")
result = predict_shortlist(3, "Bachelor", 4, 2, "Yes", "Yes")
print("Shortlist:", result[0], f"({result[1]}%)")
