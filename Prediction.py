import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
liver_model = pickle.load(open("models/liver_model.pkl", "rb"))

with open("metrics/diabetes_f1.txt") as f:
    diabetes_f1 = f.read()
with open("metrics/kidney_f1.txt") as f:
    kidney_f1 = f.read()
with open("metrics/heart_f1.txt") as f:
    heart_f1 = f.read()
with open("metrics/liver_f1.txt") as f:
    liver_f1 = f.read()

genai.configure(api_key=os.getenv("GEMINI_API")) 

# Gemini advice generation
def get_advice(disease, risk):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = f"""
        A medical ML model has predicted that the patient has a {'HIGH' if int(risk) else 'LOW'} risk of **{disease.upper()}**.

        Please provide:
        - A simple explanation of this disease
        - Lifestyle recommendations
        - Diet suggestions
        - Early warning signs
        - When to consult a doctor

        Explain in non-technical language.
        """
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"LLM Error: {str(e)}"


st.set_page_config(page_title="🩺 Disease Risk Predictor", layout="wide")


st.markdown("""
    <style>
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background-color: #4CAF50; border-radius: 10px; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🩺 Disease Navigator")
st.sidebar.image("nav.jpg", use_container_width=True)
disease = st.sidebar.radio("Select Disease", ["diabetes", "kidney", "heart", "liver"])

def preprocess_input(data, disease):
    df = pd.DataFrame([data])
    
    if disease == "diabetes":
        le = LabelEncoder()
        df['gender'] = le.fit_transform([df['gender'][0]])
        df['smoking_history'] = le.fit_transform([df['smoking_history'][0]])
        df['hypertension'] = df['hypertension'].map({'Yes': 1, 'No': 0})
        df['heart_disease'] = df['heart_disease'].map({'Yes': 1, 'No': 0})
        df = df.astype(float)

    elif disease == "kidney":
        # FIXED: Match the exact 10 features the model was trained on
        le = LabelEncoder()
        for col in ['Hypertension (yes/no)', 'Diabetes mellitus (yes/no)']:
            df[col] = le.fit_transform([df[col][0]])

        num_cols = [
            'Blood pressure (mm/Hg)',
            'Albumin in urine',
            'Random blood glucose level (mg/dl)',
            'Blood urea (mg/dl)',
            'Serum creatinine (mg/dl)',
            'Hemoglobin level (gms)',
            'Estimated Glomerular Filtration Rate (eGFR)'
        ]
        df[num_cols] = df[num_cols].astype(float)
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Exact order the model expects (10 features)
        expected_order = [
            'Age of the patient',
            'Blood pressure (mm/Hg)',
            'Albumin in urine',
            'Random blood glucose level (mg/dl)',
            'Blood urea (mg/dl)',
            'Serum creatinine (mg/dl)',
            'Hemoglobin level (gms)',
            'Hypertension (yes/no)',
            'Diabetes mellitus (yes/no)',
            'Estimated Glomerular Filtration Rate (eGFR)'
        ]
        df = df[expected_order]


    elif disease == "heart":
        # FIXED: Match the exact 11 features the model was trained on
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Hypertension'] = df['Hypertension'].map({'yes': 1, 'no': 0})
        df['Diabetes'] = df['Diabetes'].map({'yes': 1, 'no': 0})
        df['Obesity'] = df['Obesity'].map({'yes': 1, 'no': 0})
        df['Smoking'] = df['Smoking'].map({'yes': 1, 'no': 0})
        df['EKG_Results'] = df['EKG_Results'].map({'normal': 0, 'abnormal': 1})
        
        # Only 11 features the model expects
        expected_order = [
            'Age', 'Gender', 'Hypertension', 'Diabetes', 'Cholesterol',
            'Obesity', 'Smoking', 'Systolic_BP', 'Diastolic_BP', 
            'Triglycerides', 'EKG_Results'
        ]
        df = df[expected_order].fillna(0).astype(float)


    elif disease == "liver":
        df = df.fillna(0).astype(float)

    return df


st.title("🧬 Disease Risk Predictor")

# Disease-specific input fields
if disease == "diabetes":
    st.image("dbb.jpg", width=1000)
    input_data = {
        'gender': st.selectbox("Gender", ["Male", "Female"]),
        'age': st.number_input("Age", min_value=1, max_value=100, value=30),
        'hypertension': st.selectbox("Hypertension", ["Yes", "No"]),
        'heart_disease': st.selectbox("Heart Disease", ["Yes", "No"]),
        'smoking_history': st.selectbox("Smoking History", ["never", "current", "not current", "ever"]),
        'bmi': st.slider("BMI (Normal: 18.5–24.9)", 10.0, 40.0, 22.0),
        'HbA1c_level': st.slider("HbA1c Level (Normal: <5.7%)", 3.0, 10.0, 5.5),
        'blood_glucose_level': st.slider("Blood Glucose Level (Normal: 70–140 mg/dL)", 70, 200, 90)
    }


elif disease == "kidney":
    st.image("kd.jpg", use_container_width=True)
    # FIXED: Only collect the 10 features the model needs
    input_data = {
        'Age of the patient': st.number_input("Age", min_value=1, max_value=100, value=45),
        'Blood pressure (mm/Hg)': st.slider("Blood Pressure (Normal: 90–120 mm/Hg)", 80.0, 180.0, 110.0),
        'Albumin in urine': st.slider("Albumin (Normal: <30 mg/g)", 0.0, 50.0, 10.0),
        'Random blood glucose level (mg/dl)': st.slider("Glucose (Normal: <140 mg/dL)", 50.0, 300.0, 100.0),
        'Blood urea (mg/dl)': st.slider("Urea (Normal: 7–20 mg/dL)", 5.0, 100.0, 15.0),
        'Serum creatinine (mg/dl)': st.slider("Creatinine (Normal: 0.6–1.3 mg/dL)", 0.1, 15.0, 1.0),
        'Hemoglobin level (gms)': st.slider("Hemoglobin (Normal: 12–17 g/dL)", 5.0, 20.0, 15.0),
        'Estimated Glomerular Filtration Rate (eGFR)': st.slider("eGFR (Normal: ≥90 mL/min/1.73m²)", 10.0, 120.0, 95.0),
        'Hypertension (yes/no)': st.selectbox("Hypertension", ["yes", "no"]),
        'Diabetes mellitus (yes/no)': st.selectbox("Diabetes", ["yes", "no"])
    }


elif disease == "heart":
    st.image("aa.jpg", use_container_width=True)
    # FIXED: Only collect the 11 features the model needs
    input_data = {
        'Age': st.number_input("Age (Normal: 18–60)", min_value=18, max_value=90, value=45),
        'Gender': st.selectbox("Gender", ["Male", "Female"]),
        'Hypertension': st.selectbox("Hypertension", ["yes", "no"]),
        'Diabetes': st.selectbox("Diabetes", ["yes", "no"]),
        'Cholesterol': st.slider("Cholesterol (mg/dL) (Normal: < 200)", 100, 400, 200),
        'Obesity': st.selectbox("Obesity", ["yes", "no"]),
        'Smoking': st.selectbox("Smoking", ["yes", "no"]),
        'Systolic_BP': st.slider("Systolic Blood Pressure (mmHg) (Normal: < 120)", 90, 200, 120),
        'Diastolic_BP': st.slider("Diastolic Blood Pressure (mmHg) (Normal: < 80)", 60, 130, 80),
        'Triglycerides': st.slider("Triglycerides (mg/dL) (Normal: < 150)", 50, 500, 150),
        'EKG_Results': st.selectbox("EKG Results", ["normal", "abnormal"])
    }


elif disease == "liver":
    st.image("liv.jpg", use_container_width=True)
    input_data = {
        'Total_Bilirubin': st.slider("Total Bilirubin (Normal: 0.1–1.2 mg/dL)", 0.1, 10.0, 1.0),
        'Direct_Bilirubin': st.slider("Direct Bilirubin (Normal: 0.0–0.3 mg/dL)", 0.1, 5.0, 0.3),
        'Alkaline_Phosphotase': st.slider("Alkaline Phosphotase (Normal: 44–147 IU/L)", 50, 400, 200),
        'Alamine_Aminotransferase': st.slider("Alamine Aminotransferase (ALT) (Normal: 7–56 IU/L)", 10, 100, 30),
        'Aspartate_Aminotransferase': st.slider("Aspartate Aminotransferase (AST) (Normal: 10–40 IU/L)", 10, 100, 30),
        'Albumin': st.slider("Albumin (Normal: 3.4–5.4 g/dL)", 1.0, 6.0, 4.0),
        'Albumin_and_Globulin_Ratio': st.slider("Albumin/Globulin Ratio (Normal: 1.1–2.5)", 0.1, 3.0, 1.0)
    }


# Prediction
if st.button("🔍 Predict Risk"):
    processed = preprocess_input(input_data, disease)
    
    if disease == "diabetes":
        pred = diabetes_model.predict(processed)[0]
        f1_score_used = diabetes_f1
    elif disease == "kidney":
        pred = kidney_model.predict(processed)[0]
        f1_score_used = kidney_f1
    elif disease == "heart":
        pred = heart_model.predict(processed)[0]
        f1_score_used = heart_f1
    else:
        pred = liver_model.predict(processed)[0]
        f1_score_used = liver_f1

    st.success(f"🧾 Prediction: {'High Risk' if int(pred) else 'Low Risk'}")
    st.markdown(f"📊 **Model F1 Score**: `{f1_score_used}`")

    with st.spinner("Getting Gemini Advice..."):
        advice = get_advice(disease, pred)
    st.markdown("### 🧠 Gemini Advice:")
    st.write(advice)


# Chatbot circle button
with open("b.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
    <style>
    #chatbot-circle {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        background-color: #ffffff;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }}
    #chatbot-circle img {{
        width: 40px;
        height: 40px;
    }}
    </style>

    <a href="/ai-assistant" target="_self">
        <div id="chatbot-circle">
            <img src="data:image/png;base64,{encoded}" alt="Chatbot" />
        </div>
    </a>
""", unsafe_allow_html=True)