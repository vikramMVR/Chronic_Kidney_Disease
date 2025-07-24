import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



model = pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

# App title
st.title("Chonic Kidney Disese Predictor")
st.write("This app uses a XG Booster to predict Result.")


# Inputs from the user

## gender
gender=st.radio("Gender:",['Male','Female'])
if gender=='Male':
    gender=0
else:
    gender=1

## soci statues
soci_statues=st.selectbox("Status",['Low','Middle','High'])
if soci_statues=='Low':
    soci_statues=0
elif soci_statues=='Middle':
    soci_statues=1
else:
    soci_statues=2

# education level
edu_level=st.selectbox("Education Level",['None','High School',"Bachelor's","Higher"])
if edu_level=="None":
    edu_level=0
elif edu_level=="High School":
    edu_level=1
elif edu_level=="Bachelor's":
    edu_level=2
elif edu_level=="Higher":
    edu_level=3

## BMI
bmi=st.slider("Body Mass Index",0.0,40.0,20.0)

# SMoking
smoke=st.selectbox("Smoking",['No','Yes'])
if smoke=='No':
    smoke=0
else :
    smoke=1

# FamilyHistoryKidneyDisease
FamilyHistoryKidneyDisease=st.selectbox("Does any of your family member have Kidney Disease",['No','Yes'])
if FamilyHistoryKidneyDisease=='No':
    FamilyHistoryKidneyDisease=0
else:
    FamilyHistoryKidneyDisease=1

# FamilyHistoryHypertension
FamilyHistoryHypertension=st.selectbox("Does any of your family member have Hypertension",['No','Yes'])
if FamilyHistoryHypertension=='No':
    FamilyHistoryHypertension=0
else:
    FamilyHistoryHypertension=1

# UrinaryTractInfections
UrinaryTractInfections=st.selectbox("Urinary Tract Infections",['No','Yes'])
if UrinaryTractInfections=='No':
    UrinaryTractInfections=0
else:
    UrinaryTractInfections=1

# SystolicBP
SystolicBP=st.slider("Systolic BP",90.0,180.0,100.0)

# FastingBloodSugar
FastingBloodSugar=st.slider("Fasting Blood Sugar Level",70.0,200.0,100.0)

# HbA1c
HbA1c=st.slider("HbA1c",4.0,10.0,6.0)

# SerumCreatinine
SerumCreatinine=st.slider("Serum Creatinine",0.5,5.0,2.5)

# BUNLevels
BUNLevels=st.slider("BUN Levels",5.0,50.0,20.0)

# GFR
GFR=st.slider("GFR",0,120,60)

# ProteinInUrine
ProteinInUrine=st.slider("ProteinInUrine",0.0,5.0,2.5)

# Diuretics	
Diuretics=st.selectbox("Diuretics",['No','Yes'])
if Diuretics=='No':
    Diuretics=0
else:
    Diuretics=1

# Edema
Edema=st.selectbox("Edema",['No','Yes'])
if Edema=='No':
    Edema=0
else:
    Edema=1

# MuscleCramps
MuscleCramps=st.selectbox("Muscle Cramps per Week",[1,2,3,4,5,6,7])

# Itching	
Itching=st.slider("Itching severity",0,10,5)

# WaterQuality
WaterQuality=st.selectbox("Water Quality",["Good","Bad"])
if WaterQuality=="Good":
    WaterQuality=0
else:
    WaterQuality=1


# Create a dictionary of the inputs
input_data = {
    'Gender': [gender],
    'SocioeconomicStatus': [soci_statues],
    'EducationLevel': [edu_level],
    'BMI': [bmi],
    'Smoking': [smoke],
    'FamilyHistoryKidneyDisease': [FamilyHistoryKidneyDisease],
    'FamilyHistoryHypertension': [FamilyHistoryHypertension],
    'UrinaryTractInfections': [UrinaryTractInfections],
    'SystolicBP': [SystolicBP],
    'FastingBloodSugar': [FastingBloodSugar],
    'HbA1c': [HbA1c],
    'SerumCreatinine': [SerumCreatinine],
    'BUNLevels': [BUNLevels],
    'GFR': [GFR],
    'ProteinInUrine': [ProteinInUrine],
    'Diuretics': [Diuretics],
    'Edema': [Edema],
    'MuscleCramps': [MuscleCramps],
    'Itching': [Itching],
    'WaterQuality': [WaterQuality]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Scale the input
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Scale the input
scaled_input = scaler.transform(input_df)


# Button to trigger prediction
if st.button("Predict"):

    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Show prediction result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("The model predicts a **high risk of Chronic Kidney Disease**.")
    else:
        st.success("The model predicts a **low risk of Chronic Kidney Disease**.")

    # Show prediction probability
    st.subheader("Prediction Probability:")
    st.write(f"Chance of CKD: **{round(prediction_proba[0][1] * 100, 2)}%**")
    st.write(f"Chance of No CKD: **{round(prediction_proba[0][0] * 100, 2)}%**")