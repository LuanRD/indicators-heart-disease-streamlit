import pandas as pd
import pickle
import streamlit as st

pipeline = pickle.load(open('models/pipeline.pkl', 'rb'))

columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
           'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
           'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
           'SkinCancer']


def predict_disease(inputs):
    values = []

    for i, j in enumerate(inputs):
        i = []
        i.append(j)
        values.append(i)

    zip_obj = zip(columns, values)
    df = pd.DataFrame(dict(zip_obj))

    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)

    if prediction[0] == 0:
        return f'Result: You probably do not have heart disease. Probability = {(probability[0][1] * 100):.2f}%'
    else:
        return f'Result: You probably have heart disease. Probability = {(probability[0][1] * 100):.2f}%'


def main():
    st.title('Heart Disease Prediction App')

    st.subheader("This application was created based on data provided by the American CDC in an annual survey. From these data, a Machine Learning model was created that sought to predict people's heart health from personal information. This project does not is intended to be scientific.", anchor=None)

    Sex = st.radio("Are you male or female?", ("Male", "Female"))
    Race = st.radio("Select one of them below:",
                    ("American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "White", "Other"))
    AgeCategory = st.radio("Select your age category:", (
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
        "80 or older"))                    
    Height = st.number_input("Insert your height (m)", 0.01, 2.10)
    Weight = st.number_input("Insert your weight (kg)", 0.01, 300.00)
    BMI = Weight/(Height**2)
    Smoking = st.radio("Have you smoked at least 100 cigarettes in your entire life?", ("No", "Yes"))
    AlcoholDrinking = st.radio(
        "Are you a heavy drinker? [Male - more than 14 drinks/week / Female - more than 7 drinks/week]", ("No", "Yes"))
    Stroke = st.radio("Had you a stroke?", ("No", "Yes"))
    PhysicalHealth = st.number_input("How many days during the past 30 days was your physical health not good?", 0, 30)
    MentalHealth = st.number_input("How many days during the past 30 days was your mental health not good?", 0, 30)
    DiffWalking = st.radio("Do you have serious difficulty walking or climbing stairs?", ("No", "Yes"))
    Diabetic = st.radio("Do you have diabetes?", ("No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"))
    PhysicalActivity = st.radio(
        "Did you do any physical activity or exercise during the past 30 days other than their regular job?",
        ("No", "Yes"))
    GenHealth = st.radio("Would you say that in general your health is...",
                         ("Poor", "Fair", "Good", "Very good", "Excellent"))
    SleepTime = st.number_input("On average, how many hours of sleep do you get in a 24-hour period?", 0, 24)
    Asthma = st.radio("Do you have asthma?", ("No", "Yes"))
    KidneyDisease = st.radio(
        "Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?",
        ("No", "Yes"))
    SkinCancer = st.radio("Have/Had you skin cancer?", ("No", "Yes"))

    diagnosis = ''

    if st.button('Heart Disease Test Result'):
        diagnosis = predict_disease(
            [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory,
             Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer])

        st.success(diagnosis)


if __name__ == '__main__':
    main()
