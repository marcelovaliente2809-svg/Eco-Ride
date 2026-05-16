
import streamlit as st
import pandas as pd
import joblib

# Load the trained model, preprocessing pipeline, and label encoder
model = joblib.load('modelo_churn.pkl')
pipeline_preproc = joblib.load('churn_pipeline_preproc.pkl')
label_encoder_churn = joblib.load('churn_label_encoder.pkl')

st.title('Predicción de Churn de Eco-Ride')
st.write('Introduce las características del usuario para predecir si hará churn.')

# Input widgets for features
uso_mensual_km = st.slider('Uso Mensual (Km)', 0.0, 200.0, 50.0)
soporte_tickets = st.slider('Soporte Tickets', 0, 5, 1)
region = st.selectbox('Región', ['Centro', 'Norte', 'Sur'])

if st.button('Predecir Churn'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Uso_Mensual_Km': uso_mensual_km,
        'Soporte_Tickets': soporte_tickets,
        'Region': region
    }])

    # Preprocess the input data
    preprocessed_data = pipeline_preproc.transform(input_data)

    # Make prediction
    prediction_encoded = model.predict(preprocessed_data)
    prediction_decoded = label_encoder_churn.inverse_transform(prediction_encoded)

    if prediction_decoded[0] == 1:
        st.error('El usuario **hará Churn**')
    else:
        st.success('El usuario **no hará Churn**')
