
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    ohe = joblib.load('one_hot_encoder_region.joblib')
    scaler_standard = joblib.load('standard_scaler_uso_mensual_km.joblib')
    scaler_minmax = joblib.load('minmax_scaler_soporte_tickets.joblib')
    modelo = joblib.load('modelo_churn.pkl')
    return ohe, scaler_standard, scaler_minmax, modelo

ohe, scaler_standard, scaler_minmax, modelo_churn = load_models()

st.title("Predicción de Churn - Eco Ride")

uso_mensual = st.number_input("Uso Mensual (Km)", min_value=0.0)
soporte_tickets = st.number_input("Soporte Tickets", min_value=0, step=1)
region = st.selectbox("Región", ["Centro", "Norte", "Sur"])

if st.button("Predecir"):
    df_new = pd.DataFrame({
        'Uso_Mensual_Km': [uso_mensual],
        'Soporte_Tickets': [float(soporte_tickets)],
        'Region': [region]
    })

    region_encoded = ohe.transform(df_new[['Region']])
    region_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['Region']))

    df_new['Uso_Mensual_Km'] = scaler_standard.transform(df_new[['Uso_Mensual_Km']])
    df_new['Soporte_Tickets'] = scaler_minmax.transform(df_new[['Soporte_Tickets']])

    df_new = df_new.drop(columns=['Region']).reset_index(drop=True)
    df_final = pd.concat([df_new, region_df], axis=1)

    prediction = modelo_churn.predict(df_final)
    proba = modelo_churn.predict_proba(df_final)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Cliente propenso a irse — Probabilidad: {proba:.2%}")
    else:
        st.success(f"✅ Cliente no propenso a irse — Probabilidad de churn: {proba:.2%}")
