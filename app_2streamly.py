import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predicción con IA",
    page_icon='https://cdn-icons-png.flaticon.com/512/5935/5935638.png',
    layout="centered",
    initial_sidebar_state="auto"
)

# Título y descripción de la aplicación
st.title("Predicción de enfermedades cardiacas usando IA")
st.markdown("Esta aplicación predice si tienes una enfermedad cardiaca basándose en tus datos ingresados.")
st.markdown("---")

# Logo en la barra lateral
logo = "imagen.png"
st.sidebar.image(logo, width=150)

# Sección de datos del usuario en la barra lateral
st.sidebar.header('Datos ingresados por el usuario')

# Cargar datos del usuario
uploaded_file = st.sidebar.file_uploader("Cargue su archivo CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # Controles para ingresar datos
        age = st.sidebar.slider('Edad', 10, 77, 41) #ok
        sex = st.sidebar.selectbox('Género', ['Masculino', 'Femenino']) #ok
        cp = st.sidebar.selectbox('Tipo de Dolor en el Pecho', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']) #ok
        trtbps = st.sidebar.slider('Presión Arterial en Reposo', 70, 200, 132) #ok
        chol = st.sidebar.slider('Colesterol Sérico (mg/dl)', 50, 564, 242) #ok
        fbs = st.sidebar.selectbox('Nivel de Azúcar en Sangre en Ayunas > 120 mg/dl', ['No', 'Sí'])
        restecg = st.sidebar.selectbox('Resultados Electrocardiográficos en Reposo', ['Normal', 'Anomalía en la Onda ST-T', 'Hipertrofia Ventricular Izquierda'])
        thalachh = st.sidebar.slider('Frecuencia Cardíaca Máxima Lograda', 60, 202, 148)
        exng = st.sidebar.selectbox('Angina Inducida por Ejercicio', ['No', 'Sí']) #ok
        oldpeak = st.sidebar.slider('Depresión del Segmento ST Inducida por el Ejercicio mm', 0.0, 6.2, 1.0)
        slp = st.sidebar.selectbox('Inclinación del Segmento ST Pico del Ejercicio', ['Ascendente', 'Plano', 'Descendente'])
        caa = st.sidebar.slider('Número de Vasos Principales Coloreados por la Fluoroscopia', 0, 3, 0)
        thall = st.sidebar.selectbox('Resultado de la Prueba de Esfuerzo Nuclear', ['Normal', 'Defecto Fijo', 'Defecto Reversible', 'Irreversible'])
        
        data = {
            'age': age,
            'sex': 1 if sex == 'Masculino' else 0,
            'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp),
            'trtbps': trtbps,
            'chol': chol,
            'fbs': 1 if fbs == 'Sí' else 0, #ok
            'restecg': ['Normal', 'Anomalía en la Onda ST-T', 'Hipertrofia Ventricular Izquierda'].index(restecg), #ok
            'thalachh': thalachh, #ok
            'exng': 1 if exng == 'Sí' else 0,
            'oldpeak': oldpeak,
            'slp': ['Ascendente', 'Plano', 'Descendente'].index(slp),
            'caa': caa,
            'thall': ['Normal', 'Defecto Fijo', 'Defecto Reversible','Irreversible'].index(thall)
        }
        
        # Convertir columnas categóricas en valores numéricos
        encoder = LabelEncoder()
        for col in ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']:
            data[col] = encoder.fit_transform([data[col]])

        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()

# Convertir columnas categóricas en valores numéricos
encoder = LabelEncoder()
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
for col in categorical_columns:
    input_df[col] = encoder.fit_transform(input_df[col])

# Seleccionar solo la primera fila
input_df = input_df[:1]

st.subheader('Datos ingresados por el usuario')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(input_df)

# Definir las columnas categóricas y sus posibles valores
categorical_columns_values = {
    'sex': [0, 1],
    'cp': [0, 1, 2, 3],
    'fbs': [0, 1],
    'restecg': [0, 1, 2],
    'exng': [0, 1],
    'slp': [0, 1, 2],
    'caa': [0, 1, 2, 3, 4],
    'thall': [0, 1, 2, 3]
}

# Orden específico de las columnas
column_order = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 
                'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# Crear un nuevo DataFrame en el formato deseado
new_data_formatted = pd.DataFrame(columns=column_order)

# Agregar las columnas categóricas con los sufijos correspondientes
for col, values in categorical_columns_values.items():
    for value in values:
        new_data_formatted[f'{col}_{value}'] = [1 if input_df[col][0] == value else 0]

# Agregar las columnas numéricas
numeric_columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
for col in numeric_columns:
    new_data_formatted[col] = input_df[col]

# Selección del método de predicción
selected_model = st.sidebar.selectbox('Seleccione el método de predicción', ['Random Forest', 'Naive Gaussian Bayes', 'Decision Tree Classifier'])

# Cargar el modelo seleccionado
models = {
    'Random Forest': 'final_rf_model.pkl',
    'Naive Gaussian Bayes': 'final_nb_model.pkl',
    'Decision Tree Classifier': 'final_dt_model.pkl',
}

if selected_model in models:
    load_clf = pickle.load(open(models[selected_model], 'rb'))

# Realizar predicciones
prediction = load_clf.predict(new_data_formatted)
prediction_proba = load_clf.predict_proba(new_data_formatted)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Predicción')
    st.write(prediction)

with col2:
    st.subheader('Probabilidad de predicción')
    st.write(prediction_proba)

if prediction == 0:
    st.subheader('La persona no tiene problemas Cardíacos')
else:
    st.subheader('La persona tiene problemas Cardiacos')

# Calcular y mostrar la exactitud del modelo
if uploaded_file is not None:
    y_true = pd.read_csv(uploaded_file)['output'].values
    y_pred = load_clf.predict(new_data_formatted)
    accuracy = accuracy_score(y_true, y_pred)
    st.subheader(f'Exactitud (Accuracy) del modelo {selected_model}: {accuracy * 100:.2f}%')

st.markdown("---")

