import streamlit as st                            #Libreria de streamlit
import pandas as pd                               #Para la lectura de archivos                             
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Configuración de la página de Streamlit pestaña
st.set_page_config(
    page_title="Predicción con IA",
    page_icon='https://cdn.pixabay.com/photo/2020/08/04/11/45/heart-5462571_1280.png',
    layout="centered",
    initial_sidebar_state="auto"
)

# Título y descripción de la aplicación
st.title("Predicción de enfermedades cardiacas usando IA")
st.markdown("Con esta aplicación evaluaremos si tienes una enfermedad cardiaca basándose en tus datos ingresados.")
st.markdown("---")


# Logo en la barra lateral
st.sidebar.header('Tu salud es primero... !!!')
logo = "https://media3.giphy.com/media/hOnvWVCQJG8YP4jS5H/giphy.gif?cid=6c09b952ob2ltgc1cl6gvhb1mgte50elh72k1mle3mp6xrqg&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=s"
st.sidebar.image(logo, width=200)

# Sección de datos del usuario en la barra lateral
# Cargar datos del usuario
st.header('Datos ingresados por el usuario')
uploaded_file = st.file_uploader("Cargue su archivo CSV", type=["csv"])


if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # Controles para ingresar datos
        age = st.sidebar.slider('Edad (age)', 25, 100, 52)
        sex = st.sidebar.selectbox('Género (sex)', ['Masculino', 'Femenino'])
        cp = st.sidebar.slider('Tipo de Dolor en el Pecho (cp)', 0, 3, 1)
        trestbps = st.sidebar.slider('Presión Arterial en Reposo (mm Hg) (trestbps)', 80, 200, 125)
        chol = st.sidebar.slider('Colesterol Sérico (mg/dl)(chol)', 120, 564, 212)
        fbs = st.sidebar.selectbox('Nivel de Azúcar en Sangre en Ayunas > 120 (mg/dl) (fbs)', ['No', 'Sí'])
        restecg = st.sidebar.selectbox('Resultados Electrocardiográficos en Reposo (restecg)', ['Normal', 'Anomalía en la Onda ST-T', 'Hipertrofia Ventricular Izquierda'])
        thalach = st.sidebar.slider('Frecuencia Cardíaca Máxima Lograda (thalach)', 70, 202, 168)
        exang = st.sidebar.selectbox('Angina Inducida por Ejercicio (exang)', ['No', 'Sí'])
        oldpeak = st.sidebar.slider('Depresión del Segmento ST Inducida por el Ejercicio mm (oldpeak)', 0.0, 6.2, 1.0)
        slope = st.sidebar.slider('Inclinación del Segmento ST Pico del Ejercicio (slope)', 0, 2, 2)
        ca = st.sidebar.slider('Número de Vasos Principales Coloreados por la Fluoroscopia (ca)', 0, 4, 4)
        thal = st.sidebar.slider('Resultado de la Prueba de Esfuerzo Nuclear (thal)', 0, 3, 0)

        data = {
            'age': age,
            'sex': 1 if sex == 'Masculino' else 0,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == 'Sí' else 0,
            'restecg': ['Normal', 'Anomalía en la Onda ST-T', 'Hipertrofia Ventricular Izquierda'].index(restecg),
            'thalach': thalach,
            'exang': 1 if exang == 'Sí' else 0,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()


# Seleccionar solo la primera fila
input_df = input_df[:1]


if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(input_df)

# Visualiza los datos del usuario
st.subheader('Datos ingresados por el usuario')

# Obtén dummies con drop_first=False y prefix para True y False
df = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
                    prefix=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=False)

# Reordena las columnas según el orden específico que deseas
desired_order = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                 'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
                 'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2',
                 'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2',
                 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4',
                 'thal_0', 'thal_1', 'thal_2', 'thal_3']

df = df.reindex(columns=desired_order)

# Rellena NaN con False
df = df.fillna(False)


# Selección del método de predicción
selected_model = st.selectbox('Seleccione el método de predicción', ['Random Forest', 'Decision Tree Classifier'])

# Cargar el modelo seleccionado
models = {
    'Random Forest': 'final_Random_Forest_model.pkl',
    'Decision Tree Classifier': 'final_Tree_Classifier_model.pkl',
}


if selected_model in models:
    load_clf = pickle.load(open(models[selected_model], 'rb'))

# Realizar predicciones
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

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




st.markdown("---")


