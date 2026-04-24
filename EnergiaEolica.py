import numpy as np
import streamlit as st
import pandas as pd

st.title('⚡ ODS 7: Energía Asequible y No Contaminante')
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir la generación de energía eólica
a partir de la velocidad del viento, alineado con el **ODS 7: Energía Sostenible**.
""")

st.sidebar.header("Parámetros Ambientales")
wind_input = st.sidebar.slider("Velocidad del Viento (km/h)", 5.0, 30.0, 15.0)

# 1️⃣ Cargar datos
df = pd.read_csv('Energia_ODS7.csv')

# 2️⃣ 🔴 LIMPIEZA DE DATOS (Esto faltaba y causaba el error de NaN)
df = df.dropna()
df = df.reset_index(drop=True)

# 3️⃣ Seleccionar variables
X = df[['Velocidad_Viento_kmh']]
y = df['Energia_Generada_kW']

# 4️⃣ Entrenar modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ⚠️ RECUERDA CAMBIAR random_state=0 POR TU MATRÍCULA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train, y_train)

# 5️⃣ Predecir con el input del usuario
b0 = LR.intercept_
b1 = LR.coef_[0]
prediccion = b0 + b1 * wind_input
prediccion = max(prediccion, 0.0) # La energía no puede ser negativa

# 6️⃣ Mostrar resultados
st.subheader('Estado de Generación')
st.write(f'La energía estimada es: **{prediccion:.2f} kW**')

if prediccion < 5:
    st.info("Estado: Baja generación / Viento insuficiente")
elif prediccion < 15:
    st.success("Estado: Generación Óptima")
else:
    st.warning("Estado: Alta generación / Verificar límites de red")
