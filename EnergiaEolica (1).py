# Importar librerías necesarias
import numpy as np
import streamlit as st
import pandas as pd

# Insertamos título
st.write(''' # ODS 7: Energía Asequible y No Contaminante ''')
# Insertamos texto con formato
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir la generación de energía eólica
a partir de la velocidad del viento, alineado con el **ODS 7: Energía Sostenible**.
""")

# Definimos cómo ingresará los datos el usuario
# Usaremos un deslizador
st.sidebar.header("Parámetros Ambientales")
# Definimos los parámetros de nuestro deslizador:
  # Límite inferior: 4.0 km/h (corte mínimo operativo)
  # Límite superior: 35.0 km/h (límite de seguridad para evitar daños)
  # Valor inicial: 15.0 km/h
viento_input = st.sidebar.slider("Velocidad del Viento (km/h)", 4.0, 35.0, 15.0)

# Cargamos el archivo con los datos (.csv)
df = pd.read_csv('Energia_ODS7.csv')
# Seleccionamos las variables
X = df[['Velocidad_Viento_kmh']]
y = df['Energia_Generada_kW']

# Creamos y entrenamos el modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train, y_train)

# Hacemos la predicción con el modelo y la velocidad seleccionada por el usuario
b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0] * viento_input
prediccion = max(prediccion, 0.0) # La energía no puede ser negativa

# Presentamos los resultados
st.subheader('Estado de Generación')
st.write(f'La energía estimada es: **{prediccion:.2f} kW**')

if prediccion < 5.0:
    st.info("Estado: Baja generación / Viento insuficiente")
elif prediccion < 18.0:
    st.success("Estado: Generación Óptima")
else:
    st.warning("Estado: Alta generación / Verificar límites de red")
