# Importar librerías necearias
import numpy as np
import streamlit as st
import pandas as pd

# Insertamos título
st.write(''' # ODS 14: Vida submarina ''')
# Insertamos texto con formato
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir el impacto del calentamiento global
en los arrecifes de coral, alineado con el **ODS 14: Vida Submarina**.
""")
# Insertamos una imagen
st.image("corales.jpg", caption="Impacto del calentamiento global en los arrecifes de coral.")

# Definimos cómo ingresará los datos el usuario
# Usaremos un deslizador
st.sidebar.header("Parámetros Ambientales")
# Definimos los parámetros de nuestro deslizador:
  # Límite inferior: 24°C. Es el límite inferior donde los arrecifes tropicales suelen estar cómodos
  # Límite superior: 35°C. La mayoría de los corales mueren o se blanquean totalmente mucho antes de llegar a esa temperatura
  # Valor inicial: 28°C. En muchos arrecifes, a partir de los 28.5°C o 29°C comienza el estrés térmico severo
temp_input = st.sidebar.slider("Temperatura del Agua (°C)", 24.0, 35.0, 28.0)

# Cargamos el archivo con los datos (.csv)
df =  pd.read_csv('Corales_ODS14.csv', encoding='latin-1')
# Seleccionamos las variables
X = df[['Temperatura_C']]
y = df['Porcentaje_Blanqueamiento']

# Creamos y entrenamos el modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

# Hacemos la predicción con el modelo y la temperatura seleccionada por el usuario
b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*temp_input

# Presentamos loa resultados
st.subheader('Estado de los corales')
st.write(f'El porcentaje de blanqueamiento es: {prediccion:.2f}%')

if prediccion < 30:
        st.success("Estado: Saludable")
elif prediccion < 60:
        st.warning("Estado: Riesgo Moderado")
else:
        st.error("Estado: Alerta Crítica")
