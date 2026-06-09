import numpy as np
import streamlit as st
import pandas as pd
# Insertamos título
st.write(''' # ODS 4: Educación de calidad ''')
# Insertamos texto con formato
st.markdown("""
Optimización de Recursos para el Fortalecimiento Educativo.
""")
# Insertamos una imagen
st.image("Goal-04.png", caption="""Impacto de diversos factores sobre la tasa de
graduación.""")
# Usaremos un deslizador
st.sidebar.header("Presupuesto")
# Definimos los parámetros de nuestro deslizador:
# Límite inferior: 33000000000
# Límite superior: 35000000000
# Valor inicial: 40000000000
presupuesto = st.sidebar.slider("Presupuesto", 6000, 2500, 1000)
datos = pd.read_csv('datos (1).csv' , encoding= 'latin-1')
# Seleccionamos las variables
X = pd.DataFrame(datos, columns=['Inversion'])
y = datos['Termino']
# Creamos y entrenamos el modelo
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,y)
# Extraemos los coeficientes de la regresión
b1 = LR.coef_
b0 = LR.intercept_

impacto = b0 + presupuesto*b1[0] 
st.metric("Impacto Proyectado ODS 4", f" +{float(impacto):.3f}%")
