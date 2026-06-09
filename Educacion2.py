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
presupuesto = st.sidebar.slider("Presupuesto", 10000000000, 45000000000,
30000000000)
datos = pd.read_csv('datos (1).csv' , encoding= 'latin-1')
st.sidebar.header("Porcentaje de Becas")
porcentaje_becas = st.sidebar.slider("Porcentaje de Becas", 0.0, 1.0, 0.2)
st.sidebar.header("Porcentaje de Infraestructura")
porcentaje_infra = st.sidebar.slider("Porcentaje de Infraestructura", 0.0, 1.0,
0.5)
st.sidebar.header("Porcentaje de Docentes")
porcentaje_docentes = st.sidebar.slider("Porcentaje de Docentes", 0.0, 1.0, 0.15)
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
# Especificamos datos por población
numero = 1073736
prep_alumno = presupuesto/numero
# Calculamos el presupuesto asignado a cada rubro
presupuesto_becas = presupuesto * porcentaje_becas
presupuesto_infra = presupuesto * porcentaje_infra
presupuesto_docentes = presupuesto * porcentaje_docentes
# Presentamos loa resultados
total_gastado = presupuesto*(porcentaje_becas + porcentaje_infra +porcentaje_docentes)
# Validar restricciones
import warnings
if total_gastado != presupuesto:
    st.error("El total gastado debe ser exactamente igual al presupuesto.")
elif presupuesto_becas < presupuesto*.40:
    st.warning("El presupuesto de becas no cumple con el mínimo del 20%.", UserWarning)
elif presupuesto_infra > presupuesto*.45:
    st.warning("El presupuesto de infraestructura excede el tope del 50%.", UserWarning)
elif presupuesto_docentes < presupuesto*.35:
    st.warning("La capacitación docente está por debajo del 15% obligatorio.", UserWarning)
else:
    st.success("Combinación de presupuesto válida.")
st.subheader('Impacto alcanzado')
impacto = b0 + prep_alumno*b1[0] + presupuesto_infra/1000000000*0.15 + presupuesto_docentes/1000000000*.14
st.metric("Impacto Proyectado ODS 4", f" +{float(impacto):.3f}%")
# Presentamos el tipo de filosofía
if porcentaje_becas >= 0.40:
    filosofia = "Bienestar Primero (Equidad y Movilidad Social)"
elif porcentaje_infra >= 0.45:
    filosofia = "Rendimiento Estructural (Desarrollo Sostenible)"
elif porcentaje_docentes >= 0.35:
    filosofia = "Efecto Multiplicador (Excelencia Académica)"
else:
    filosofia = "Gobernanza Equilibrada (Modelo Balanceado)"
st.subheader("Clasificación Estratégica del Modelo")
st.info(f"Su propuesta: {filosofia}")
