import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilos de gráficos
sns.set(style="whitegrid")

# Título principal
st.title("Análisis Exploratorio de Datos")

# Cargar archivo CSV
try:
    data = pd.read_csv("https://raw.githubusercontent.com/mleal2004/mineriaStreamlit/refs/heads/main/dataset.csv")
    st.success("¡Archivo cargado con éxito!")
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
    st.stop()

# Corregir errores potenciales en tipos de datos
for col in data.select_dtypes(include=["object"]):
    data[col] = data[col].astype(str).str.strip()  # limpieza de espacios

st.write(data.head())

# Dimensiones
st.write("**Dimensiones de los datos:**")
st.write(f"Filas: {data.shape[0]}, Columnas: {data.shape[1]}")

# Estadísticas descriptivas
st.header("Estadísticas Descriptivas")
st.write(data.describe(include="all"))

# Tablas dinámicas
st.header("Tablas Dinámicas")
col1, col2 = st.columns(2)

with col1:
    row_variable = st.selectbox("Variable para filas", options=data.columns, key="rows")
with col2:
    col_variable = st.selectbox("Variable para columnas", options=data.columns, key="cols")

if row_variable and col_variable:
    try:
        pivot = pd.pivot_table(data, index=row_variable, columns=col_variable, aggfunc="size", fill_value=0)
        st.write("Tabla dinámica:")
        st.dataframe(pivot)
    except Exception as e:
        st.warning(f"No se pudo crear la tabla dinámica: {e}")

# Gráficos
st.header("Gráficos")
st.subheader("Gráficos de distribución")

numeric_columns = data.select_dtypes(include=["number"]).columns
categorical_columns = data.select_dtypes(include=["object", "category"]).columns

# Gráfico de distribución numérica
if len(numeric_columns) > 0:
    column_to_plot = st.selectbox("Selecciona una columna numérica", options=numeric_columns, key="num_col")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data[column_to_plot].dropna(), kde=True, color="blue", ax=ax)
    ax.set_title(f"Distribución de {column_to_plot}")
    plt.tight_layout()
    st.pyplot(fig)

# Gráfico de conteo categórico
if len(categorical_columns) > 0:
    column_to_plot = st.selectbox("Selecciona una columna categórica", options=categorical_columns, key="cat_col")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = data[column_to_plot].value_counts().index
    sns.countplot(data=data, x=column_to_plot, order=order, palette="viridis", ax=ax)
    ax.set_title(f"Conteo de {column_to_plot}")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Matriz de correlación
st.subheader("Matriz de correlación")
if len(numeric_columns) > 1:
    corr_matrix = data[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlación")
    plt.tight_layout()
    st.pyplot(fig)

# Filtros dinámicos
st.header("Filtros Dinámicos")
selected_columns = st.multiselect("Selecciona columnas para aplicar filtros:", options=data.columns)

if selected_columns:
    filters = {}
    for column in selected_columns:
        if data[column].dtype == 'object':
            filters[column] = st.multiselect(f"Filtrar {column}", options=sorted(data[column].unique()))
        else:
            min_val = float(data[column].min())
            max_val = float(data[column].max())
            filters[column] = st.slider(f"Filtrar {column}", min_val, max_val, (min_val, max_val))

    filtered_data = data.copy()
    for column, filter_value in filters.items():
        if isinstance(filter_value, list):
            if filter_value:
                filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
        else:
            filtered_data = filtered_data[
                (filtered_data[column] >= filter_value[0]) & (filtered_data[column] <= filter_value[1])
            ]

    st.write("Datos filtrados:")
    st.dataframe(filtered_data)

# Resumen final
st.header("Resumen Final")
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Número de filas", data.shape[0])
        st.metric("Número de columnas", data.shape[1])
    with col2:
        st.metric("Valores nulos", int(data.isnull().sum().sum()))
        st.metric("Campos únicos", int(data.nunique().sum()))
