import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import date
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Estudiantil - Grupo 001",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === LEER EL EXCEL ORIGINAL ===

df_original = pd.read_excel('ListadoDeEstudiantesGrupo_001.xlsx')
local_url="ListadoDeEstudiantesGrupo_001.xlsx"
drive_url = "https://docs.google.com/spreadsheets/d/1vpRR2UtcMm9ANHjBQnikS3qkFfa82v_fnLxph0gFJSg/export?format=xlsx"
df_original = pd.read_excel(drive_url)




# === COPIA PARA PROCESAR ===
df = df_original.copy()

# Convertir la columna a tipo datetime
df['Fecha_Nacimiento'] = pd.to_datetime(df['Fecha_Nacimiento'], errors='coerce')

# Calcular el promedio de fechas (ignorando nulos)
promedio_fecha = df['Fecha_Nacimiento'].dropna().mean()

# Rellenar vacíos o None con la fecha promedio
df['Fecha_Nacimiento'] = df['Fecha_Nacimiento'].fillna(promedio_fecha)

# Valores Numericos
df['Estatura'] = pd.to_numeric(df['Estatura'], errors='coerce')
df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')
df['Talla_Zapato'] = pd.to_numeric(df['Talla_Zapato'], errors='coerce')

# === LIMPIAR ESPACIOS VACIOS ===
df = df.replace(r'^\s*$', pd.NA, regex=True)

# === RELLENAR COLUMNAS NUMÉRICAS CON PROMEDIO ===
columnas_numericas = ['Estatura', 'Peso', 'Talla_Zapato']
for col in columnas_numericas:
    if col in df.columns:
        promedio = df[col].dropna().mean()
        df[col] = df[col].fillna(promedio)

# === RELLENAR COLUMNAS CATEGÓRICAS CON MODA ===
columnas_categoricas = ['RH', 'Color_Cabello', 'Barrio_Residencia']

for col in columnas_categoricas:
    if col in df.columns:
        # Limpiar valores vacíos, comas o espacios
        df[col] = df[col].astype(str).str.strip()                # Quita espacios al inicio y final
        df[col] = df[col].replace([",", " ,", ", ", "", "None", "nan", "NaN"], pd.NA)
        
        # Calcular la moda (valor más frecuente)
        moda = df[col].mode().iloc[0] if not df[col].mode().empty else None
        
        # Rellenar vacíos o comas con la moda
        df[col] = df[col].fillna(moda)


# Asegurar que columnas clave sean numéricas
df['Estatura'] = pd.to_numeric(df['Estatura'], errors='coerce')
df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')

# === CONVERSIÓN DE ESTATURA ===
if df['Estatura'].max() < 3:
    df['Estatura_cm'] = df['Estatura'] * 100
else:
    df['Estatura_cm'] = df['Estatura']

# === CALCULAR IMC ===
df['IMC'] = df['Peso'] / ((df['Estatura_cm'] / 100) ** 2)

def clasificar_imc(imc):
    if pd.isna(imc):
        return None
    elif imc < 18.5:
        return 'Bajo peso'
    elif imc < 25:
        return 'Adecuado'
    elif imc < 30:
        return 'Sobrepeso'
    elif imc < 35:
        return 'Obesidad grado 1'
    elif imc < 40:
        return 'Obesidad grado 2'
    else:
        return 'Obesidad grado 3'


df['Clasificacion IMC'] = df['IMC'].apply(clasificar_imc)

def descripcion_popular(imc):
    if pd.isna(imc):
        return None
    elif imc < 18.5:
        return 'Delgado'
    elif imc < 25:
        return 'Aceptable'
    elif imc < 30:
        return 'Sobrepeso'
    else:
        return 'Obesidad'

df['Descripcion IMC Popular'] = df['IMC'].apply(descripcion_popular)


# --- Calcular la edad ---
hoy = pd.Timestamp(date.today())
df['Edad'] = (hoy - df['Fecha_Nacimiento']).dt.days // 365

df['Nombre_Estudiante'] = df['Nombre_Estudiante'].astype(str).str.strip()
df['Apellido_Estudiante'] = df['Apellido_Estudiante'].astype(str).str.strip()

df['Nombre_Completo'] = df['Nombre_Estudiante'] + " " + df['Apellido_Estudiante']

# Normalizar texto de la columna Barrio_Residencia
df['Barrio_Residencia'] = (
    df['Barrio_Residencia']
    .astype(str)                      # por si hay valores nulos o no string
    .str.strip()                      # elimina espacios al inicio y final
    .str.title()                      # convierte a formato tipo "Belén"
)

df['Color_Cabello'] = (
    df['Color_Cabello']
    .astype(str)                      # por si hay valores nulos o no string
    .str.strip()                      # elimina espacios al inicio y final
    .str.title()                      # convierte a formato tipo "Belén"
)


# Reemplazar espacios en blanco y strings vacíos por NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Eliminar columnas completamente vacías (NaN, None o vacías)
df_limpio = df.dropna(axis=1, how='all')

integrantes_equipo = [
    "JAIME ALBERTO ALZATE MARULANDA",
    "JHON STIVEN CORTES RIVERA",
    "ESTEBAN ESPINOSA ARBOLEDA",
    "JUAN CAMILO HERRERA OSORIO"
]


df_equipo = df_limpio[df_limpio['Nombre_Completo'].isin(integrantes_equipo)]


st.header("Trabajo Final Programación Avanzada")
st.subheader("Integrantes:")

st.markdown("""
- **Jaime Alberto Alzate Marulanda**  
- **Jhon Stiven Cortes Rivera**  
- **Esteban Espinosa Arboleda**  
- **Juan Camilo Herrera Osorio**
""")



st.subheader("DataFrame Original")
st.caption(
        "Este archivo se obtiene **en tiempo real desde Google Drive** a través de la siguiente URL: "
        "[Ver en Drive](https://docs.google.com/spreadsheets/d/1vpRR2UtcMm9ANHjBQnikS3qkFfa82v_fnLxph0gFJSg/edit?usp=sharing). "
        "Los datos se actualizan automáticamente cada vez que se ejecuta la aplicación, mostrando la versión más reciente "
        "del archivo compartido."
    )

def resaltar_nulos_y_comas(val):
    # Convertimos a string para detectar comas o espacios
    if pd.isna(val) or str(val).strip() in [",", " ,", ", ", ""]:
        return 'background-color: #2f3542; color: #ffffff;'
    return ''

st.dataframe(
    df_original.style.applymap(resaltar_nulos_y_comas),
    use_container_width=True
)

st.header('Dashboard Estudiantil - Grupo 001')

# === SECCIÓN DE FILTROS (AL LADO IZQUIERDO) ===
st.sidebar.subheader("Filtros Interactivos")

# === FILTROS EN TRES COLUMNAS (DENTRO DEL SIDEBAR NO HAY COLUMNAS) ===
filtro_rh = st.sidebar.multiselect(
    "Tipo de Sangre (RH):",
    options=sorted(df_limpio['RH'].dropna().unique())
)

filtro_cabello = st.sidebar.multiselect(
    "Color de Cabello:",
    options=sorted(df_limpio['Color_Cabello'].dropna().unique())
)

filtro_barrio = st.sidebar.multiselect(
    "Barrio de Residencia:",
    options=sorted(df_limpio['Barrio_Residencia'].dropna().unique())
)

# === SLIDERS ===
rango_edad = st.sidebar.slider(
    "Rango de Edad:",
    min_value=int(df_limpio["Edad"].min()),
    max_value=int(df_limpio["Edad"].max()),
    value=(int(df_limpio["Edad"].min()), int(df_limpio["Edad"].max())),
    step=1
)

rango_estatura = st.sidebar.slider(
    "Rango de Estatura (cm):",
    min_value=int(df_limpio["Estatura_cm"].min()),
    max_value=int(df_limpio["Estatura_cm"].max()),
    value=(int(df_limpio["Estatura_cm"].min()), int(df_limpio["Estatura_cm"].max())),
    step=1
)

st.sidebar.markdown("### Integrantes del Equipo")

st.sidebar.markdown(
    "<div style='font-size:12px; line-height:1.2;'>"
    + "<br>".join(integrantes_equipo) +
    "</div>",
    unsafe_allow_html=True
)

# Crear copias filtradas
df_filtrado_todos = df_limpio.copy()
df_filtrado_equipo = df_equipo.copy()

# Aplicar filtros en cascada
if filtro_rh:
    df_filtrado_todos = df_filtrado_todos[df_filtrado_todos['RH'].isin(filtro_rh)]
    df_filtrado_equipo = df_filtrado_equipo[df_filtrado_equipo['RH'].isin(filtro_rh)]

if filtro_cabello:
    df_filtrado_todos = df_filtrado_todos[df_filtrado_todos['Color_Cabello'].isin(filtro_cabello)]
    df_filtrado_equipo = df_filtrado_equipo[df_filtrado_equipo['Color_Cabello'].isin(filtro_cabello)]

if filtro_barrio:
    df_filtrado_todos = df_filtrado_todos[df_filtrado_todos['Barrio_Residencia'].isin(filtro_barrio)]
    df_filtrado_equipo = df_filtrado_equipo[df_filtrado_equipo['Barrio_Residencia'].isin(filtro_barrio)]


# === FILTRO DE RANGO DE EDAD ===
if rango_edad:
    df_filtrado_todos = df_filtrado_todos[
        (df_filtrado_todos['Edad'] >= rango_edad[0]) & (df_filtrado_todos['Edad'] <= rango_edad[1])
    ]
    df_filtrado_equipo = df_filtrado_equipo[
        (df_filtrado_equipo['Edad'] >= rango_edad[0]) & (df_filtrado_equipo['Edad'] <= rango_edad[1])
    ]

# === FILTRO DE RANGO DE ESTATURA ===
if rango_estatura:
    df_filtrado_todos = df_filtrado_todos[
        (df_filtrado_todos['Estatura_cm'] >= rango_estatura[0]) & (df_filtrado_todos['Estatura_cm'] <= rango_estatura[1])
    ]
    df_filtrado_equipo = df_filtrado_equipo[
        (df_filtrado_equipo['Estatura_cm'] >= rango_estatura[0]) & (df_filtrado_equipo['Estatura_cm'] <= rango_estatura[1])
    ]

col1, col2 = st.columns(2)


with col1:
    st.subheader("Dataframe Procesado (Todos los Estudiantes)")
    st.caption("Este archivo fue procesado para convertir estaturas a centímetros, calcular el IMC y clasificarlo, "
               "además de eliminar columnas vacías y reemplazar valores faltantes (incluyendo comas o celdas vacías) "
               "con el promedio o la moda según el tipo de dato. Tambien se normalizaron algunos datos claves ya que algunos usuario colocaban algunas cosas en mayusculas y otros en minusculas esto se detecto en las columnas Barrio_Residencia y Color_Cabello")
    st.dataframe(df_filtrado_todos, use_container_width=True)


with col2:
    st.subheader("Dataframe Procesado (Equipo de Trabajo)")
    st.caption("Este archivo fue procesado para convertir estaturas a centímetros, calcular el IMC y clasificarlo, "
               "además de eliminar columnas vacías y reemplazar valores faltantes (incluyendo comas o celdas vacías) "
               "con el promedio o la moda según el tipo de dato. Tambien se normalizaron algunos datos claves ya que algunos usuario colocaban algunas cosas en mayusculas y otros en minusculas esto se detecto en las columnas Barrio_Residencia y Color_Cabello")
    st.dataframe(df_filtrado_equipo, use_container_width=True)

indicadores_todos = {
    "Edad": round(df_filtrado_todos["Edad"].mean(), 2),
    "Estatura": round(df_filtrado_todos["Estatura"].mean(), 2),
    "Peso": round(df_filtrado_todos["Peso"].mean(), 2),
    "IMC": round(df_filtrado_todos["IMC"].mean(), 2)
}

indicadores_equipo = {
    "Edad": round(df_filtrado_equipo["Edad"].mean(), 2),
    "Estatura": round(df_filtrado_equipo["Estatura"].mean(), 2),
    "Peso": round(df_filtrado_equipo["Peso"].mean(), 2),
    "IMC": round(df_filtrado_equipo["IMC"].mean(), 2)
}


st.subheader("Indicadores")

# Titulares
colstitulos = st.columns(2)
colstitulos[0].markdown("### Grupo Completo")
colstitulos[1].markdown("### Mi Equipo")

cols = st.columns([1,1,1,1,1,0.2,1,1,1,1,1])


cols[0].metric("Total Estudiantes", len(df_filtrado_todos))
cols[1].metric("Edad Promedio", indicadores_todos["Edad"])
cols[2].metric("Estatura Promedio", indicadores_todos["Estatura"])
cols[3].metric("Peso Promedio", indicadores_todos["Peso"])
cols[4].metric("IMC Promedio", indicadores_todos["IMC"])


with cols[5]:
    st.markdown("<div style='border-left:2px solid gray;height:60px;margin:auto;'></div>", unsafe_allow_html=True)


cols[6].metric("Total Estudiantes", len(df_filtrado_equipo))
cols[7].metric("Edad Promedio", indicadores_equipo["Edad"])
cols[8].metric("Estatura Promedio", indicadores_equipo["Estatura"])
cols[9].metric("Peso Promedio", indicadores_equipo["Peso"])
cols[10].metric("IMC Promedio", indicadores_equipo["IMC"])


# Crear columnas (4 gráficos lado a lado)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# === 1. Grupo Completo - Edad ===
with col1:
    fig_edad_todos = px.bar(
        df_filtrado_todos.groupby('Edad').size().reset_index(name='Cantidad'),
        x='Edad',
        y='Cantidad',
        title='Distribución por Edad - Grupo Completo',
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_edad_todos, use_container_width=True)

# === 2. Grupo Completo - Tipo de Sangre ===
with col2:
    fig_sangre_todos = px.pie(
        df_filtrado_todos,
        names='RH',
        title='Tipo de Sangre - Grupo Completo',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_sangre_todos, use_container_width=True)



# === 3. Mi Equipo - Edad ===
with col3:
    fig_edad_equipo = px.bar(
        df_filtrado_equipo.groupby('Edad').size().reset_index(name='Cantidad'),
        x='Edad',
        y='Cantidad',
        title='Distribución por Edad - Mi Equipo',
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig_edad_equipo, use_container_width=True)

# === 4. Mi Equipo - Tipo de Sangre ===
with col4:
    fig_sangre_equipo = px.pie(
        df_filtrado_equipo,
        names='RH',
        title='Tipo de Sangre - Mi Equipo',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(fig_sangre_equipo, use_container_width=True)



col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# === 1. Grupo Completo - Scatter Estatura vs Peso ===
with col1:
    fig_scatter_todos = px.scatter(
        df_filtrado_todos,
        x='Estatura',
        y='Peso',
        title='Estatura vs Peso - Grupo Completo',
        color_discrete_sequence=px.colors.qualitative.Set1,
        trendline="ols"  # línea de tendencia opcional
    )
    st.plotly_chart(fig_scatter_todos, use_container_width=True)

# === 2. Grupo Completo - Barras por Color de Cabello ===
with col2:
    fig_cabello_todos = px.bar(
        df_filtrado_todos.groupby('Color_Cabello').size().reset_index(name='Cantidad'),
        x='Color_Cabello',
        y='Cantidad',
        title='Color de Cabello - Grupo Completo',
        color='Color_Cabello',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_cabello_todos, use_container_width=True)



# === 3. Mi Equipo - Scatter Estatura vs Peso ===
with col3:
    fig_scatter_equipo = px.scatter(
        df_filtrado_equipo,
        x='Estatura',
        y='Peso',
        title='Estatura vs Peso - Mi Equipo',
        color_discrete_sequence=px.colors.qualitative.Set2,
        trendline="ols"
    )
    st.plotly_chart(fig_scatter_equipo, use_container_width=True)

# === 4. Mi Equipo - Barras por Color de Cabello ===
with col4:
    fig_cabello_equipo = px.bar(
        df_filtrado_equipo.groupby('Color_Cabello').size().reset_index(name='Cantidad'),
        x='Color_Cabello',
        y='Cantidad',
        title='Color de Cabello - Mi Equipo',
        color='Color_Cabello',
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(fig_cabello_equipo, use_container_width=True)

    # Crear columnas (2 + separador + 2)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# === 1. Grupo Completo - Línea (Tallas de Zapatos) ===
with col1:
    fig_tallas_todos = px.line(
        df_filtrado_todos.groupby('Talla_Zapato').size().reset_index(name='Cantidad'),
        x='Talla_Zapato',
        y='Cantidad',
        markers=True,
        title='Distribución de Tallas - Grupo Completo',
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_tallas_todos, use_container_width=True)

with col2:
    top_barrios_todos = (
        df_filtrado_todos['Barrio_Residencia']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_barrios_todos.columns = ['Barrio_Residencia', 'Cantidad']  # ✅ renombrar correctamente
    fig_barrios_todos = px.bar(
        top_barrios_todos,
        x='Barrio_Residencia',
        y='Cantidad',
        title='Top 10 Barrios - Grupo Completo',
        color='Barrio_Residencia',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_barrios_todos, use_container_width=True)


# === 3. Mi Equipo - Línea (Tallas de Zapatos) ===
with col3:
    fig_tallas_equipo = px.line(
        df_filtrado_equipo.groupby('Talla_Zapato').size().reset_index(name='Cantidad'),
        x='Talla_Zapato',
        y='Cantidad',
        markers=True,
        title='Distribución de Tallas - Mi Equipo',
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig_tallas_equipo, use_container_width=True)

with col4:
    top_barrios_equipo = (
        df_filtrado_equipo['Barrio_Residencia']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_barrios_equipo.columns = ['Barrio_Residencia', 'Cantidad']  # ✅ renombrar correctamente
    fig_barrios_equipo = px.bar(
        top_barrios_equipo,
        x='Barrio_Residencia',
        y='Cantidad',
        title='Top 10 Barrios - Mi Equipo',
        color='Barrio_Residencia',
        color_discrete_sequence=px.colors.sequential.Greens
    )
    st.plotly_chart(fig_barrios_equipo, use_container_width=True)

st.subheader("Top 5 de Mayor Estatura y Mayor Peso (Grupo Completo)")

# === Top 5 por Estatura ===
top_estatura = df_filtrado_todos.nlargest(5, 'Estatura_cm')[['Nombre_Completo', 'Estatura_cm']]
# Cambia 'Nombre' por la columna real que tenga el nombre del estudiante

# === Top 5 por Peso ===
top_peso = df_filtrado_todos.nlargest(5, 'Peso')[['Nombre_Completo', 'Peso']]
# Cambia 'Nombre' por el nombre de la columna que uses

# === Mostrar en dos columnas ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 5 Mayor Estatura")
    st.dataframe(top_estatura.reset_index(drop=True))

with col2:
    st.markdown("### Top 5 Mayor Peso")
    st.dataframe(top_peso.reset_index(drop=True))



# Calcular estadísticas principales
est_min, est_max, est_prom = (
    df_filtrado_todos["Estatura_cm"].min(),
    df_filtrado_todos["Estatura_cm"].max(),
    df_filtrado_todos["Estatura_cm"].mean()
)

peso_min, peso_max, peso_prom = (
    df_filtrado_todos["Peso"].min(),
    df_filtrado_todos["Peso"].max(),
    df_filtrado_todos["Peso"].mean()
)

imc_min, imc_max, imc_prom = (
    df_filtrado_todos["IMC"].min(),
    df_filtrado_todos["IMC"].max(),
    df_filtrado_todos["IMC"].mean()
)

edad_min, edad_max, edad_prom = (
    df_filtrado_todos["Edad"].min(),
    df_filtrado_todos["Edad"].max(),
    df_filtrado_todos["Edad"].mean()
)

# Redondear a 2 decimales
est_prom, peso_prom, imc_prom ,edad_prom = round(est_prom, 2), round(peso_prom, 2), round(imc_prom, 2), round(edad_prom, 2)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Resumen Estadístico - Grupo Completo")

with col2:
    st.subheader("Resumen Estadístico - Mi Equipo")



# Crear columnas    
col1, col2, col3,col4, col5, col6, col7,col8 = st.columns(8)




# === Estatura ===
with col1:
    st.markdown("### Estatura (cm)")
    st.metric("Promedio", est_prom)
    st.metric("Mínima", est_min)
    st.metric("Máxima", est_max)

# === Peso ===
with col2:
    st.markdown("### Peso (kg)")
    st.metric("Promedio", peso_prom)
    st.metric("Mínimo", peso_min)
    st.metric("Máximo", peso_max)

# === IMC ===
with col3:
    st.markdown("### IMC")
    st.metric("Promedio", imc_prom)
    st.metric("Mínimo", round(imc_min, 2))
    st.metric("Máximo", round(imc_max, 2))

# === IMC ===
with col4:
    st.markdown("### Edad")
    st.metric("Promedio", edad_prom)
    st.metric("Mínimo", round(edad_min, 2))
    st.metric("Máximo", round(edad_max, 2))


# Calcular estadísticas principales
est_min, est_max, est_prom = (
    df_filtrado_equipo["Estatura_cm"].min(),
    df_filtrado_equipo["Estatura_cm"].max(),
    df_filtrado_equipo["Estatura_cm"].mean()
)

peso_min, peso_max, peso_prom = (
    df_filtrado_equipo["Peso"].min(),
    df_filtrado_equipo["Peso"].max(),
    df_filtrado_equipo["Peso"].mean()
)

imc_min, imc_max, imc_prom = (
    df_filtrado_equipo["IMC"].min(),
    df_filtrado_equipo["IMC"].max(),
    df_filtrado_equipo["IMC"].mean()
)

edad_min, edad_max, edad_prom = (
    df_filtrado_equipo["Edad"].min(),
    df_filtrado_equipo["Edad"].max(),
    df_filtrado_equipo["Edad"].mean()
)

# Redondear a 2 decimales
est_prom, peso_prom, imc_prom ,edad_prom = round(est_prom, 2), round(peso_prom, 2), round(imc_prom, 2), round(edad_prom, 2)


# === Estatura ===
with col5:
    st.markdown("### Estatura (cm)")
    st.metric("Promedio", est_prom)
    st.metric("Mínima", est_min)
    st.metric("Máxima", est_max)

# === Peso ===
with col6:
    st.markdown("### Peso (kg)")
    st.metric("Promedio", peso_prom)
    st.metric("Mínimo", peso_min)
    st.metric("Máximo", peso_max)

# === IMC ===
with col7:
    st.markdown("### IMC")
    st.metric("Promedio", imc_prom)
    st.metric("Mínimo", round(imc_min, 2))
    st.metric("Máximo", round(imc_max, 2))

with col8:
    st.markdown("### Edad")
    st.metric("Promedio", edad_prom)
    st.metric("Mínimo", round(edad_min, 2))
    st.metric("Máximo", round(edad_max, 2))

st.markdown("""
<hr style='border:1px solid #ccc; margin-top:50px;'>
<p style='text-align:center; color:#6B7280; font-size:16px;'>
Muchas gracias por la atención prestada.<br> — Programación Avanzada (2025-2)
</p>
""", unsafe_allow_html=True)


