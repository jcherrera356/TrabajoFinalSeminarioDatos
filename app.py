import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================
# CONFIGURACIÓN GENERAL
# ============================

renombrar_columnas = {
    "PatientID": "ID_Paciente",
    "Age": "Edad",
    "Gender": "Genero",
    "Ethnicity": "Etnicidad",
    "EducationLevel": "Nivel_Educativo",
    "BMI": "IMC",
    "Smoking": "Fuma",
    "AlcoholConsumption": "Consumo_Alcohol",
    "PhysicalActivity": "Actividad_Fisica",
    "DietQuality": "Calidad_Dieta",
    "SleepQuality": "Calidad_Sueno",
    "FamilyHistoryAlzheimers": "Antecedentes_Alzheimer",
    "CardiovascularDisease": "Enfermedad_Cardiovascular",
    "Diabetes": "Diabetes",
    "Depression": "Depresion",
    "HeadInjury": "Lesion_Cabeza",
    "Hypertension": "Hipertension",
    "SystolicBP": "Presion_Sistolica",
    "DiastolicBP": "Presion_Diastolica",
    "CholesterolTotal": "Colesterol_Total",
    "CholesterolLDL": "Colesterol_LDL",
    "CholesterolHDL": "Colesterol_HDL",
    "CholesterolTriglycerides": "Trigliceridos",
    "MMSE": "Puntaje_MMSE",
    "FunctionalAssessment": "Evaluacion_Funcional",
    "MemoryComplaints": "Quejas_Memoria",
    "BehavioralProblems": "Problemas_Comportamiento",
    "ADL": "Actividades_Diarias",
    "Confusion": "Confusion",
    "Disorientation": "Desorientacion",
    "PersonalityChanges": "Cambios_Personalidad",
    "DifficultyCompletingTasks": "Dificultad_Tareas",
    "Forgetfulness": "Olvidos",
    "Diagnosis": "Diagnostico",
    "DoctorInCharge": "Medico_Encargado"
}
st.set_page_config(
    page_title="Informe Alzheimer - Seminario Ciencia de Datos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ESTILOS GLOBALES (TEXTO GRANDE TIPO PRESENTACIÓN)
# ============================
st.markdown("""
    <style>
    .texto-grande {
        font-size: 22px;
        line-height: 1.5;
    }
    .texto-mediano {
        font-size: 20px;
        line-height: 1.5;
    }
    .metric-card {
        padding: 12px 18px;
        border-radius: 10px;
        background-color: #111827;
        border: 1px solid #374151;
        margin-bottom: 10px;
        font-size: 18px;
    }
    .metric-valor {
        color: #10B981;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# NAVBAR (SIDEBAR)
# ============================
st.sidebar.title("Navegación")

opcion = st.sidebar.radio(
    "Ir a:",
    [
        "Inicio",
        "Introducción",
        "Objetivos",
        "Metodología",
        "Dataset",
        "Columnas del Dataset",
        "Análisis Exploratorio",
        "Modelamiento Predictivo",
        "Resultados",
       
        "Conclusiones y Recomendaciones"
    ]
)

# ============================
# CARGA DEL CSV DESDE DRIVE
# ============================
drive_url = "https://drive.google.com/file/d/1KN8H0rPfdtM1F7PMbb4M_2srgsJER6Zu/view?usp=sharing"
file_id = drive_url.split('/d/')[1].split('/')[0]
csv_url = f"https://drive.google.com/uc?id={file_id}"

df_original = pd.read_csv(csv_url, index_col='PatientID')
df_original = df_original.rename(columns=renombrar_columnas)
# ============================
# SECCIONES
# ============================

# ---------------------------- INICIO ----------------------------
if opcion == "Inicio":
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.title("Informe Final – Seminario de Ciencia de los Datos")
        st.subheader("Análisis y modelado predictivo de Alzheimer")

        st.markdown("""
        <div class="texto-grande">
        Trabajo desarrollado por Juan Camilo Herrera Osorio

        - Contexto e introducción del problema.  
        - Objetivo general y objetivos específicos del análisis.  
        - Metodología aplicada en cada etapa.  
        - Exploración y análisis del dataset de pacientes.  
        - Construcción y evaluación de un modelo predictivo.  
        - Discusión, conclusiones y recomendaciones finales.  

       
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(
            "https://ubikare.io/wp-content/uploads/que-siente-una-persona-con-Alzheimer-1-768x513.jpg",
            use_container_width=True
        )

# ---------------------------- INTRODUCCIÓN ----------------------------
elif opcion == "Introducción":
    st.title("Introducción")

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="texto-grande">
        La enfermedad de Alzheimer es uno de los principales trastornos neurodegenerativos 
        y se caracteriza por deterioro progresivo de la memoria, el pensamiento y la capacidad
        para realizar actividades de la vida diaria. Su impacto clínico, social y económico
        la convierte en un problema prioritario para los sistemas de salud.

        En este contexto, la ciencia de los datos proporciona herramientas para analizar 
        grandes volúmenes de información clínica, identificar patrones relevantes y apoyar
        la toma de decisiones mediante modelos predictivos.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(
            "https://www.quironsalud.com/idcsalud-client/cm/images?locale=es_ES&idMmedia=3290416",
            use_container_width=True
        )

    st.markdown("---")

    col3, col4 = st.columns([1, 1.6])

    with col3:
        st.image(
            "https://sagradafamilia.com.ar/wp-content/uploads/2023/07/CLSF-PH-Freepik-Dia-mundial-del-cerebro-scaled.jpg",
            use_container_width=True
        )

    with col4:
        st.markdown("""
        <div class="texto-grande">
        El presente informe aplica técnicas de exploración, visualización y aprendizaje
        automático sobre un dataset de pacientes, con el fin de comprender el comportamiento
        de distintas variables clínicas y construir un modelo que anticipe el diagnóstico
        de Alzheimer.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
<div class="texto-grande">
Este proyecto busca unir el análisis técnico con una explicación clara y comprensible. 
La ciencia de datos no solo se centra en los modelos, sino también en la forma en que 
los resultados se comunican. Por eso, toda la información se presenta de manera organizada 
y visual, permitiendo entender la historia que cuentan los datos sin perder rigurosidad 
ni claridad.
</div>
""", unsafe_allow_html=True)


# ---------------------------- OBJETIVOS ----------------------------
elif opcion == "Objetivos":
    st.title("Objetivo General")

    st.markdown("""
    <div class="texto-grande">
    Aplicar técnicas de ciencia de datos y aprendizaje automático para analizar, procesar 
    y modelar la información de pacientes relacionada con la enfermedad de Alzheimer, con 
    el propósito de identificar patrones clínicos relevantes y construir un modelo predictivo 
    confiable que apoye la toma de decisiones en salud.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.title("Objetivos Específicos")

    st.markdown("""
<div class="texto-grande">
    <ul>
        <li>Realizar un análisis exploratorio detallado para comprender el comportamiento del dataset.</li>
        <li>Identificar valores faltantes, inconsistencias y realizar el preprocesamiento adecuado.</li>
        <li>Analizar correlaciones entre variables clínicas, demográficas y cognitivas.</li>
        <li>Explorar la distribución de variables clave como edad y puntaje MMSE según el diagnóstico.</li>
        <li>Entrenar un modelo predictivo basado en Random Forest para clasificar el diagnóstico.</li>
        <li>Evaluar el desempeño del modelo utilizando métricas como exactitud, precisión, sensibilidad y F1.</li>
        <li>Presentar los resultados en un dashboard interactivo, facilitando su interpretación.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ---------------------------- METODOLOGÍA ----------------------------
elif opcion == "Metodología":
    st.title("Metodología")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
        <div class="texto-grande">

        1. Recolección del dataset proporcionado por la asignatura.  
        2. Limpieza y tratamiento de valores faltantes o inconsistentes.  
        3. Análisis exploratorio y visualización de distribuciones y correlaciones.  
        4. Preparación del dataset para el modelo (selección de variables y partición en entrenamiento y prueba).  
        5. Entrenamiento de un modelo de clasificación basado en Random Forest.  
        6. Evaluación del desempeño del modelo con diferentes métricas.  
        7. Interpretación de resultados y elaboración de conclusiones y recomendaciones.  
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(
            "https://azura.mx/satelite/wp-content/uploads/sites/4/2022/09/Conoce-los-3-tipos-de-alzheimer-que-existen-550x309.jpg",
            use_container_width=True
        )

    st.markdown("""
    <div class="texto-mediano">
    Para el desarrollo se utilizaron herramientas como Python, Pandas, Plotly, Scikit-Learn 
    y Streamlit, lo que permitió integrar el análisis con una interfaz visual interactiva.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------- DATASET ----------------------------
elif opcion == "Dataset":
    st.title("Dataset de Alzheimer")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div class="texto-grande">
        El dataset está compuesto por información clínica, demográfica y cognitiva de pacientes.
        Incluye variables como edad, nivel educativo, hábitos, antecedentes médicos, resultados
        de pruebas cognitivas y el diagnóstico final de Alzheimer.

        A continuación se presenta una vista general de los datos:
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df_original, use_container_width=True)

    with col2:
        st.markdown("<div class='texto-mediano'><strong>Dimensiones del dataset:</strong></div>", unsafe_allow_html=True)
        st.write("Número de pacientes (filas):", df_original.shape[0])
        st.write("Número de variables (columnas):", df_original.shape[1])

        # Distribución del diagnóstico
        dist_diag = df_original["Diagnostico"].value_counts().rename({0: "Sin Alzheimer", 1: "Con Alzheimer"})
        fig_diag = px.bar(
            dist_diag,
            x=dist_diag.index,
            y=dist_diag.values,
            labels={"x": "Diagnóstico", "y": "Cantidad de pacientes"},
            title="Distribución de pacientes según diagnóstico"
        )
        st.plotly_chart(fig_diag, use_container_width=True)

# ---------------------------- COLUMNAS DEL DATASET ----------------------------
elif opcion == "Columnas del Dataset":
    st.title("Descripción de las Columnas")

    columnas_descripcion = {
        "Age": "Edad del paciente.",
        "Gender": "Género (0 = Femenino, 1 = Masculino).",
        "Ethnicity": "Grupo étnico del paciente.",
        "EducationLevel": "Nivel educativo.",
        "BMI": "Índice de masa corporal.",
        "Smoking": "Hábito de fumar.",
        "AlcoholConsumption": "Consumo de alcohol.",
        "PhysicalActivity": "Actividad física.",
        "DietQuality": "Calidad de la dieta.",
        "SleepQuality": "Calidad del sueño.",
        "FamilyHistoryAlzheimers": "Antecedentes familiares.",
        "CardiovascularDisease": "Enfermedades cardiovasculares.",
        "Diabetes": "Presencia de diabetes.",
        "Depression": "Presencia de depresión.",
        "HeadInjury": "Lesión en la cabeza.",
        "Hypertension": "Hipertensión.",
        "SystolicBP": "Presión arterial sistólica.",
        "DiastolicBP": "Presión arterial diastólica.",
        "CholesterolTotal": "Colesterol total.",
        "CholesterolLDL": "Colesterol LDL.",
        "CholesterolHDL": "Colesterol HDL.",
        "CholesterolTriglycerides": "Triglicéridos.",
        "MMSE": "Puntaje de evaluación cognitiva MMSE.",
        "FunctionalAssessment": "Evaluación funcional del paciente.",
        "MemoryComplaints": "Quejas de memoria.",
        "BehavioralProblems": "Problemas de comportamiento.",
        "ADL": "Actividades de la vida diaria.",
        "Confusion": "Presencia de confusión.",
        "Disorientation": "Desorientación.",
        "PersonalityChanges": "Cambios de personalidad.",
        "DifficultyCompletingTasks": "Dificultad para completar tareas.",
        "Forgetfulness": "Olvidos frecuentes.",
        "Diagnosis": "Diagnóstico final de Alzheimer.",
        "DoctorInCharge": "Médico encargado (dato confidencial)."
    }

    # Texto científico por variable
    explicacion_cientifica = {
        "Age": "La edad es el principal factor de riesgo para Alzheimer; aumenta drásticamente después de los 65 años.",
        "Gender": "Las mujeres presentan mayor riesgo por factores hormonales y mayor esperanza de vida.",
        "Ethnicity": "Algunas etnias presentan mayor riesgo debido a genética y acceso desigual a salud.",
        "EducationLevel": "Un bajo nivel educativo disminuye la reserva cognitiva, aumentando el riesgo de demencia.",
        "BMI": "La obesidad y el bajo peso afectan la salud cerebral y aumentan la probabilidad de deterioro.",
        "Smoking": "El tabaquismo incrementa el daño vascular y el estrés oxidativo en el cerebro.",
        "AlcoholConsumption": "El consumo excesivo de alcohol produce daño neuronal asociado al deterioro cognitivo.",
        "PhysicalActivity": "La actividad física protege la salud cerebral y reduce el riesgo de Alzheimer.",
        "DietQuality": "Una mala alimentación aumenta la inflamación y el envejecimiento cerebral.",
        "SleepQuality": "Dormir mal reduce la eliminación de beta-amiloide, favoreciendo Alzheimer.",
        "FamilyHistoryAlzheimers": "Los antecedentes familiares incrementan fuertemente el riesgo genético.",
        "CardiovascularDisease": "La mala salud cardiovascular afecta el flujo sanguíneo cerebral y acelera la demencia.",
        "Diabetes": "La diabetes mal controlada daña los vasos cerebrales y favorece el deterioro cognitivo.",
        "Depression": "La depresión sostenida se asocia a inflamación cerebral y mayor riesgo de Alzheimer.",
        "HeadInjury": "Los traumatismos craneales se relacionan con acumulación anormal de proteínas cerebrales.",
        "Hypertension": "La hipertensión daña vasos cerebrales y aumenta el riesgo de demencia.",
        "SystolicBP": "Una presión sistólica elevada deteriora la perfusión cerebral.",
        "DiastolicBP": "Valores diastólicos anormales afectan la oxigenación y salud del cerebro.",
        "CholesterolTotal": "El colesterol elevado se relaciona con acumulación de placas en el cerebro.",
        "CholesterolLDL": "El LDL alto aumenta inflamación y deterioro vascular cerebral.",
        "CholesterolHDL": "Un HDL bajo reduce la protección neuronal.",
        "CholesterolTriglycerides": "Triglicéridos elevados afectan el metabolismo energético del cerebro.",
        "MMSE": "Es una prueba clave del estado cognitivo: valores bajos indican deterioro compatible con Alzheimer.",
        "FunctionalAssessment": "Una baja función es un marcador directo de deterioro cognitivo y Alzheimer.",
        "MemoryComplaints": "Las quejas de memoria suelen ser un síntoma temprano del deterioro cognitivo.",
        "BehavioralProblems": "Los problemas conductuales aparecen en etapas intermedias y avanzadas de Alzheimer.",
        "ADL": "La pérdida de independencia en actividades diarias es indicativa de deterioro severo.",
        "Confusion": "La confusión frecuente refleja afectación de memoria y orientación.",
        "Disorientation": "La desorientación es un signo típico de deterioro cognitivo avanzado.",
        "PersonalityChanges": "Cambios de personalidad ocurren cuando áreas cerebrales frontales se ven afectadas.",
        "DifficultyCompletingTasks": "Dificultad para realizar tareas es un signo temprano del Alzheimer.",
        "Forgetfulness": "El olvido frecuente es uno de los primeros síntomas reportados por pacientes.",
        "Diagnosis": "Indica si el paciente fue diagnosticado con Alzheimer.",
        "DoctorInCharge": "Dato administrativo sin impacto clínico."
    }

    nulos = df_original.isnull().sum()

    cols = st.columns(3)
    i = 0
    for col, desc in columnas_descripcion.items():
        with cols[i % 3]:
            st.subheader(col)
            st.markdown(f"<div class='texto-mediano'>{desc}</div>", unsafe_allow_html=True)

            # Estado de nulos
            if col in df_original.columns:
                cant = nulos[col]
                estado = "Sin nulos" if cant == 0 else f"{cant} valores nulos"
                st.write("Estado de datos:", estado)

            # Explicación científica
            if col in explicacion_cientifica:
                st.markdown(
                    f"<div class='texto-mediano' style='color:#9CA3AF; font-size:17px;'>{explicacion_cientifica[col]}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
        i += 1

# ---------------------------- ANÁLISIS EXPLORATORIO ----------------------------
elif opcion == "Análisis Exploratorio":
    st.title("Análisis Exploratorio del Dataset")

    # Distribución de edades según diagnóstico
    st.subheader("Distribución de edades según diagnóstico")
    fig = px.histogram(
        df_original,
        x="Edad",
        color="Diagnostico",
        barmode="overlay",
        color_discrete_map={0: "lightblue", 1: "red"},
        labels={"Edad": "Edad", "Diagnostico": "Diagnóstico"},
        title="Distribución de edad para pacientes con y sin Alzheimer"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="texto-mediano">
    La gráfica muestra cómo se distribuye la edad entre los pacientes con diagnóstico positivo y negativo.
    Se observa que la mayoría de casos de Alzheimer se concentran en edades mayores, lo que coincide con
    la literatura que relaciona la enfermedad con el envejecimiento.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # ===========================================
    # DISTRIBUCIÓN DE GÉNERO EN PACIENTES CON ALZHEIMER
    # ===========================================
    
    st.subheader("Distribución por género en pacientes con Alzheimer")
    
    col1, col2 = st.columns([1, 1.2])
    
    # ======== COLUMNA 1: Gráfica ========
    with col1:
    
# Crear columna combinada
        df_original["Grupo"] = df_original.apply(
            lambda row: (
                "Mujeres con Alzheimer" if row["Genero"] == 0 and row["Diagnostico"] == 1 else
                "Mujeres sin Alzheimer" if row["Genero"] == 0 and row["Diagnostico"] == 0 else
                "Hombres con Alzheimer" if row["Genero"] == 1 and row["Diagnostico"] == 1 else
                "Hombres sin Alzheimer"
            ),
            axis=1
        )
        
        conteo_grupos = df_original["Grupo"].value_counts()
        
        fig_torta_completa = px.pie(
            names=conteo_grupos.index,
            values=conteo_grupos.values,
            title="Distribución de género y diagnóstico en el dataset",
            color=conteo_grupos.index,
            color_discrete_map={
                "Mujeres con Alzheimer": "#FF4B6E",   # rojo rosado fuerte
                "Mujeres sin Alzheimer": "#FFB6C1",   # rosado suave
                "Hombres con Alzheimer": "#4B9CD3",   # azul fuerte
                "Hombres sin Alzheimer": "#ADD8E6"    # azul suave
            }
        )
        
        fig_torta_completa.update_traces(textinfo="percent+label")
        
        st.plotly_chart(fig_torta_completa, use_container_width=True)
    
    # ======== COLUMNA 2: Texto + Imagen ========
    with col2:
    
        st.markdown("""
        <div class="texto-mediano">
        Según el análisis de nuestro dataset, <strong>las mujeres presentan una mayor proporción de casos de Alzheimer</strong> en comparación con los hombres.<br><br>
        Este comportamiento coincide con lo reportado por la ciencia, ya que las mujeres son más propensas a desarrollar 
        Alzheimer debido a una combinación de factores biológicos, genéticos, hormonales y psicosociales. 
        </div>
        """, unsafe_allow_html=True)
    
        st.image(
            "https://cloudfront-us-east-1.images.arcpublishing.com/infobae/JLFZROQ5XVASNBVUGKKUDHMVCM.jpg",
            use_container_width=True
        )
    
    st.markdown("---")

# ===========================================
# DISTRIBUCIÓN DEL PUNTAJE MMSE SEGÚN DIAGNÓSTICO
# ===========================================

    st.subheader("Distribución del puntaje MMSE según diagnóstico")
    
    col1Box, col2Box = st.columns([1, 1])
    
    # ------------------ COLUMNA 1: TEXTO EXPLICATIVO ------------------
    with col1Box:
        st.markdown("""
        <div class="texto-mediano">
        El <strong>MMSE (Mini-Mental State Examination)</strong> es una de las pruebas
        más utilizadas en el mundo para evaluar el estado cognitivo de una persona.
        Se usa especialmente para detectar deterioro cognitivo y sospecha de Alzheimer.<br><br>
    
        Esta prueba evalúa:<br>
        • Orientación en tiempo y lugar<br>
        • Memoria inmediata y a corto plazo<br>
        • Atención y cálculo<br>
        • Lenguaje y comprensión<br>
        • Capacidad para seguir instrucciones<br><br>
    
        <strong>Interpretación del puntaje MMSE:</strong><br><br>
    
        • <strong>24 – 30:</strong> Función cognitiva normal.<br>
        • <strong>18 – 23:</strong> Deterioro cognitivo leve.<br>
        • <strong>0 – 17:</strong> Deterioro cognitivo moderado o severo.<br><br>
    
        Valores bajos suelen indicar problemas de memoria, desorientación y deterioro funcional,
        los cuales están estrechamente relacionados con la enfermedad de Alzheimer.
        </div>
        """, unsafe_allow_html=True)
    
    # ------------------ COLUMNA 2: GRÁFICO + ANÁLISIS ------------------
    with col2Box:
    
        fig_mmse = px.box(
            df_original,
            x="Diagnostico",
            y="Puntaje_MMSE",
            labels={"Diagnostico": "Diagnóstico", "Puntaje_MMSE": "Puntaje MMSE"},
            title="Comparación del puntaje cognitivo MMSE entre diagnósticos"
        )
    
        st.plotly_chart(fig_mmse, use_container_width=True)
    
        st.markdown("""
        <div class="texto-mediano">
        En el boxplot se observa que los pacientes diagnosticados con Alzheimer presentan
        <strong>puntajes MMSE mucho más bajos</strong> que quienes no tienen la enfermedad.<br><br>
    
        Esto coincide con lo que indica la ciencia: a medida que el deterioro cognitivo avanza,
        el MMSE disminuye significativamente, siendo uno de los indicadores más utilizados 
        para detectar el Alzheimer.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Mapa de correlaciones
    st.subheader("Mapa de correlaciones")
    corr = df_original.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Mapa de correlación entre variables numéricas"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="texto-mediano">
    El mapa de correlaciones permite identificar qué variables numéricas tienen mayor relación 
    con el diagnóstico de Alzheimer. En el análisis realizado, se observa que las correlaciones 
    más destacadas corresponden a variables directamente asociadas al deterioro cognitivo 
    y funcional del paciente. Las relaciones más importantes fueron:

    - <strong>Evaluacion Funcional (-0.36):</strong> los pacientes con menor desempeño funcional 
      presentan mayor probabilidad de diagnóstico positivo.
    - <strong>Actividades Diarias (-0.33):</strong> la pérdida de independencia en actividades diarias es un 
      indicador fuertemente asociado al Alzheimer.
    - <strong>Puntaje MMSE (-0.23):</strong> un puntaje cognitivo bajo se relaciona con mayor presencia 
      de la enfermedad.
    - <strong>Quejas Memoria (+0.30):</strong> las quejas de memoria muestran una correlación 
      clara con el diagnóstico.
    - <strong>Problemas Comportamiento (+0.22):</strong> los problemas de comportamiento tienden a 
      aumentar la probabilidad de diagnóstico positivo.

    Estas variables representan síntomas y manifestaciones clínicas directas de la enfermedad, 
    por lo que su alta correlación es consistente con la literatura médica. Asimismo, variables 
    como colesterol, presión arterial o diabetes muestran correlaciones bajas, indicando que en 
    este dataset no son determinantes para la clasificación del Alzheimer.
    </div>
    """, unsafe_allow_html=True)

    # ============================
    # HEATMAP DE CORRELACIONES VS Diagnostico
    # ============================
    
    st.subheader("Correlación de cada variable con el Diagnóstico de Alzheimer")
    
    # Calcular correlaciones solo contra Diagnostico
    correlaciones_diag = df_original.corr(numeric_only=True)["Diagnostico"].sort_values(ascending=False)
    
    # Crear dataframe ordenado
    df_correlaciones = pd.DataFrame({
        "Variable": correlaciones_diag.index,
        "Correlación": correlaciones_diag.values
    })
    
    # Heatmap horizontal
    fig_corr_diag = px.imshow(
        df_correlaciones[["Correlación"]].T,
        labels=dict(x="Variable", y="", color="Correlación"),
        x=df_correlaciones["Variable"],
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    
    fig_corr_diag.update_layout(
        title="Heatmap de correlación de cada variable vs Diagnostico",
        xaxis_tickangle=45,
        height=350
    )
    
    st.plotly_chart(fig_corr_diag, use_container_width=True)
    
    # Texto explicativo
    st.markdown("""
    <div class="texto-mediano">
    Este heatmap muestra exclusivamente la correlación de cada variable con el diagnóstico de Alzheimer.
    Las variables están ordenadas de mayor a menor correlación absoluta, lo que permite identificar 
    rápidamente cuáles tienen mayor influencia en la clasificación del paciente.
    
    Se observa que las variables más asociadas al diagnóstico son:
    
    - <strong>Evaluacion Funcional</strong>  
    - <strong>Actividades Diarias</strong>  
    - <strong>Puntaje MMSE</strong>  
    - <strong>Quejas Memoria</strong>  
    - <strong>Problemas Comportamiento</strong>  
    
    Estas variables presentan los valores más altos (positivos y negativos), confirmando que 
    los factores cognitivos y funcionales son los que más determinan el diagnóstico en este dataset.
    </div>
     """, unsafe_allow_html=True)
    

    st.markdown("---")
    st.subheader("Nivel educativo según diagnóstico")

    fig_edu = px.histogram(
        df_original,
        x="Nivel_Educativo",
        color="Diagnostico",
        barmode="group",
        color_discrete_map={0: "lightblue", 1: "red"},
        title="Comparación del nivel educativo entre diagnósticos"
    )
    st.plotly_chart(fig_edu, use_container_width=True)

    st.markdown("""
    <div class="texto-mediano">
    En estudios epidemiológicos, un menor nivel educativo se asocia con mayor riesgo de Alzheimer.
    Esto se explica por la teoría de la <strong>reserva cognitiva</strong>: las personas con más años
    de educación desarrollan más conexiones neuronales que las protegen frente al deterioro cognitivo.
    <br><br>
    En nuestro dataset se observa una tendencia similar: los pacientes con Alzheimer se concentran más
    en niveles educativos bajos y medios, mientras que los niveles altos tienen menor presencia de la
    enfermedad.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Clustering de pacientes (K-Means)")
    
    # Seleccionar pocas variables muy relevantes
    variables_cluster = [
        "Edad",
        "Puntaje_MMSE",
        "Evaluacion_Funcional",
        "Actividades_Diarias"
    ]
    
    df_cluster = df_original[variables_cluster].dropna()
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # K-Means con 3 grupos
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster["Cluster"] = clusters
    
    # Gráfica 3D mejor organizada
    fig_cluster = px.scatter_3d(
        df_cluster,
        x="Puntaje_MMSE",
        y="Evaluacion_Funcional",
        z="Edad",
        color="Cluster",
        title="Clustering de pacientes usando variables clínicas clave",
        color_continuous_scale="Viridis",
        opacity=0.8,             # puntos más suaves
        height=600
    )
    
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # EXPLICACIÓN
    st.markdown("""
    <div class="texto-mediano">
    El clustering se realizó usando **Edad**, **Puntaje MMSE**, **Evaluación Funcional** y 
    **Actividades Diarias**, ya que son las variables más asociadas al Alzheimer según la literatura científica.
    
    ### Interpretación de los clústeres:
    <ul>
    <li><strong>Cluster 0:</strong> Puntajes MMSE altos, buena funcionalidad, mayor independencia. Pacientes típicos <strong>sin deterioro cognitivo</strong>.</li>
    
    <li><strong>Cluster 1:</strong> Puntajes MMSE bajos, mala evaluación funcional, y baja capacidad para actividades diarias. Perfil de <strong>deterioro severo</strong> compatible con Alzheimer.</li>
    
    <li><strong>Cluster 2:</strong> Valores intermedios: MMSE medio, funciones parcialmente afectadas. Posibles <strong>casos leves o etapa temprana</strong> del deterioro cognitivo.</li>
    </ul>
    
    Este análisis permite identificar patrones sin usar el diagnóstico real, lo cual es importante para 
    predecir la progresión temprana de la enfermedad y diferenciar entre niveles de gravedad.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    # ===========================================
    # FUMADORES vs DIAGNÓSTICO DE ALZHEIMER
    # ===========================================

    st.subheader("Relación entre el hábito de fumar y el diagnóstico de Alzheimer")

    colFum1, colFum2 = st.columns([1.2, 1])

    # ---------------------- COLUMNA IZQUIERDA (IMAGEN + TEXTO) ----------------------
    with colFum1:
        st.image(
            "https://isanidad.com/wp-content/uploads/2014/07/fumador_alzheimer.jpg",
            use_container_width=True
        )

        st.markdown("""
        <div class="texto-mediano">
        El tabaquismo es un factor de riesgo bien documentado en enfermedades neurológicas.
        La ciencia ha demostrado que fumar:

        • Aumenta el estrés oxidativo en el cerebro.<br>
        • Reduce el flujo sanguíneo cerebral.<br>
        • Acelera la inflamación y la neurodegeneración.<br>
        • Favorece la acumulación de beta-amiloide, proteína clave en el Alzheimer.<br><br>

        Diversos estudios indican que los fumadores tienen **mayor riesgo de deterioro cognitivo**,
        especialmente en edades avanzadas.  
        <br>
        Esta comparación permite evaluar si nuestro dataset refleja esa misma tendencia.
        </div>
        """, unsafe_allow_html=True)


    # ---------------------- COLUMNA DERECHA (GRÁFICA) ----------------------
    with colFum2:

    # Crear columna combinada
        df_original["Grupo_Fumar"] = df_original.apply(
            lambda row: (
                "Fumador con Alzheimer" if row["Fuma"] == 1 and row["Diagnostico"] == 1 else
                "Fumador sin Alzheimer" if row["Fuma"] == 1 and row["Diagnostico"] == 0 else
                "No fumador con Alzheimer" if row["Fuma"] == 0 and row["Diagnostico"] == 1 else
                "No fumador sin Alzheimer"
            ),
            axis=1
        )

        # Conteo y porcentaje
        conteo_fumar = df_original["Grupo_Fumar"].value_counts()
        porcentajes_fumar = (conteo_fumar / conteo_fumar.sum()) * 100

        # Gráfica con porcentajes
        fig_fumar = px.bar(
            x=porcentajes_fumar.index,
            y=porcentajes_fumar.values,
            title="Distribución de fumadores y diagnóstico de Alzheimer",
            labels={"x": "Grupo", "y": "Porcentaje (%)"},
            color=porcentajes_fumar.index,
            color_discrete_map={
                "Fumador con Alzheimer": "#D9534F",
                "Fumador sin Alzheimer": "#F5B7B1",
                "No fumador con Alzheimer": "#5DADE2",
                "No fumador sin Alzheimer": "#AED6F1"
            }
        )

        fig_fumar.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        fig_fumar.update_layout(xaxis_tickangle=20)

        st.plotly_chart(fig_fumar, use_container_width=True)


        st.markdown("""
      <div class="texto-mediano">
        Según datos oficiales de la Organización Mundial de la Salud (OMS), alrededor del 
        <strong>14% de los casos de Alzheimer en el mundo están relacionados con el consumo de tabaco</strong>.
        Esto se debe a que fumar acelera procesos de inflamación, daño vascular y estrés oxidativo en el cerebro,
        afectando regiones clave para la memoria y el razonamiento.

        Este dato es especialmente relevante porque muestra que el tabaquismo no solo afecta los pulmones o el sistema
        cardiovascular, sino que también <strong>aumenta el riesgo de deterioro cognitivo y demencia</strong>.  
        Comparar este comportamiento con nuestro dataset permite evidenciar si los pacientes fumadores presentan
        mayor probabilidad de diagnóstico positivo o perfiles compatibles con deterioro cognitivo.
        </div>

        """, unsafe_allow_html=True)




 


# ---------------------------- MODELAMIENTO PREDICTIVO ----------------------------
elif opcion == "Modelamiento Predictivo":
    st.title("Modelamiento Predictivo del Diagnóstico de Alzheimer")

    st.markdown("""
    <div class="texto-grande">
    En esta sección se construye un modelo de aprendizaje automático para predecir el diagnóstico
    de Alzheimer a partir de las variables disponibles en el dataset. Se utiliza un clasificador 
    Random Forest, adecuado para datos tabulares y capaz de manejar relaciones no lineales.
    </div>
    """, unsafe_allow_html=True)

    # Preparación de datos
    st.subheader("Preparación del Dataset")

    df_modelo = df_original.copy()
    if "Medico_Encargado" in df_modelo.columns:
        df_modelo = df_modelo.drop(columns=["Medico_Encargado"])

    X = df_modelo.drop(columns=["Diagnostico"])
    y = df_modelo["Diagnostico"]

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    st.markdown(
        f"<div class='texto-mediano'>Total de registros: {len(df_modelo)} | "
        f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}</div>",
        unsafe_allow_html=True
    )

    # Entrenamiento
    st.subheader("Entrenamiento del Modelo Random Forest")

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    # Métricas
    st.subheader("Métricas del Modelo")

    accuracy = accuracy_score(y_test, predicciones)
    precision = precision_score(y_test, predicciones)
    recall = recall_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
    <div class="metric-card">
        <strong>Exactitud (Accuracy):</strong> 
        <span class="metric-valor">{accuracy*100:.1f}%</span><br><br>
        <div style="font-size:16px; color:#D1D5DB;">
            Indica el porcentaje total de predicciones correctas.<br>
            Mide qué tanto acierta el modelo en general.
        </div>
        <hr style="opacity:0.2; margin:12px 0;">
        <strong>Precisión (Precision):</strong><br>
        <span class="metric-valor">{precision*100:.1f}%</span>
        <div style="font-size:16px; color:#D1D5DB; margin-top:6px;">
            De los pacientes que el modelo dijo que <b>SÍ</b> tenían Alzheimer,<br>
            ¿cuántos realmente lo tenían?<br>
            Evalúa qué tan confiables son los positivos.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
    <div class="metric-card">
        <strong>Sensibilidad (Recall):</strong> 
        <span class="metric-valor">{recall*100:.1f}%</span><br><br>
        <div style="font-size:16px; color:#D1D5DB;">
            De los pacientes que realmente tenían Alzheimer,<br>
            ¿cuántos logró detectar el modelo?<br>
            Evita falsos negativos.
        </div>
        <hr style="opacity:0.2; margin:12px 0;">
        <strong>Puntaje F1:</strong><br>
        <span class="metric-valor">{f1*100:.1f}%</span>
        <div style="font-size:16px; color:#D1D5DB; margin-top:6px;">
            Combina precisión y sensibilidad.<br>
            Resume el desempeño global del modelo en una sola métrica.
        </div>
    </div>
    """, unsafe_allow_html=True)


 

    st.markdown("---")

# ================================
# MATRIZ DE CONFUSIÓN + IMPORTANCIA EN DOS COLUMNAS
# ================================

    col_izq, col_der = st.columns(2)
    
    # -------------------- MATRIZ DE CONFUSIÓN --------------------
    with col_izq:
        st.subheader("Matriz de Confusión")
        matriz = confusion_matrix(y_test, predicciones)
    
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)
    
        st.markdown("""
        <div class="texto-mediano">
        La matriz de confusión permite observar cuántos pacientes fueron clasificados correctamente
        como con o sin Alzheimer, así como los errores de clasificación. El número de verdaderos positivos 
        y verdaderos negativos es considerablemente mayor que el de falsos positivos y falsos negativos,
        lo que confirma un buen comportamiento del modelo.
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------- IMPORTANCIA DE VARIABLES --------------------
    with col_der:
        st.subheader("Importancia de Variables")
        importancias = modelo.feature_importances_
        variables = X.columns
    
        df_importancias = pd.DataFrame({
            "Variable": variables,
            "Importancia": importancias
        }).sort_values(by="Importancia", ascending=False)
    
        fig2 = px.bar(
            df_importancias,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de las variables en el modelo Random Forest"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
        st.markdown("""
        <div class="texto-mediano">
        Las variables con mayor importancia corresponden principalmente al puntaje MMSE, la evaluación funcional,
        la edad y algunos factores clínicos asociados a la salud cardiovascular. Esto respalda la idea de que 
        el deterioro cognitivo y ciertas condiciones médicas influyen de manera directa en el diagnóstico de Alzheimer.
        </div>
        """, unsafe_allow_html=True)


elif opcion == "Resultados":
    st.title("Resultados del Análisis")

    st.markdown("""
    <div class="texto-mediano">
    
    <ul>
        <li><strong>El análisis exploratorio confirmó patrones consistentes con la literatura científica:</strong>
            <ul>
                <li>Mayor prevalencia en mujeres.</li>
                <li>Menor nivel educativo asociado a diagnóstico positivo.</li>
                <li>Puntaje MMSE y evaluación funcional disminuyen notablemente con la enfermedad.</li>
            </ul>
        </li>
        <li><strong>El clustering mostró tres grupos claros de pacientes:</strong>
            <ul>
                <li>Un grupo sano.</li>
                <li>Un grupo intermedio.</li>
                <li>Un grupo con deterioro severo, coincidente con los diagnosticados.</li>
            </ul>
        </li>
        <li><strong>Las correlaciones destacaron que los factores cognitivos y funcionales son los principales marcadores del Alzheimer en este dataset.</strong></li>
        <li><strong>El modelo Random Forest alcanzó 94% de exactitud</strong>, superando el rendimiento típico esperado para modelos simples en datasets sintéticos.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    




elif opcion == "Conclusiones y Recomendaciones":
    st.title("Conclusiones del Informe")

    st.markdown("""
<div class="texto-grande">
A partir del análisis realizado se concluye:
<ul>
<li><strong>El análisis de datos permitió identificar patrones clínicos y cognitivos</strong>
asociados al diagnóstico de Alzheimer.</li>

<li><strong>Las variables cognitivas y funcionales fueron las más influyentes</strong>
en el diagnóstico, tanto en las correlaciones como en el modelo predictivo.</li>

<li><strong>El modelo Random Forest presentó un desempeño sólido</strong>,
con alta exactitud, precisión, sensibilidad y F1.</li>

<li><strong>La visualización interactiva facilitó la interpretación de resultados</strong>,
permitiendo conectar las gráficas con el análisis estadístico.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.title("Recomendaciones")

    st.markdown("""
<div class="texto-mediano">
<ul>
<li><strong>Ampliar el dataset</strong> con pacientes de distintos entornos clínicos
para mejorar la generalización del modelo.</li>

<li><strong>Integrar el modelo en pruebas piloto</strong> dentro de un sistema clínico
mediante APIs, para evaluar su rendimiento en tiempo real.</li>

<li><strong>Realizar análisis longitudinales</strong> que permitan estudiar la evolución
de los pacientes y construir modelos más precisos.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.success("Muchas gracias.")
