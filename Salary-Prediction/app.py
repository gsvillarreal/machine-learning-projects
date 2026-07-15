from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide",
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIRECTORY / "models" / "salary_model.joblib"


# ---------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """Carga el modelo entrenado desde el archivo joblib."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo. "
            "Ejecuta primero: python train_model.py"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error("No fue posible cargar el modelo.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

st.title("💰 Salary Prediction")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning para estimar
    el salario anual de una persona según su edad, educación, cargo,
    experiencia, tamaño de empresa y modalidad de trabajo remoto.
    """
)

st.divider()


# ---------------------------------------------------------
# FORMULARIO
# ---------------------------------------------------------

st.subheader("Ingresa los datos profesionales")

with st.form("salary_prediction_form"):

    left_column, right_column = st.columns(2)

    with left_column:

        age = st.slider(
            label="Edad",
            min_value=21,
            max_value=60,
            value=30,
            step=1,
        )

        education_level = st.selectbox(
            label="Nivel educativo",
            options=[
                "High School",
                "Bachelor",
                "Master",
                "PhD",
            ],
        )

        job_title = st.selectbox(
            label="Cargo profesional",
            options=[
                "Data Analyst",
                "Software Developer",
                "Systems Engineer",
                "Data Scientist",
                "Machine Learning Engineer",
                "DevOps Engineer",
                "Cybersecurity Analyst",
                "Project Manager",
                "Product Manager",
                "Database Administrator",
            ],
        )

    with right_column:

        max_experience = max(age - 18, 0)

        years_experience = st.slider(
            label="Años de experiencia",
            min_value=0,
            max_value=max_experience,
            value=min(5, max_experience),
            step=1,
        )

        company_size = st.selectbox(
            label="Tamaño de la empresa",
            options=[
                "Small",
                "Medium",
                "Large",
            ],
        )

        remote_ratio = st.select_slider(
            label="Porcentaje de trabajo remoto",
            options=[0, 50, 100],
            value=50,
            format_func=lambda value: f"{value} %",
        )

    submitted = st.form_submit_button(
        label="Predecir salario",
        use_container_width=True,
    )


# ---------------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Education_Level": education_level,
                "Job_Title": job_title,
                "Years_Experience": years_experience,
                "Company_Size": company_size,
                "Remote_Ratio": remote_ratio,
            }
        ]
    )

    predicted_salary = model.predict(input_data)[0]

    st.divider()
    st.subheader("Resultado de la predicción")

    result_column, information_column = st.columns(2)

    with result_column:

        st.metric(
            label="Salario anual estimado",
            value=f"${predicted_salary:,.2f}",
        )

        monthly_salary = predicted_salary / 12

        st.metric(
            label="Salario mensual aproximado",
            value=f"${monthly_salary:,.2f}",
        )

    with information_column:

        st.write("**Datos utilizados por el modelo:**")

        st.write(f"Edad: {age} años")
        st.write(f"Nivel educativo: {education_level}")
        st.write(f"Cargo: {job_title}")
        st.write(f"Experiencia: {years_experience} años")
        st.write(f"Empresa: {company_size}")
        st.write(f"Trabajo remoto: {remote_ratio} %")

    st.info(
        "Esta estimación fue generada con fines educativos usando "
        "un dataset sintético. No representa una oferta salarial real."
    )


# ---------------------------------------------------------
# INFORMACIÓN DEL MODELO
# ---------------------------------------------------------

st.divider()

with st.expander("Información técnica del proyecto"):

    st.write(
        """
        **Algoritmo:** Random Forest Regressor

        **Preprocesamiento:** One-Hot Encoding para variables categóricas

        **Variables de entrada:**
        - Edad
        - Nivel educativo
        - Cargo profesional
        - Años de experiencia
        - Tamaño de empresa
        - Porcentaje de trabajo remoto

        **Variable objetivo:** Salario anual
        """
    )

    st.write("**Resultados obtenidos durante el entrenamiento:**")

    metrics_data = pd.DataFrame(
        {
            "Métrica": [
                "MAE",
                "RMSE",
                "R²",
            ],
            "Resultado": [
                "$6,161.12",
                "$7,792.35",
                "0.8926",
            ],
        }
    )

    st.dataframe(
        metrics_data,
        hide_index=True,
        use_container_width=True,
    )