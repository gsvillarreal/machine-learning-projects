from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


MODEL_PATH = Path(__file__).parent / "models" / "titanic_survival_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error("No se pudo cargar el modelo.")
    st.code(str(error))
    st.stop()


st.title("🚢 Titanic Survival Prediction")

st.write(
    "Esta aplicación utiliza un modelo de Machine Learning "
    "para estimar si un pasajero del Titanic habría sobrevivido."
)

st.divider()

st.subheader("Datos del pasajero")


class_options = {
    "Primera clase": 1,
    "Segunda clase": 2,
    "Tercera clase": 3
}

selected_class = st.selectbox(
    "Clase del pasajero",
    options=list(class_options.keys())
)

pclass = class_options[selected_class]


sex_options = {
    "Mujer": "female",
    "Hombre": "male"
}

selected_sex = st.selectbox(
    "Sexo",
    options=list(sex_options.keys())
)

sex = sex_options[selected_sex]


age = st.slider(
    "Edad",
    min_value=0,
    max_value=100,
    value=30,
    step=1
)


sibsp = st.number_input(
    "Hermanos o pareja a bordo",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


parch = st.number_input(
    "Padres o hijos a bordo",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


fare = st.number_input(
    "Tarifa pagada",
    min_value=0.0,
    max_value=600.0,
    value=32.0,
    step=1.0
)


embarked_options = {
    "Southampton": "S",
    "Cherbourg": "C",
    "Queenstown": "Q"
}

selected_embarked = st.selectbox(
    "Puerto de embarque",
    options=list(embarked_options.keys())
)

embarked = embarked_options[selected_embarked]


st.divider()


if st.button(
    "Realizar predicción",
    type="primary",
    use_container_width=True
):
    passenger_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(passenger_data)[0]

    probability = model.predict_proba(
        passenger_data
    )[0][1]

    if prediction == 1:
        st.success("Resultado: El pasajero habría sobrevivido")
    else:
        st.error("Resultado: El pasajero no habría sobrevivido")

    st.metric(
        label="Probabilidad estimada de supervivencia",
        value=f"{probability:.2%}"
    )

    st.progress(float(probability))

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(
            passenger_data,
            use_container_width=True
        )


st.divider()

st.caption(
    "Proyecto educativo de Machine Learning. "
    "La predicción es una estimación basada en datos históricos."
)