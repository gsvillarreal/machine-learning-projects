from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷",
    layout="wide",
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "wine_classification_model.joblib"
METADATA_PATH = MODEL_DIR / "wine_metadata.joblib"


# ---------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------
@st.cache_resource
def load_project_files():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    trained_model = joblib.load(MODEL_PATH)
    project_metadata = joblib.load(METADATA_PATH)

    return trained_model, project_metadata


try:
    model, metadata = load_project_files()
except Exception as error:
    st.error("The trained model or metadata could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Dataset information for input limits
# ---------------------------------------------------------
wine_data = load_wine()

dataset_df = pd.DataFrame(
    wine_data.data,
    columns=wine_data.feature_names,
)

feature_names = metadata["feature_names"]
class_names = metadata["class_names"]
best_model_name = metadata["best_model_name"]
test_accuracy = metadata["test_accuracy"]


# ---------------------------------------------------------
# Friendly names
# ---------------------------------------------------------
friendly_feature_names = {
    "alcohol": "Alcohol",
    "malic_acid": "Malic Acid",
    "ash": "Ash",
    "alcalinity_of_ash": "Alcalinity of Ash",
    "magnesium": "Magnesium",
    "total_phenols": "Total Phenols",
    "flavanoids": "Flavanoids",
    "nonflavanoid_phenols": "Nonflavanoid Phenols",
    "proanthocyanins": "Proanthocyanins",
    "color_intensity": "Color Intensity",
    "hue": "Hue",
    "od280/od315_of_diluted_wines": "OD280/OD315 of Diluted Wines",
    "proline": "Proline",
}

friendly_class_names = {
    "class_0": "Wine Class 0",
    "class_1": "Wine Class 1",
    "class_2": "Wine Class 2",
}


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🍷 Wine Classification")

st.write(
    """
    This application predicts the category of a wine from its chemical
    characteristics using a trained machine learning classification model.
    """
)

st.divider()


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

with metric_column_1:
    st.metric(
        label="Selected Model",
        value=best_model_name,
    )

with metric_column_2:
    st.metric(
        label="Test Accuracy",
        value=f"{test_accuracy * 100:.2f}%",
    )

with metric_column_3:
    st.metric(
        label="Number of Features",
        value=len(feature_names),
    )

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("About the project")

    st.write(
        """
        The model classifies wines into three categories using the
        Wine Recognition dataset included in scikit-learn.
        """
    )

    st.subheader("Available classes")

    for class_name in class_names:
        st.write(
            f"- {friendly_class_names.get(class_name, class_name)}"
        )

    st.subheader("Instructions")

    st.write(
        """
        1. Enter the chemical measurements.
        2. Press **Predict Wine Class**.
        3. Review the predicted class and probabilities.
        """
    )


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.header("Wine Chemical Characteristics")

st.write(
    "Adjust the values below or keep the dataset averages as an example."
)

input_values = {}

with st.form("wine_prediction_form"):
    columns = st.columns(2)

    for index, feature in enumerate(feature_names):
        feature_minimum = float(dataset_df[feature].min())
        feature_maximum = float(dataset_df[feature].max())
        feature_average = float(dataset_df[feature].mean())

        input_step = max(
            (feature_maximum - feature_minimum) / 100,
            0.01,
        )

        with columns[index % 2]:
            input_values[feature] = st.number_input(
                label=friendly_feature_names.get(
                    feature,
                    feature.replace("_", " ").title(),
                ),
                min_value=feature_minimum,
                max_value=feature_maximum,
                value=feature_average,
                step=input_step,
                format="%.4f",
            )

    predict_button = st.form_submit_button(
        "Predict Wine Class",
         width="stretch",
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_button:
    input_dataframe = pd.DataFrame(
        [[input_values[feature] for feature in feature_names]],
        columns=feature_names,
    )

    predicted_class_index = int(
        model.predict(input_dataframe)[0]
    )

    predicted_class_name = class_names[predicted_class_index]

    displayed_class_name = friendly_class_names.get(
        predicted_class_name,
        predicted_class_name,
    )

    st.divider()
    st.header("Prediction Result")

    st.success(
        f"The predicted category is: **{displayed_class_name}**"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            input_dataframe
        )[0]

        probability_dataframe = pd.DataFrame(
            {
                "Wine Class": [
                    friendly_class_names.get(
                        class_name,
                        class_name,
                    )
                    for class_name in class_names
                ],
                "Probability": probabilities,
            }
        )

        probability_dataframe["Probability Percentage"] = (
            probability_dataframe["Probability"] * 100
        ).round(2)

        st.subheader("Prediction Probabilities")

        st.dataframe(
           probability_dataframe[
               ["Wine Class", "Probability Percentage"]
         ],
         width="stretch",
         hide_index=True,
        )

        chart_dataframe = probability_dataframe.set_index(
            "Wine Class"
        )[["Probability"]]

        st.bar_chart(chart_dataframe)

    with st.expander("View entered values"):
        st.dataframe(
            input_dataframe.T.rename(
                columns={0: "Entered Value"}
            ),
            width="stretch",
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Machine Learning portfolio project developed with "
    "Python, scikit-learn and Streamlit."
)