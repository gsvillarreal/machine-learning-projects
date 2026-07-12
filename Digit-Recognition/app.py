from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered",
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "digit_recognition_cnn.keras"


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_digit_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------
def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """
    Convert the uploaded image into the same format used by MNIST:
    - Grayscale
    - White digit over black background
    - 28 x 28 pixels
    - Pixel values between 0 and 1
    - Shape: (1, 28, 28, 1)
    """

    grayscale_image = image.convert("L")

    # Automatically invert images with a light background
    image_array = np.array(grayscale_image)

    if image_array.mean() > 127:
        grayscale_image = ImageOps.invert(grayscale_image)

    resized_image = grayscale_image.resize(
        (28, 28),
        Image.Resampling.LANCZOS,
    )

    processed_array = np.array(
        resized_image,
        dtype=np.float32,
    )

    processed_array = processed_array / 255.0

    processed_array = np.expand_dims(
        processed_array,
        axis=-1,
    )

    processed_array = np.expand_dims(
        processed_array,
        axis=0,
    )

    return processed_array, resized_image


# ---------------------------------------------------------
# Application interface
# ---------------------------------------------------------
st.title("🔢 Handwritten Digit Recognition")

st.write(
    """
    Upload an image containing a handwritten digit from **0 to 9**.
    The convolutional neural network will analyze the image and predict
    the corresponding number.
    """
)

st.info(
    "For the best result, use a clear image with one centered digit."
)


# ---------------------------------------------------------
# Load trained CNN
# ---------------------------------------------------------
try:
    model = load_digit_model()

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg"],
)


if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)

        st.subheader("Uploaded image")

        st.image(
            original_image,
            caption="Original image",
            width=250,
        )

        processed_array, processed_image = preprocess_image(
            original_image
        )

        st.subheader("Processed image")

        st.image(
            processed_image,
            caption="Image converted to 28 × 28 pixels",
            width=250,
        )

        probabilities = model.predict(
            processed_array,
            verbose=0,
        )[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        st.success(
            f"Predicted digit: {predicted_digit}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%",
        )

        st.subheader("Probability by digit")

        probability_data = {
            str(digit): float(probability)
            for digit, probability in enumerate(probabilities)
        }

        st.bar_chart(probability_data)

        with st.expander("View detailed probabilities"):
            for digit, probability in enumerate(probabilities):
                st.write(
                    f"Digit {digit}: {probability * 100:.2f}%"
                )

    except Exception as error:
        st.error(
            "The uploaded image could not be processed."
        )
        st.exception(error)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "CNN model trained with TensorFlow and the MNIST dataset."
)