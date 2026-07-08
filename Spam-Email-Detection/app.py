import streamlit as st
import joblib

# ==========================
# Cargar modelo y vectorizador
# ==========================

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ==========================
# Configuración
# ==========================

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Email Detection")
st.write("Escribe un mensaje y el modelo predecirá si es SPAM o HAM.")

mensaje = st.text_area("Mensaje")

if st.button("Predecir"):

    if mensaje.strip() == "":
        st.warning("Por favor escribe un mensaje.")
    else:

        texto = vectorizer.transform([mensaje])

        pred = model.predict(texto)

        if pred[0] == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM")