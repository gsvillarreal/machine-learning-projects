from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIRECTORY = Path(__file__).resolve().parent
MODEL_FILE = (
    PROJECT_DIRECTORY
    / "models"
    / "loan_approval_model.pkl"
)


@st.cache_resource
def load_model():
    """Load the trained loan approval model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Run src/train_model.py first."
        )

    return joblib.load(MODEL_FILE)


def predict_application(
    model,
    gender: str,
    married: str,
    dependents: str,
    education: str,
    self_employed: str,
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_amount_term: float,
    credit_history: int,
    property_area: str,
) -> tuple[str, float]:
    """Generate the prediction and its confidence."""

    applicant_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )

    prediction = model.predict(applicant_data)[0]

    probabilities = model.predict_proba(applicant_data)[0]
    class_names = model.classes_

    probability_by_class = dict(
        zip(class_names, probabilities)
    )

    confidence = probability_by_class[prediction]

    return prediction, confidence


def main() -> None:
    st.set_page_config(
        page_title="Loan Approval Prediction",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Loan Approval Prediction")
    st.write(
        "Complete the applicant information to estimate "
        "whether the loan will be approved or rejected."
    )

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.form("loan_application_form"):
        st.subheader("Applicant information")

        column_1, column_2, column_3 = st.columns(3)

        with column_1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            married = st.selectbox(
                "Married",
                ["Yes", "No"],
            )

            dependents = st.selectbox(
                "Dependents",
                ["0", "1", "2", "3+"],
            )

            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"],
            )

        with column_2:
            self_employed = st.selectbox(
                "Self-employed",
                ["No", "Yes"],
            )

            applicant_income = st.number_input(
                "Applicant income",
                min_value=0.0,
                value=7500.0,
                step=100.0,
            )

            coapplicant_income = st.number_input(
                "Coapplicant income",
                min_value=0.0,
                value=2500.0,
                step=100.0,
            )

            loan_amount = st.number_input(
                "Loan amount",
                min_value=1.0,
                value=180.0,
                step=10.0,
                help="Loan amount expressed in thousands.",
            )

        with column_3:
            loan_amount_term = st.selectbox(
                "Loan term",
                [120, 180, 240, 300, 360],
                index=4,
            )

            credit_history_label = st.selectbox(
                "Credit history",
                [
                    "Good credit history",
                    "No credit history",
                ],
            )

            credit_history = (
                1
                if credit_history_label
                == "Good credit history"
                else 0
            )

            property_area = st.selectbox(
                "Property area",
                ["Urban", "Semiurban", "Rural"],
                index=1,
            )

        submit_button = st.form_submit_button(
            "Predict loan approval",
            use_container_width=True,
        )

    if submit_button:
        prediction, confidence = predict_application(
            model=model,
            gender=gender,
            married=married,
            dependents=dependents,
            education=education,
            self_employed=self_employed,
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_amount_term=loan_amount_term,
            credit_history=credit_history,
            property_area=property_area,
        )

        st.divider()
        st.subheader("Prediction result")

        result_column, confidence_column = st.columns(2)

        with result_column:
            if prediction == "Approved":
                st.success("Loan application approved")
            else:
                st.error("Loan application rejected")

        with confidence_column:
            st.metric(
                "Model confidence",
                f"{confidence:.2%}",
            )

        st.progress(float(confidence))

        st.caption(
            "This project is an educational Machine Learning "
            "demonstration and must not be used as a real "
            "financial decision system."
        )


if __name__ == "__main__":
    main()