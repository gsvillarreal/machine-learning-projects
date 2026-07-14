from pathlib import Path

import joblib
import pandas as pd


def load_model():
    """Load the trained loan approval model."""

    project_directory = Path(__file__).resolve().parent.parent
    model_file = (
        project_directory
        / "models"
        / "loan_approval_model.pkl"
    )

    if not model_file.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Run train_model.py first."
        )

    return joblib.load(model_file)


def predict_loan_approval(
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
    """Predict whether a loan application will be approved."""

    model = load_model()

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

    prediction_probability = probability_by_class[prediction]

    return prediction, prediction_probability


def main() -> None:
    prediction, probability = predict_loan_approval(
        gender="Male",
        married="Yes",
        dependents="1",
        education="Graduate",
        self_employed="No",
        applicant_income=7500,
        coapplicant_income=2500,
        loan_amount=180,
        loan_amount_term=360,
        credit_history=1,
        property_area="Semiurban",
    )

    print("=" * 55)
    print("LOAN APPROVAL PREDICTION")
    print("=" * 55)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {probability:.2%}")


if __name__ == "__main__":
    main()