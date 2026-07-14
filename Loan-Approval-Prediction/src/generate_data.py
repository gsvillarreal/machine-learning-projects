from pathlib import Path

import numpy as np
import pandas as pd


def generate_loan_dataset(number_of_records: int = 1200) -> pd.DataFrame:
    """Generate a synthetic dataset for loan approval prediction."""

    np.random.seed(42)

    gender = np.random.choice(
        ["Male", "Female"],
        size=number_of_records,
        p=[0.65, 0.35],
    )

    married = np.random.choice(
        ["Yes", "No"],
        size=number_of_records,
        p=[0.65, 0.35],
    )

    dependents = np.random.choice(
        ["0", "1", "2", "3+"],
        size=number_of_records,
        p=[0.50, 0.20, 0.18, 0.12],
    )

    education = np.random.choice(
        ["Graduate", "Not Graduate"],
        size=number_of_records,
        p=[0.75, 0.25],
    )

    self_employed = np.random.choice(
        ["Yes", "No"],
        size=number_of_records,
        p=[0.15, 0.85],
    )

    applicant_income = np.random.randint(
        1500,
        15000,
        size=number_of_records,
    )

    coapplicant_income = np.random.randint(
        0,
        8000,
        size=number_of_records,
    )

    loan_amount = np.random.randint(
        50,
        600,
        size=number_of_records,
    )

    loan_amount_term = np.random.choice(
        [120, 180, 240, 300, 360],
        size=number_of_records,
        p=[0.05, 0.10, 0.10, 0.15, 0.60],
    )

    credit_history = np.random.choice(
        [1, 0],
        size=number_of_records,
        p=[0.80, 0.20],
    )

    property_area = np.random.choice(
        ["Urban", "Semiurban", "Rural"],
        size=number_of_records,
        p=[0.35, 0.40, 0.25],
    )

    total_income = applicant_income + coapplicant_income
    monthly_payment = loan_amount * 1000 / loan_amount_term
    payment_ratio = monthly_payment / total_income

    approval_score = (
        credit_history * 4.0
        + (total_income >= 5000) * 1.3
        + (total_income >= 9000) * 1.0
        + (payment_ratio <= 0.04) * 1.2
        + (education == "Graduate") * 0.5
        + (property_area == "Semiurban") * 0.4
        - (loan_amount >= 450) * 1.2
    )

    approval_probability = 1 / (
        1 + np.exp(-(approval_score - 3.5))
    )

    random_values = np.random.random(number_of_records)

    loan_status = np.where(
        random_values < approval_probability,
        "Approved",
        "Rejected",
    )

    dataset = pd.DataFrame(
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
            "Loan_Status": loan_status,
        }
    )

    return dataset


def main() -> None:
    project_directory = Path(__file__).resolve().parent.parent
    data_directory = project_directory / "data"
    output_file = data_directory / "loan_data.csv"

    data_directory.mkdir(parents=True, exist_ok=True)

    dataset = generate_loan_dataset()
    dataset.to_csv(output_file, index=False)

    print("=" * 55)
    print("LOAN APPROVAL DATASET GENERATED SUCCESSFULLY")
    print("=" * 55)
    print(f"Records: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"File: {output_file}")
    print()
    print("Loan status distribution:")
    print(dataset["Loan_Status"].value_counts())


if __name__ == "__main__":
    main()