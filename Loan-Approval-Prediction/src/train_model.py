from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the loan approval dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"The dataset was not found at: {file_path}"
        )

    dataset = pd.read_csv(file_path)

    if dataset.empty:
        raise ValueError("The dataset is empty.")

    return dataset


def build_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
) -> Pipeline:
    """Create the preprocessing and classification pipeline."""

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def save_confusion_matrix(
    y_test: pd.Series,
    predictions,
    output_file: Path,
) -> None:
    """Create and save the confusion matrix image."""

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Rejected", "Approved"],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Rejected", "Approved"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Loan Approval - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    project_directory = Path(__file__).resolve().parent.parent
    data_file = project_directory / "data" / "loan_data.csv"
    models_directory = project_directory / "models"

    model_file = models_directory / "loan_approval_model.pkl"
    metrics_file = models_directory / "model_metrics.txt"
    confusion_matrix_file = (
        models_directory / "confusion_matrix.png"
    )

    models_directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_file)

    features = dataset.drop(columns=["Loan_Status"])
    target = dataset["Loan_Status"]

    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    numerical_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model_pipeline = build_pipeline(
        categorical_features,
        numerical_features,
    )

    print("=" * 60)
    print("TRAINING LOAN APPROVAL MODEL")
    print("=" * 60)
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print()

    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["Rejected", "Approved"],
        zero_division=0,
    )

    joblib.dump(model_pipeline, model_file)

    save_confusion_matrix(
        y_test,
        predictions,
        confusion_matrix_file,
    )

    metrics_content = (
        "LOAN APPROVAL MODEL METRICS\n"
        "===========================\n\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-score:  {f1:.4f}\n\n"
        "Classification report:\n"
        f"{report}"
    )

    metrics_file.write_text(
        metrics_content,
        encoding="utf-8",
    )

    print("MODEL TRAINED SUCCESSFULLY")
    print("-" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print()
    print("Classification report:")
    print(report)
    print(f"Model saved at: {model_file}")
    print(f"Metrics saved at: {metrics_file}")
    print(
        "Confusion matrix saved at: "
        f"{confusion_matrix_file}"
    )


if __name__ == "__main__":
    main()