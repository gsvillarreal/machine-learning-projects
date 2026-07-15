from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TEST_SIZE = 0.20


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Carga el dataset salarial desde un archivo CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {file_path}"
        )

    dataset = pd.read_csv(file_path)

    if dataset.empty:
        raise ValueError("El dataset está vacío.")

    return dataset


def build_model() -> Pipeline:
    """Construye el pipeline de preprocesamiento y regresión."""

    categorical_columns = [
        "Education_Level",
        "Job_Title",
        "Company_Size",
    ]

    numerical_columns = [
        "Age",
        "Years_Experience",
        "Remote_Ratio",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns,
            ),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return pipeline


def evaluate_model(
    actual_values: pd.Series,
    predicted_values,
) -> dict:
    """Calcula las métricas principales del modelo."""

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    mse = mean_squared_error(
        actual_values,
        predicted_values,
    )

    rmse = mse**0.5

    r2 = r2_score(
        actual_values,
        predicted_values,
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def main() -> None:
    """Entrena, evalúa y guarda el modelo."""

    project_directory = Path(__file__).resolve().parent
    dataset_path = project_directory / "data" / "salary_data.csv"
    models_directory = project_directory / "models"
    model_path = models_directory / "salary_model.joblib"

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO DE PREDICCIÓN SALARIAL")
    print("=" * 60)

    dataset = load_dataset(dataset_path)

    print(f"\nDataset cargado: {dataset_path}")
    print(f"Filas: {dataset.shape[0]}")
    print(f"Columnas: {dataset.shape[1]}")

    features = dataset.drop(columns=["Salary"])
    target = dataset["Salary"]

    print("\nVariables utilizadas:")
    for column in features.columns:
        print(f"- {column}")

    print("\nVariable objetivo:")
    print("- Salary")

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    print("\nDivisión del dataset:")
    print(f"Datos de entrenamiento: {len(features_train)}")
    print(f"Datos de prueba: {len(features_test)}")

    model = build_model()

    print("\nEntrenando modelo...")
    model.fit(
        features_train,
        target_train,
    )

    print("Modelo entrenado correctamente.")

    predictions = model.predict(features_test)

    metrics = evaluate_model(
        target_test,
        predictions,
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    print(f"MAE:  ${metrics['mae']:,.2f}")
    print(f"MSE:  {metrics['mse']:,.2f}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")

    comparison = pd.DataFrame(
        {
            "Actual_Salary": target_test.values,
            "Predicted_Salary": predictions,
        }
    )

    comparison["Absolute_Error"] = (
        comparison["Actual_Salary"]
        - comparison["Predicted_Salary"]
    ).abs()

    print("\nEjemplos de predicciones:")
    print(comparison.head(10).round(2))

    joblib.dump(model, model_path)

    print("\n" + "=" * 60)
    print("MODELO GUARDADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Archivo: {model_path}")


if __name__ == "__main__":
    main()