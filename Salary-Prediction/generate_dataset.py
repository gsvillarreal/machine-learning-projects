from pathlib import Path

import numpy as np
import pandas as pd


# Semilla para obtener siempre los mismos resultados
RANDOM_SEED = 42
NUMBER_OF_RECORDS = 1500

rng = np.random.default_rng(RANDOM_SEED)


# Puestos incluidos en el dataset
job_base_salaries = {
    "Data Analyst": 42000,
    "Software Developer": 48000,
    "Systems Engineer": 52000,
    "Data Scientist": 60000,
    "Machine Learning Engineer": 68000,
    "DevOps Engineer": 62000,
    "Cybersecurity Analyst": 58000,
    "Project Manager": 65000,
    "Product Manager": 70000,
    "Database Administrator": 55000,
}


# Bonificación aproximada según el nivel educativo
education_bonus = {
    "High School": 0,
    "Bachelor": 7000,
    "Master": 14000,
    "PhD": 22000,
}


# Bonificación aproximada según el tamaño de la empresa
company_bonus = {
    "Small": 0,
    "Medium": 6000,
    "Large": 13000,
}


def generate_salary_dataset(number_of_records: int) -> pd.DataFrame:
    """Genera un dataset sintético para predicción salarial."""

    ages = rng.integers(21, 61, size=number_of_records)

    years_experience = np.array(
        [
            rng.integers(0, max(1, age - 18))
            for age in ages
        ]
    )

    education_levels = rng.choice(
        list(education_bonus.keys()),
        size=number_of_records,
        p=[0.15, 0.50, 0.28, 0.07],
    )

    job_titles = rng.choice(
        list(job_base_salaries.keys()),
        size=number_of_records,
    )

    company_sizes = rng.choice(
        list(company_bonus.keys()),
        size=number_of_records,
        p=[0.30, 0.45, 0.25],
    )

    remote_ratios = rng.choice(
        [0, 50, 100],
        size=number_of_records,
        p=[0.40, 0.30, 0.30],
    )

    salaries = []

    for (
        job_title,
        education,
        experience,
        company_size,
        remote_ratio,
    ) in zip(
        job_titles,
        education_levels,
        years_experience,
        company_sizes,
        remote_ratios,
    ):
        base_salary = job_base_salaries[job_title]
        experience_bonus = experience * 2200
        academic_bonus = education_bonus[education]
        business_bonus = company_bonus[company_size]
        remote_bonus = remote_ratio * 40
        random_variation = rng.normal(0, 6000)

        salary = (
            base_salary
            + experience_bonus
            + academic_bonus
            + business_bonus
            + remote_bonus
            + random_variation
        )

        salaries.append(round(max(salary, 25000), 2))

    dataset = pd.DataFrame(
        {
            "Age": ages,
            "Education_Level": education_levels,
            "Job_Title": job_titles,
            "Years_Experience": years_experience,
            "Company_Size": company_sizes,
            "Remote_Ratio": remote_ratios,
            "Salary": salaries,
        }
    )

    return dataset


def main() -> None:
    """Crea y guarda el archivo CSV."""

    project_directory = Path(__file__).resolve().parent
    data_directory = project_directory / "data"
    output_file = data_directory / "salary_data.csv"

    data_directory.mkdir(parents=True, exist_ok=True)

    dataset = generate_salary_dataset(NUMBER_OF_RECORDS)
    dataset.to_csv(output_file, index=False)

    print("=" * 60)
    print("DATASET GENERADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Archivo: {output_file}")
    print(f"Registros: {len(dataset)}")
    print(f"Columnas: {len(dataset.columns)}")
    print("\nPrimeras cinco filas:")
    print(dataset.head())
    print("\nInformación estadística:")
    print(dataset.describe())


if __name__ == "__main__":
    main()