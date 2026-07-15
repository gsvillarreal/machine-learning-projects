# 💰 Salary Prediction

Machine Learning project that predicts an estimated annual salary based on professional and employment information.

The project includes data generation, preprocessing, model training, evaluation, model persistence, and an interactive web application built with Streamlit.

---

## 📌 Project Overview

The application estimates a person's annual salary using the following information:

- Age
- Education level
- Job title
- Years of experience
- Company size
- Remote work percentage

The prediction is generated using a Random Forest regression model.

---

## 🤖 Machine Learning Model

The project uses:

- **Algorithm:** Random Forest Regressor
- **Categorical preprocessing:** One-Hot Encoding
- **Train/test split:** 80% training and 20% testing
- **Dataset size:** 1,500 records
- **Target variable:** Annual salary

Categorical features:

- Education level
- Job title
- Company size

Numerical features:

- Age
- Years of experience
- Remote work percentage

---

## 📊 Model Performance

The trained model obtained the following results:

| Metric | Result |
|---|---:|
| Mean Absolute Error | $6,161.12 |
| Root Mean Squared Error | $7,792.35 |
| R² Score | 0.8926 |

The model explains approximately **89.26% of the salary variation** in the test dataset.

> The dataset used in this project is synthetic and was created for educational purposes.

---

## 🗂️ Project Structure

```text
Salary-Prediction/
│
├── data/
│   └── salary_data.csv
│
├── models/
│   └── salary_model.joblib
│
├── notebooks/
├── src/
│   └── __init__.py
│
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/gsvillarreal/machine-learning-projects.git
```

Enter the project directory:

```bash
cd machine-learning-projects/Salary-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Generate the Dataset

The project already includes the generated dataset. To generate it again, run:

```bash
python generate_dataset.py
```

This command creates:

```text
data/salary_data.csv
```

---

## 🧠 Train the Model

To train and evaluate the model, run:

```bash
python train_model.py
```

The trained pipeline will be saved as:

```text
models/salary_model.joblib
```

---

## 🚀 Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

---

## ⚠️ Disclaimer

This project uses synthetic data and was developed for educational and portfolio purposes.

The predictions do not represent real employment offers or guaranteed market salaries.

---

## 👨‍💻 Author

**Gabriel Villarreal**

Computer Science Engineer focused on Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, and Python development.