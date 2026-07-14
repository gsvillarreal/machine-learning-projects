# Loan Approval Prediction

Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on applicant financial and personal information.

## Project Overview

This project uses a Random Forest classification model and a complete Scikit-learn preprocessing pipeline.

The application includes:

- Synthetic loan dataset generation
- Data preprocessing
- Categorical variable encoding
- Numerical feature scaling
- Random Forest model training
- Model evaluation
- Individual predictions
- Interactive Streamlit application

## Model Performance

The trained model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 88.33% |
| Precision | 93.58% |
| Recall | 91.62% |
| F1-score | 92.59% |

## Input Features

The model uses the following variables:

- Gender
- Marital status
- Number of dependents
- Education
- Self-employment status
- Applicant income
- Coapplicant income
- Loan amount
- Loan term
- Credit history
- Property area

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Project Structure

```text
Loan-Approval-Prediction/
├── data/
│   └── loan_data.csv
├── models/
│   ├── confusion_matrix.png
│   ├── loan_approval_model.pkl
│   └── model_metrics.txt
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── predict.py
│   └── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
Installation

Clone the repository and enter the project directory:

git clone YOUR_REPOSITORY_URL
cd Loan-Approval-Prediction

Create and activate a virtual environment:

python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install the dependencies:

pip install -r requirements.txt
Generate the Dataset
python src/generate_data.py
Train the Model
python src/train_model.py
Run a Prediction
python src/predict.py
Run the Streamlit Application
streamlit run app.py

Then open the local URL shown in the terminal.

Disclaimer

This project is an educational Machine Learning demonstration. It must not be used to make real financial or lending decisions.

Author

Gabriel Villarreal
Artificial Intelligence Engineer
Machine Learning · Deep Learning · NLP · Python