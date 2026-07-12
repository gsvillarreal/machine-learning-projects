# Titanic Survival Prediction

A Machine Learning project that predicts whether a passenger aboard the Titanic would have survived based on personal and travel information.

## Project Overview

This project uses historical Titanic passenger data to build and compare classification models.

The application allows users to enter passenger information and receive:

- A survival prediction
- An estimated survival probability
- A summary of the data submitted to the model

## Dataset

The dataset contains 891 passenger records and 12 original variables.

The model uses the following features:

- Passenger class
- Sex
- Age
- Number of siblings or spouses aboard
- Number of parents or children aboard
- Fare paid
- Port of embarkation

The target variable is:

- Survived

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
│   └── titanic.csv
│
├── models/
│   └── titanic_survival_model.joblib
│
├── notebooks/
│   └── titanic_survival_prediction.ipynb
│
├── app.py
├── README.md
└── requirements.txt