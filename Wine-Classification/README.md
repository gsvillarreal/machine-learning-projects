# Wine Classification

A machine learning classification project that predicts the category of a wine using its chemical characteristics.

The project uses the Wine Recognition dataset included in scikit-learn and compares multiple classification algorithms to identify the best-performing model.

## Project Overview

The objective of this project is to classify wines into one of three categories based on 13 chemical measurements.

The complete workflow includes:

- Data loading and inspection
- Exploratory data analysis
- Missing-value and duplicate checks
- Feature analysis
- Data visualization
- Train-test splitting
- Feature scaling
- Model comparison
- Cross-validation
- Final model evaluation
- Model persistence with Joblib
- Interactive prediction application with Streamlit

## Dataset

The project uses the Wine Recognition dataset available in scikit-learn.

The dataset contains:

- 178 samples
- 13 numerical input features
- 3 wine classes
- No missing values
- No duplicate records

## Input Features

The model uses the following 13 chemical characteristics:

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

## Target Classes

The target variable contains three wine categories:

- Class 0
- Class 1
- Class 2

## Exploratory Data Analysis

The notebook includes:

- Dataset dimensions and data types
- Missing-value verification
- Duplicate-record verification
- Descriptive statistics
- Class distribution analysis
- Alcohol distribution by class
- Mean feature values by wine class
- Correlation matrix
- Graphical comparison of model performance

## Models Compared

The following machine learning algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest

The models were compared using 5-fold cross-validation accuracy.

## Model Results

| Model | Mean Cross-Validation Accuracy |
|---|---:|
| Logistic Regression | 99.31% |
| Support Vector Machine | 98.62% |
| Random Forest | 98.62% |
| K-Nearest Neighbors | 96.50% |

Logistic Regression achieved the highest cross-validation accuracy and was selected as the final model.

## Final Model Performance

The selected Logistic Regression model achieved:

- Test accuracy: 97.22%
- High precision across all three wine classes
- High recall across all three wine classes
- High F1-score across all three wine classes
- Strong generalization on unseen test data

The final model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

## Project Structure

```text
Wine-Classification/
│
├── app/
│   └── app.py
│
├── models/
│   ├── wine_classification_model.joblib
│   └── wine_metadata.joblib
│
├── notebooks/
│   └── wine_classification.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Enter the 13 chemical characteristics of a wine
- Generate a wine-class prediction
- View the predicted wine category
- Inspect prediction probabilities
- Compare probabilities using a bar chart
- Review the values entered into the application
- View the selected model and its test accuracy

## Saved Files

The trained model is saved as:

```text
models/wine_classification_model.joblib
```

The project metadata is saved as:

```text
models/wine_metadata.joblib
```

The metadata file contains:

- Feature names
- Class names
- Selected model name
- Test accuracy

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/machine-learning-projects.git
```

Enter the project directory:

```bash
cd machine-learning-projects/Wine-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/wine_classification.ipynb
```

## Run the Streamlit Application

From the main project directory, execute:

```bash
python -m streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Example Prediction

An example wine sample can use these values:

| Feature | Value |
|---|---:|
| Alcohol | 14.23 |
| Malic acid | 1.71 |
| Ash | 2.43 |
| Alcalinity of ash | 15.60 |
| Magnesium | 127 |
| Total phenols | 2.80 |
| Flavanoids | 3.06 |
| Nonflavanoid phenols | 0.28 |
| Proanthocyanins | 2.29 |
| Color intensity | 5.64 |
| Hue | 1.04 |
| OD280/OD315 of diluted wines | 3.92 |
| Proline | 1065 |

Expected prediction:

```text
Wine Class 0
```

## Requirements

The main project dependencies are:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter
- IPykernel

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

## Conclusion

This project demonstrates a complete machine learning classification workflow, from data exploration and model comparison to final deployment through an interactive web application.

Logistic Regression achieved the best cross-validation performance and was selected as the final model.

The trained model was successfully saved, tested and integrated into a Streamlit interface that allows users to classify wines using their chemical characteristics.

## Author

Gabriel Villarreal

Computer Science Engineer focused on Artificial Intelligence, Machine Learning and Python development.