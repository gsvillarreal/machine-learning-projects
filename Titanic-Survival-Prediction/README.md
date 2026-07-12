# 🚢 Titanic Survival Prediction

A complete Machine Learning classification project that predicts whether a Titanic passenger would have survived based on personal and travel information.

The project includes data exploration, preprocessing, model training, model comparison, model serialization, and an interactive Streamlit application.

---

## 📌 Project Overview

The objective of this project is to build a classification model capable of predicting passenger survival using historical Titanic data.

The application allows users to enter passenger information and obtain:

- A survival prediction
- An estimated survival probability
- A summary of the information submitted to the model

---

## 📊 Dataset

The dataset contains information about 891 Titanic passengers and 12 original columns.

The original variables include:

- `PassengerId`
- `Survived`
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`

The target variable is:

- `Survived`
  - `0`: Did not survive
  - `1`: Survived

---

## 🧠 Features Used

The final model uses the following seven features:

| Feature | Description |
|---|---|
| `Pclass` | Passenger ticket class |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Fare` | Ticket fare paid |
| `Embarked` | Port of embarkation |

The following columns were not used directly in this version of the model:

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

---

## 📁 Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
│   └── titanic.csv
│
├── images/
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
```

---

## 🔍 Exploratory Data Analysis

The notebook includes an exploratory analysis of:

- Dataset dimensions and data types
- Missing values
- Survival distribution
- Survival according to passenger sex
- Survival according to passenger class

Important observations:

- Female passengers had a considerably higher survival rate
- First-class passengers had a higher survival rate
- Third-class passengers represented the largest number of non-survivors
- `Age`, `Cabin`, and `Embarked` contained missing values

---

## ⚙️ Data Preprocessing

A Scikit-learn preprocessing pipeline was created to ensure that the same transformations are applied during training and prediction.

### Numerical variables

The numerical variables are:

- `Age`
- `SibSp`
- `Parch`
- `Fare`

The following transformations are applied:

- Missing values are replaced using the median
- Numerical values are standardized using `StandardScaler`

### Categorical variables

The categorical variables are:

- `Pclass`
- `Sex`
- `Embarked`

The following transformations are applied:

- Missing values are replaced using the most frequent value
- Categories are transformed using One-Hot Encoding
- Unknown categories are handled automatically

---

## 🤖 Models Evaluated

Two classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest Classifier

The dataset was divided into:

- 80% training data
- 20% testing data

A stratified split was used to preserve the proportion of survivors and non-survivors.

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8045 | 0.7297 | 0.7826 | 0.7552 | 0.8405 |
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8427 |

---

## 🏆 Selected Model

Random Forest was selected as the final model because it achieved:

- The highest F1-score
- The highest recall
- A good balance between identifying survivors and non-survivors

The trained pipeline was saved as:

```text
models/titanic_survival_model.joblib
```

The saved object includes both:

- Data preprocessing
- Random Forest classification model

This allows the Streamlit application to receive raw passenger information and transform it automatically before prediction.

---

## 🧪 Logistic Regression Confusion Matrix

The Logistic Regression model produced the following results on the test dataset:

| Actual / Predicted | Did not survive | Survived |
|---|---:|---:|
| Did not survive | 98 | 12 |
| Survived | 23 | 46 |

The model correctly classified 144 of the 179 test passengers.

---

## 🖥️ Streamlit Application

The interactive application allows users to provide:

- Passenger class
- Sex
- Age
- Number of siblings or spouses aboard
- Number of parents or children aboard
- Fare paid
- Port of embarkation

The application then displays:

- Predicted survival result
- Estimated survival probability
- Passenger data sent to the model

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook
- Git
- GitHub
- Visual Studio Code

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/gsvillarreal/machine-learning-projects.git
```

Move into the project directory:

```bash
cd machine-learning-projects/Titanic-Survival-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the `Titanic-Survival-Prediction` directory, execute:

```bash
python -m streamlit run app.py
```

Then open the local address displayed in the terminal, normally:

```text
http://localhost:8501
```

---

## 🧾 Example Prediction

Example passenger information:

- First class
- Female
- 29 years old
- No siblings or spouse aboard
- No parents or children aboard
- Fare of 80
- Embarked at Cherbourg

For this passenger profile, the application returned:

```text
Predicted result: Survived
Estimated survival probability: 99.16%
```

The result is only an estimation based on the patterns learned from the historical dataset.

---

## 📓 Notebook

The complete Machine Learning workflow is available in:

```text
notebooks/titanic_survival_prediction.ipynb
```

It contains:

- Library imports
- Dataset loading
- Exploratory Data Analysis
- Missing-value analysis
- Data preprocessing
- Train-test split
- Logistic Regression training
- Random Forest training
- Model evaluation
- Model comparison
- Model selection
- Model serialization
- Prediction test

---

## ⚠️ Disclaimer

This project was developed for educational and portfolio purposes.

Predictions are estimates generated from historical data and should not be interpreted as certain outcomes.

---

## 👨‍💻 Author

**Gabriel Villarreal**

Computer Science Engineer focused on:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Python Development