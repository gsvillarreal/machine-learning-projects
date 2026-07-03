# 🏠 House Price Prediction

A complete Machine Learning project that predicts house prices using property features. This project includes data exploration, preprocessing, visualization, model training, evaluation, and model serialization using Scikit-Learn.

---

# 📌 Overview

The objective of this project is to build a Linear Regression model capable of estimating house prices from different house characteristics such as:

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Preferred Area
- Furnishing Status

---

# 📂 Dataset

Dataset: **Housing.csv**

Number of observations:

- **545 houses**

Number of features:

- **13 original variables**

Target variable:

- **Price**

The dataset contains both numerical and categorical variables.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebook

---

# 📊 Exploratory Data Analysis (EDA)

The project includes:

- Dataset inspection
- Shape analysis
- Data types
- Missing value verification
- Statistical summary
- Price distribution visualization
- Area vs Price relationship visualization

---

# ⚙ Data Preprocessing

The preprocessing stage includes:

- Checking missing values
- Converting categorical variables using One-Hot Encoding
- Feature selection
- Train/Test split (80% / 20%)

---

# 🤖 Machine Learning Model

Model used:

**Linear Regression**

The model was trained using Scikit-Learn.

---

# 📈 Model Performance

Evaluation Metrics

| Metric | Value |
|---------|--------|
| MAE | 970,043 |
| RMSE | 1,324,507 |
| R² Score | 0.653 |

The model explains approximately **65%** of the variability in house prices.

---

# 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── Housing.csv
│
├── images/
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── src/
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
```

Enter the project

```bash
cd House-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook

```bash
jupyter notebook
```

Run:

```
house_price_prediction.ipynb
```

---

# 💾 Saved Model

The trained model is stored as:

```
models/house_price_model.pkl
```

It can be loaded using:

```python
import joblib

model = joblib.load("models/house_price_model.pkl")
```

---

# 📌 Future Improvements

- Random Forest Regressor
- XGBoost Regressor
- Feature Engineering
- Hyperparameter Tuning
- Cross Validation
- Deployment with Streamlit

---

# 👨‍💻 Author

**Gabriel Villarreal**

Artificial Intelligence Engineer

Python • Machine Learning • Deep Learning • Data Science

---

# ✅ Project Status

**Completed**