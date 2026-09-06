# 📉 Customer_Churn_ML

A complete beginner-friendly Machine Learning project on a **Customer Churn dataset**. This notebook walks through every stage of the ML lifecycle — from raw data to a trained model — with clear code, observations, and explanations.

---

## 📋 Project Overview

| | |
|---|---|
| **Dataset** | Customer Churn Dataset |
| **Total Rows** | 64,374 |
| **Total Columns** | 12 |
| **Goal** | Predict whether a customer will churn or not |
| **Type** | Binary Classification (0 = Stayed, 1 = Churned) |
| **Algorithm** | Logistic Regression |
| **Language** | Python |
| **Tools** | Google Colab, Pandas, Seaborn, Scikit-learn |

---

## 🗺️ ML Lifecycle — Stages Covered

| # | Stage | Emoji | Status |
|---|---|---|---|
| 1 | Problem Definition | 🎯 | ✅ Done |
| 2 | Data Collection | 📦 | ✅ Done |
| 3 | Data Cleaning & Preprocessing | 🧹 | ✅ Done |
| 4 | Exploratory Data Analysis (EDA) | 📊 | ✅ Done |
| 5 | Feature Engineering & Selection | ⚙️ | ✅ Done |
| 6 | Model Selection & Building | 🤖 | ✅ Done |
| 7 | Model Evaluation | 📈 | ✅ Done |

---

## 📁 Project Structure

```
Customer_Churn_ML/
│
├── Customer_Churn_ML_GREEKS.ipynb                  ← Main notebook
├── customer_churn_dataset-testing-master.csv       ← Dataset
├── README.md                                       ← This file
└── LICENSE                                         ← MIT License
```

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| CustomerID | int | Unique customer ID — not used in model |
| Age | int | Age of the customer |
| Gender | text | Male / Female |
| Tenure | int | How long customer has been with company (months) |
| Usage Frequency | int | How often customer uses the service |
| Support Calls | int | Number of support calls made |
| Payment Delay | int | Days of payment delay |
| Subscription Type | text | Basic / Standard / Premium |
| Contract Length | text | Monthly / Quarterly / Annual |
| Total Spend | int | Total money spent by customer |
| Last Interaction | int | Days since last interaction |
| Churn | int | 0 = Stayed, 1 = Churned ← Target Variable |

---

## 🎯 Stage 1 — Problem Definition

- **Goal:** Predict whether a customer will churn (leave the service)
- **Type:** Binary Classification
- **Target Variable:** `Churn` (0 = Stayed, 1 = Churned)
- **Business Value:** Help companies identify at-risk customers before they leave
- **Success Metric:** Accuracy Score, Confusion Matrix

---

## 📦 Stage 2 — Data Collection

- Dataset loaded from CSV file
- **64,374 rows** and **12 columns**
- Data covers customer demographics, usage behavior, and payment history

---

## 🧹 Stage 3 — Data Cleaning & Preprocessing

### 3.1 — Data Cleaning:

| Step | Action | Result |
|---|---|---|
| Duplicates | Checked and removed | Verified |
| Irrelevant columns | Dropped CustomerID | 12 → 11 columns |
| Missing values | Checked all columns | Handled |
| Datatype correction | Fixed incorrect types | Corrected |
| Outliers | Detected using IQR + Boxplot | Handled |

### 3.2 — Preprocessing:
- Inspected data structure and shape
- Statistical summary generated
- Outliers visualized using boxplots
- Correlation analysis performed
- Features and target variable separated

---

## 📊 Stage 4 — Exploratory Data Analysis (EDA)

### Univariate Analysis:
- Age Distribution
- Total Spend Distribution
- Churn Distribution (0 vs 1)
- Subscription Type Distribution

### Bivariate Analysis:
- Age vs Churn
- Total Spend vs Churn
- Gender vs Churn
- Contract Length vs Churn
- Support Calls vs Churn

### Multivariate Analysis:
- Correlation Heatmap of all columns
- Age + Gender + Churn combined
- Contract Length + Total Spend + Churn combined

---

## ⚙️ Stage 5 — Feature Engineering & Selection

### 🔤➡️🔢 Encoding:

| Column | Method | Result |
|---|---|---|
| Gender | Label Encoding | Male=0, Female=1 |
| Subscription Type | One Hot Encoding | Subscription_Basic, Subscription_Standard, Subscription_Premium |
| Contract Length | One Hot Encoding | Contract_Monthly, Contract_Quarterly, Contract_Annual |

### ⚖️ Scaling:

| Column | Method | Result |
|---|---|---|
| Age | Min-Max Scaler | 0.0 to 1.0 |
| Tenure | Min-Max Scaler | 0.0 to 1.0 |
| Usage Frequency | Min-Max Scaler | 0.0 to 1.0 |
| Support Calls | Min-Max Scaler | 0.0 to 1.0 |
| Payment Delay | Min-Max Scaler | 0.0 to 1.0 |
| Last Interaction | Min-Max Scaler | 0.0 to 1.0 |
| Total Spend | Standard Scaler | mean=0, std=1 |

### 🎯 Feature Selection — Correlation Heatmap:

| Column | Correlation with Churn | Decision |
|---|---|---|
| Support Calls | High Positive | 🔴 Keep ✅ |
| Payment Delay | High Positive | 🔴 Keep ✅ |
| Total Spend | Moderate | 🟠 Keep ✅ |
| Tenure | Moderate Negative | 🟠 Keep ✅ |
| Contract Length | Moderate Negative | 🟠 Keep ✅ |
| Age | Weak | 🟡 Keep ✅ |
| CustomerID | No relation | ⚪ Remove ❌ |

**Final Features Used:**
```python
X = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
     'Payment Delay', 'Total Spend', 'Last Interaction',
     'Subscription_Basic', 'Subscription_Standard', 'Subscription_Premium',
     'Contract_Monthly', 'Contract_Quarterly', 'Contract_Annual']

y = ['Churn']
```

---

## 🤖 Stage 6 — Model Selection & Building

- **Algorithm Selected:** Logistic Regression
- **Why Logistic Regression?** Target variable is binary (0 or 1)
- **Train/Test Split:** 80% training, 20% testing

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 📈 Stage 7 — Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## ▶️ How to Run on Google Colab

**Step 1** — Open Google Colab: colab.research.google.com

**Step 2** — Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3** — Load the dataset:
```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Machine_learning/customer_churn_dataset-testing-master.csv')
print(df.shape)
```

**Step 4** — Run all cells from top to bottom ✅

---

## 👩‍💻 Author

**Iqra**
Beginner ML Student | Pakistan 🇵🇰
Learning Machine Learning step by step 🚀

---

## 📚 References

- Pandas Documentation: pandas.pydata.org
- Scikit-learn Documentation: scikit-learn.org
- Seaborn Documentation: seaborn.pydata.org
- GeeksforGeeks — ML Lifecycle (2026)
- Kaggle — Customer Churn Datasets

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
