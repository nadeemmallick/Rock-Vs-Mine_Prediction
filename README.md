# Rock-Vs-Mine_Prediction

This is a beginner-friendly **machine learning classification project** that predicts whether a given sonar signal is coming from a **rock** or a **mine**. The model is built using **Logistic Regression** and trained on the classic **Sonar Dataset**.

## Dataset Used

- The dataset contains 208 rows and 61 columns.
- Each row represents a sonar signal reflected off a surface.
- **Columns 0 to 59:** Numerical values (signal strength from 60 sensors)
- **Column 60:** Label (`R` = Rock, `M` = Mine)

**Download Dataset From:** [https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view?usp=drivesdk]


## ML Workflow in This Project

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Load Dataset
```python
data = pd.read_csv("sonar.all-data.csv", header=None)
```

### 3. Preprocess Data
```python
X = data.iloc[:, :-1]   # features
y = data.iloc[:, -1]    # labels
```

### 4. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
```

### 5. Model Training
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 6. Evaluation
```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

### 7. Prediction on New Data
```python
sample = X.iloc[0].values.reshape(1, -1)
model.predict(sample)
```

---

##  Accuracy Achieved
- The Logistic Regression model achieves approximately **85-90% accuracy** depending on the train/test split.

---

##  Libraries Used
- `pandas`, `numpy` → data handling
- `sklearn` → model, metrics, splitting

---

##  Why Logistic Regression?
- Simple and fast for binary classification
- Works well for linearly separable data
- Good baseline model

---

##  How to Run This Project

1. Install required libraries:
```bash
pip install pandas numpy scikit-learn
```

2. Run the notebook `Rock_Vs_Mine.ipynb` step by step

---