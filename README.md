# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Logistic Regression is used to classify data into different categories based on input features.

2.Label Encoding converts categorical target values into numerical form for model training.

3.Min-Max Scaling normalizes feature values to improve model performance.

4.Train-Test Split divides the dataset into training and testing sets for evaluation.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Balasurya S
RegisterNumber: 212225100003 
*/

EXP-6

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("food_items.csv")

# Inspect the dataset
print("Name: Balasurya S")
print("Reg. No: 212225100003")
print("Dataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()

# Scaling the raw input features
X = scaler.fit_transform(X_raw)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())
# Note that ravel() function flattens the vector.

# First, let's split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 123)

# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'

# Our classification problem is multinomial
multi_class = 'multinomial'

# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'

# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

# Evaluate the model
print("Name:Balasurya S")
print("Reg. No:212225100003")
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("Name: Balasurya S")
print("Reg. No: 212225100003")
```

## Output:

<img width="796" height="735" alt="Screenshot 2026-03-28 144158" src="https://github.com/user-attachments/assets/520a1b23-c362-4fa2-9739-56941c3c5555" />

<img width="561" height="434" alt="Screenshot 2026-03-28 144207" src="https://github.com/user-attachments/assets/6957f375-898c-46e4-9b1d-65624f06d15a" />

<img width="203" height="92" alt="Screenshot 2026-03-28 144211" src="https://github.com/user-attachments/assets/eecd8b29-3f27-4318-b528-db05bdb0b6d1" />

<img width="221" height="73" alt="Screenshot 2026-03-28 144217" src="https://github.com/user-attachments/assets/b292e78d-e51f-4818-9ec4-159f00f4746f" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
