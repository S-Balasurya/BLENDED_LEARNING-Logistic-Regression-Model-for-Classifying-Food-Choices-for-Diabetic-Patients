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

Name:Balasurya S


Reg. No:212225100003

Dataset Overview:

   Calories  Total Fat  Saturated Fat  Monounsaturated Fat  \
0     149.0          0            0.0                  0.0   
1     123.0          0            0.0                  0.0   
2     150.0          0            0.0                  0.0   
3     110.0          0            0.0                  0.0   
4     143.0          0            0.0                  0.0   

   Polyunsaturated Fat  Trans Fat  Cholesterol  Sodium  Total Carbohydrate  \
0                  0.0        0.0            0     9.0                 9.8   
1                  0.0        0.0            0     5.0                 6.6   
2                  0.0        0.0            0     4.0                11.4   
3                  0.0        0.0            0     6.0                 7.0   
4                  0.0        0.0            0     7.0                13.1   

   Dietary Fiber  Sugars  Sugar Alcohol  Protein  Vitamin A  Vitamin C  \
0            0.0     0.0              0      1.3          0          0   
1            0.0     0.0              0      0.8          0          0   
2            0.0     0.0              0      1.3          0          0   
3            0.0     0.0              0      0.8          0          0   
4            0.0     0.0              0      1.0          0          0   

   Calcium  Iron            class  
0        0     0  'In Moderation'  
1        0     0  'In Moderation'  
2        0     0  'In Moderation'  
3        0     0  'In Moderation'  
4        0     0  'In Moderation'  

Dataset Info:


<class 'pandas.core.frame.DataFrame'>

RangeIndex: 13260 entries, 0 to 13259

Data columns (total 18 columns):

 #   Column               Non-Null Count  Dtype  

---  ------               --------------  -----  

 0   Calories             13260 non-null  float64
 
 1   Total Fat            13260 non-null  int64  
 
 2   Saturated Fat        13260 non-null  float64
 
 3   Monounsaturated Fat  13260 non-null  float64
 
 4   Polyunsaturated Fat  13260 non-null  float64
 
 5   Trans Fat            13260 non-null  float64
 
 6   Cholesterol          13260 non-null  int64  
 
 7   Sodium               13260 non-null  float64
 
 8   Total Carbohydrate   13260 non-null  float64
 
 9   Dietary Fiber        13260 non-null  float64
 
 10  Sugars               13260 non-null  float64
 
 11  Sugar Alcohol        13260 non-null  int64  
 
 12  Protein              13260 non-null  float64
 
 13  Vitamin A            13260 non-null  int64  
 
 14  Vitamin C            13260 non-null  int64  
 
 15  Calcium              13260 non-null  int64  
 
 16  Iron                 13260 non-null  int64  
 
 17  class                13260 non-null  object 

types: float64(10), int64(7), object(1)

memory usage: 1.8+ MB

None


Model Evaluation:

Accuracy: 0.7748868778280543

Classification Report:
            
              precision    recall  f1-score   support

           
           0       0.73      0.87      0.80      1330
           
           1       0.84      0.73      0.78      1124
           
           2       0.92      0.35      0.51       198



    
    accuracy                           0.77      2652
   macro avg       0.83      0.65      0.70      2652
weighted avg       0.79      0.77      0.77      2652


[[1162  162    6]
 [ 301  823    0]
 [ 128    0   70]]

## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
