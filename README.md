# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries & Load Dataset
2.Divide the dataset into training and testing sets.
3.Select a suitable ML model, train it on the training data, and make predictions.
4.Assess model performance using metrics and interpret the results.
```
## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: ARTHI S
RegisterNumber:  212225220011
*/
```
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment (1) (2).csv')
data.head()

data = data.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data.head()
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print('Name: ARTHI S')
print('Reg. No: 212225220011')
print("\n== Cross-Validation ==")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R^2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R^2:{cv_scores.mean():.4f}")

y_pred =model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):>10.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()

```

## Output:

<img width="1753" height="343" alt="Screenshot 2026-03-28 205910" src="https://github.com/user-attachments/assets/5c462c8b-1c84-44b1-ac75-57b0e1aca901" />
<img width="1751" height="304" alt="Screenshot 2026-03-28 205926" src="https://github.com/user-attachments/assets/cec62095-e65e-4e63-8e18-25337137d84a" />
<img width="853" height="777" alt="Screenshot 2026-03-28 205943" src="https://github.com/user-attachments/assets/4e0d8107-d3a9-4d3e-b6cb-043c24534b1c" />

## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
