# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries:
    numpy for numerical operations.
    pandas for data loading and manipulation.
    StandardScaler from sklearn to normalize (standardize) the data.
2. Define a function for Linear Regression using Gradient Descent:

      Initialize weights and bias with zeros (or small random values).
      Loop for a fixed number of iterations:
      Compute predictions: 
      𝑦 = wx+b.
      Calculate the cost function (Mean Squared Error).

3. Train the model using training data.
4. Predict the output.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pranav Bhargav M
RegisterNumber:  212224040239
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

```

```
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![linear regression using gradient descent](sam.png)
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/e545bc3c-e099-4fcc-8e24-73f63e57dc75" />
<img width="591" height="329" alt="image" src="https://github.com/user-attachments/assets/5eba72bd-db0d-4243-a35a-5cba64f46655" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
