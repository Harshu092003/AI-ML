# y = wx + b
# Linear Regression is a supervised machine learning algorithm 
# used to model the relationship between an independent variable (x) and a dependent variable (y) using a straight line.
# In linear regression, the best-fit line is the straight line that most accurately represents the relationship between 
#the independent variable (input) and the dependent variable (output). 
#It is the line that minimizes the difference between the actual data points and the predicted values from the model.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/advertising.csv')

# Prepare X and y
X = df[['Advertising']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# slope (w): change in Sales(Y) for a 1-unit increase in Advertising(X)
slope = model.coef_[0]

# intercept (b): predicted Sales(Y) when Advertising(X) spend is 0
intercept = model.intercept_

print(slope, intercept)
print(model.predict([[6]]))

# Plot
plt.figure(figsize=(7,5))
plt.scatter(df['Advertising'], df['Sales'], label="Data Points")

plt.plot(df['Advertising'], model.predict(X), label=f"Regression Line (y={slope:.2f}x+{intercept:.2f})")


plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Linear Regression Line - Understanding Slope & Intercept")
plt.legend()
plt.savefig('/home/harshvardhan/AI-ML/Supervised/matplot_visuals/linear_regression.png')

slope, intercept
