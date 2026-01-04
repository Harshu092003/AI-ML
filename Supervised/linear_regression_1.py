import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/advertising.csv')

# Prepare X and y
X = df[['Advertising']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_
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
