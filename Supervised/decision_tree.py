# A Decision Tree asks questions step-by-step on your data to make a final prediction.

# Example: Predict whether a person will buy a product

# The tree may ask:

# Age < 30?

# Yes → go right

# No → go left

# Next question:

# Income > 50k?

# Yes → Predict: Buy

# No → Predict: Not Buy

# Each question is a split in the tree.

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/buy_data.csv')
X = df[['Age', 'Income']]   
y = df['Buy']

model = DecisionTreeClassifier()
model.fit(X, y)

test = [[28, 60000]]
prediction = model.predict(test)
print("Test data (Age, Income):", test[0])
print("Prediction (1 = Buy, 0 = Not Buy):", prediction[0])
print(f" Probability : {model.predict_proba(test)}")
