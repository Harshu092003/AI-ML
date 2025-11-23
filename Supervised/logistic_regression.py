# Logistic Regression is a classification algorithm, not a regression algorithm (even though the name says “regression”).

# It is used when the output is YES/NO, 0/1, Spam/Not Spam, Survived/Not Survived, etc.

# What is the Sigmoid Function?

# The sigmoid is a mathematical function that takes any number (–∞ to +∞) and squeezes it into a value between:

# 0 and 1

# This makes it perfect for probabilities.

from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/study.csv')
X = df[['hours']]
y = df['result']

model = LogisticRegression()
model.fit(X, y)

hours = 4.5
prediction = model.predict([[hours]])
probability = model.predict_proba([[hours]])

print("Hours studied:", hours)
print("Prediction (1 = Pass, 0 = Fail):", prediction[0])
print("Probability of passing:", probability[0][1])
print("Probability of failing:", probability[0][0])
