# Naive Bayes chooses the class with the highest probability based on the input features.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("/home/harshvardhan/AI-ML/Supervised/datasets/pizza_buy_data.csv")

# Convert text labels to numbers
le_weather = LabelEncoder()
le_hungry = LabelEncoder()
le_buy = LabelEncoder()

df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Hungry'] = le_hungry.fit_transform(df['Hungry'])
df['Buy'] = le_buy.fit_transform(df['Buy'])

# Split features & target
X = df[['Weather', 'Hungry']]
y = df['Buy']

# Train model
model = GaussianNB()
model.fit(X, y)

# Test input
test = [[ le_weather.transform(['Sunny'])[0],
          le_hungry.transform(['Yes'])[0] ]]

prediction = model.predict(test)[0]
proba = model.predict_proba(test)

# Decode prediction back to Yes/No
prediction_label = le_buy.inverse_transform([prediction])[0]

print("Test Input: Sunny, Yes")
print("Prediction person will buy or not :", prediction_label)
print("Probabilities:", proba)
