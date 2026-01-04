# To predict something for a new point, KNN looks at the K closest data points (neighbors) and takes a majority vote (classification) or average (regression).
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/buy_data.csv')
X = df[['Age', 'Income']]
y = df['Buy']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

test = [[20, 30000]]
prediction = model.predict(test)
print("Test data (Age, Income):", test[0])
print("Prediction (1 = Buy, 0 = Not Buy):", prediction[0])
print(f" Probability : {model.predict_proba(test)}")
