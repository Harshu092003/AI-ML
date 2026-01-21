# Random Forest:
# Random Forest is an ensemble machine learning algorithm that builds multiple
# decision trees using random subsets of data and features, and combines their
# predictions (by voting or averaging) to improve accuracy, reduce overfitting,
# and increase robustness compared to a single decision tree.


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

titanic_data = pd.read_csv('/home/harshvardhan/AI-ML/Supervised/datasets/titanic.csv')
titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = titanic_data['Survived']

X.loc[:,'Sex'] = X['Sex'].map({'female' : 0, 'male' :1 })
X.loc[:,'Age'].fillna(X['Age'].median(),inplace = True)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state= 42)

rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)

rf_classifier.fit(X_train , y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)

classification_rep = classification_report(y_test , y_pred)
print(f'Accuracy Score : {accuracy}')
print(f'\nclassification Report : {classification_rep}')

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nsample passenger : {sample_dict}")
print(f"Predicted Survival : { 'Survived' if prediction[0] == 1 else 'Did not survive'}")


plt.figure(figsize=(8, 5))
plt.bar(X.columns, rf_classifier.feature_importances_ ,color = 'pink')  # No colors specified
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png")

