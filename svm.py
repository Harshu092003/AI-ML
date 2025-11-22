from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


titanic = pd.read_csv("/home/harshvardhan/AI-ML/titanic.csv")
X = titanic[['Sex','Age']]
y = titanic['Survived']

X.loc[:,'Sex'] = X['Sex'].map({'male':1, 'female' : 0})
X.loc[:,'Age'].fillna(X['Age'].median(), inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel="linear", C=1)
svm.fit(X_train, y_train)

DecisionBoundaryDisplay.from_estimator(
    svm,
    X_test,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1",
    xlabel="Age",
    ylabel="Sex",
)

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=y_test, 
            s=20, edgecolors="k" )

plt.savefig("svm.png")


