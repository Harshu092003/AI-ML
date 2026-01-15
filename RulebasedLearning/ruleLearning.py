"""
Rule-Based System using:
- Knowledge Graph (symbolic facts)
- Decision Tree (explicit rules)
- scikit-learn

Domain: Student Marks → Pass / Fail
"""

from sklearn.tree import DecisionTreeClassifier, export_text    
import warnings
warnings.filterwarnings("ignore")

knowledge_graph = {
    "harshvardhan" : {"marks" : "high"} ,
    "pradhnesh" : {"marks" : "medium"} ,
    "Varad": {'marks' : "high"} ,
    "mihir" : {"marks" : "low"} ,
    "sahil" : {"marks" : "high"}
}

def extract_features(student , kg):
    feature = [0,0,0]

    if kg[student]["marks"] == 'high' :
        feature[0] = 1
    if kg[student]["marks"]  == 'medium' :
        feature[1] = 1
    if kg[student]['marks'] == 'low' :
        feature[2] = 1

    return feature

X = []
y = []

for student in knowledge_graph:
    X.append(extract_features(student,knowledge_graph))

    if knowledge_graph[student]['marks'] in ["high" , "medium"] :
        y.append(1)
    else : 
        y.append(0)

model = DecisionTreeClassifier()
model.fit(X,y)

def infer(student):
    features = extract_features(student,knowledge_graph)
    prediction = model.predict([features])[0]

    if prediction == 1:
        return f"student passed"
    else :
        return f"student failed"    
    
print(infer('sahil'))


# -----------------------------
# 7. Show Learned Rules
# -----------------------------
print("\nLearned Rules (Decision Tree):")
rules = export_text(
    model,
    feature_names=["high_marks", "medium_marks", "low_marks"]
)
print(rules)
