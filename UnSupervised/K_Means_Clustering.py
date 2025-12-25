import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------
# Step 1: Dataset
# -----------------------------
# Student scores: [Math, Science]
X = np.array([
    [40, 45], # Student 1
    [42, 43], # Student 2
    [38, 40], # Student 3
    [90, 92], # Student 4
    [88, 85], # Student 5
    [92, 90]  # Student 6
])

# -----------------------------
# Step 2: Apply K-Means
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit model and predict clusters
labels = kmeans.fit_predict(X)

# -----------------------------
# Step 3: Print Results
# -----------------------------
print("Data Points:")
print(X)

print("\nCluster Labels:")
print(labels)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# -----------------------------
# Step 4: Visualization
# -----------------------------
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=200
)

plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.title("K-Means Clustering Example")
plt.savefig("visuals/kmeans_clustering.png")