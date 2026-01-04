# Dimesionality Reduction technique

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = {
    'Maths': [80, 60, 90, 70, 85],
    'Science': [75, 65, 95, 60, 80],
    'English': [70, 60, 85, 65, 75]
}

df = pd.DataFrame(data)
print(df)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print(scaled_data)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
print(pca_df)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
