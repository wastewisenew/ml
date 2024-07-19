from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = load_iris()

# Prepare the data
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name='Targets')

# Plot configuration
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# Real plot
plt.subplot(1, 3, 1)
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[y], s=40)
plt.title('Real')

# KMeans plot
plt.subplot(1, 3, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
pred_y_kmeans = np.choose(kmeans.labels_, [0, 1, 2]).astype(np.int64)
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[pred_y_kmeans], s=40)
plt.title('KMeans')

# GMM plot
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)
pred_y_gmm = gmm.predict(X_scaled)

plt.subplot(1, 3, 3)
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[pred_y_gmm], s=40)
plt.title('GMM Classification')

# Show plots
plt.show()
