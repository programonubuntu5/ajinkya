
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Elbow Method
inertia = []

for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 7), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# K-Means with 3 clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

print("Cluster Labels:", labels)
print("Silhouette Score:", silhouette_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("K-Means Clustering")
plt.show()
