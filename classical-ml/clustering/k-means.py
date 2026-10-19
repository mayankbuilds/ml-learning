"""K-Means clustering with visualization."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate sample data.
X, _ = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=1.2,
    random_state=42,
)

# Scale features.
X_scaled = StandardScaler().fit_transform(X)

# Fit K-Means.
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)
labels = model.fit_predict(X_scaled)

print("Inertia:", round(model.inertia_, 3))
print("Silhouette score:", round(silhouette_score(X_scaled, labels), 3))
print("Cluster sizes:", {
    int(label): int((labels == label).sum())
    for label in set(labels)
})

# Visualize the clusters.
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=25,
    alpha=0.8,
)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    c="red",
    label="Centroids",
)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig(
    "classical-ml/clustering/k-means-plot.png",
    dpi=150,
    bbox_inches="tight",
)
print("Plot saved to classical-ml/clustering/k-means-plot.png")