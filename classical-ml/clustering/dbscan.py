"""DBSCAN clustering with noise detection and visualization."""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate non-spherical sample data.
X, _ = make_moons(
    n_samples=500,
    noise=0.07,
    random_state=42,
)

# Scale features.
X_scaled = StandardScaler().fit_transform(X)

# Fit DBSCAN.
model = DBSCAN(eps=0.25, min_samples=5)
labels = model.fit_predict(X_scaled)

# Separate clusters from noise.
cluster_labels = set(labels) - {-1}
noise_count = int((labels == -1).sum())

print("Number of clusters:", len(cluster_labels))
print("Noise points:", noise_count)

for label in sorted(cluster_labels):
    print(f"Cluster {label} size:", int((labels == label).sum()))

# Silhouette score is defined only when at least two clusters exist.
if len(cluster_labels) >= 2:
    score = silhouette_score(X_scaled, labels)
    print("Silhouette score (including noise):", round(score, 3))
else:
    print("Silhouette score unavailable: fewer than two clusters.")

# Visualize clusters and noise.
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=25,
    alpha=0.8,
)
plt.title("DBSCAN Clustering (noise label = -1)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.tight_layout()

plot_path = "classical-ml/clustering/dbscan-plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print("Plot saved to", plot_path)
