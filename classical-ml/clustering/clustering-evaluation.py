"""Evaluate K-Means using inertia and silhouette scores."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Generate reproducible data with four underlying groups.
X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42,
)

X_scaled = StandardScaler().fit_transform(X)

k_values = range(2, 9)
inertias = []
silhouette_scores = []

print(f"{'K':<5}{'Inertia':<15}{'Silhouette':<15}")

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    score = silhouette_score(X_scaled, labels)

    inertias.append(inertia)
    silhouette_scores.append(score)

    print(f"{k:<5}{inertia:<15.3f}{score:<15.3f}")

best_index = silhouette_scores.index(max(silhouette_scores))
print(
    "\nHighest silhouette score:",
    f"K={list(k_values)[best_index]},",
    f"score={silhouette_scores[best_index]:.3f}",
)

# Elbow method plot.
plt.figure()
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(
    "classical-ml/clustering/elbow-plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Silhouette score plot.
plt.figure()
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.title("Silhouette Score vs K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(
    "classical-ml/clustering/silhouette-plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("Saved: classical-ml/clustering/elbow-plot.png")
print("Saved: classical-ml/clustering/silhouette-plot.png")
