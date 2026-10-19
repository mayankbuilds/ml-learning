# DBSCAN Clustering

## 1. Overview

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.

It groups points that belong to sufficiently dense regions and identifies isolated points as noise. Unlike K-Means, it does not require the number of clusters in advance.

## 2. Core Concepts

- **Epsilon (`eps`):** Maximum neighborhood distance used to find nearby points.
- **Min samples (`min_samples`):** Minimum number of nearby points required for a dense region.
- **Core point:** A point with at least `min_samples` points in its neighborhood, including itself.
- **Border point:** A point near a core point that does not itself meet the core-point requirement.
- **Noise point:** A point that does not belong to any discovered cluster.

## 3. How It Works

1. Select an unvisited point.
2. Find its neighbors within `eps`.
3. If it has enough neighbors, start or expand a cluster.
4. Continue expanding through density-connected core points.
5. Mark points that cannot be assigned to a cluster as noise.

## 4. Implementation

See [`dbscan.py`](dbscan.py) for an implementation using scikit-learn, feature scaling, cluster counts, noise counts, and visualization.

## 5. Important Parameters

- `eps`: Neighborhood radius.
- `min_samples`: Minimum neighborhood density.
- `metric`: Distance metric, defaulting to Euclidean distance.

The appropriate `eps` depends heavily on the scale of the features and the density of the dataset.

## 6. Limitations

- Sensitive to `eps` and `min_samples`.
- Can struggle when clusters have very different densities.
- High-dimensional data can make distance-based neighborhoods less informative.
- Noise is assigned label `-1`.

## 7. K-Means vs DBSCAN

| Property | K-Means | DBSCAN |
|---|---|---|
| Number of clusters | Must be specified | Discovered from density |
| Noise detection | No built-in noise label | Yes |
| Cluster shape | Works best for compact, roughly spherical clusters | Can identify irregularly shaped clusters |
| Main parameters | `n_clusters` | `eps`, `min_samples` |

## 8. Key Takeaway

DBSCAN is useful when clusters may have irregular shapes or when detecting outliers matters. Always inspect the number of clusters and noise points rather than assuming every dataset will produce useful groups.
