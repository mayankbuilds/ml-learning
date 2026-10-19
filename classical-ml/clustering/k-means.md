# K-Means Clustering

## 1. Overview

K-Means is an unsupervised learning algorithm that partitions data into \(K\) clusters. Each data point is assigned to the cluster with the nearest centroid.

Unlike supervised learning, K-Means does not require target labels.

## 2. How It Works

1. Choose the number of clusters, \(K\).
2. Initialize \(K\) centroids.
3. Assign each point to its nearest centroid.
4. Recalculate each centroid as the mean of its assigned points.
5. Repeat assignment and centroid updates until convergence or the iteration limit.

## 3. Objective Function

K-Means minimizes the within-cluster sum of squared distances:

\[
J = \sum_{i=1}^{K}\sum_{x\in C_i}\|x-\mu_i\|^2
\]

Here, \(C_i\) is cluster \(i\), and \(\mu_i\) is its centroid.

## 4. Implementation

See [`k-means.py`](k-means.py) for a complete implementation with generated data and visualization.

## 5. Important Parameters

- `n_clusters`: Number of clusters.
- `random_state`: Reproducibility.
- `n_init`: Number of centroid initializations to try.
- `max_iter`: Maximum iterations per initialization.

## 6. Limitations

- You must choose \(K\) beforehand.
- Sensitive to feature scales and outliers.
- Can struggle with non-spherical clusters or clusters with very different densities.
- Different initializations can produce different results.

## 7. Key Takeaway

K-Means is a useful starting point for clustering numerical data. Scale features when their magnitudes differ significantly, and do not assume the resulting clusters represent real-world categories without further analysis.
