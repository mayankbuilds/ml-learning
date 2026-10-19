# Clustering Evaluation

## 1. Why Evaluation Matters

Clustering is unsupervised learning, so ground-truth labels are often unavailable. Evaluation helps us understand cluster compactness, separation, and whether a chosen configuration is useful.

No single metric proves that clusters are meaningful.

## 2. Inertia

Inertia is the sum of squared distances between each point and its assigned centroid.

\[
\text{Inertia} = \sum_{i=1}^{K}\sum_{x\in C_i}\|x-\mu_i\|^2
\]

- Lower inertia means points are closer to their centroids.
- It is commonly used with K-Means.
- Inertia generally decreases as the number of clusters increases, so it should not be minimized blindly.
- It is not a standard evaluation metric for DBSCAN.

## 3. Silhouette Score

The silhouette score measures how well a point fits its own cluster compared with the nearest alternative cluster.

\[
s = \frac{b-a}{\max(a,b)}
\]

Where:

- \(a\): Average distance from the point to other points in its own cluster.
- \(b\): Lowest average distance to points in another cluster.

Interpretation:

- Near `+1`: Point is well matched to its own cluster.
- Near `0`: Point is near a cluster boundary.
- Below `0`: Point may be closer to another cluster.

The score ranges from `-1` to `1`. It requires at least two clusters and fewer clusters than samples.

## 4. Elbow Method

1. Fit K-Means with several values of \(K\).
2. Record inertia for each value.
3. Plot \(K\) against inertia.
4. Look for a bend where further reductions in inertia become less substantial.

The elbow is a heuristic, not a guaranteed optimal cluster count.

## 5. Evaluating DBSCAN

For DBSCAN, inspect:

- Number of discovered clusters.
- Number and proportion of noise points (`label == -1`).
- Cluster sizes.
- Silhouette score when its requirements are satisfied.
- Whether the resulting groups make sense for the problem.

Noise points need careful treatment when computing silhouette scores. If noise is treated as one ordinary cluster, the resulting score can be misleading.

## 6. Limitations

- Internal metrics do not guarantee real-world usefulness.
- Scores depend on the distance metric and feature scaling.
- Silhouette score may favor compact, well-separated groups and may not fully capture irregular cluster shapes.
- Domain knowledge and visual inspection can provide additional evidence.

## 7. Key Takeaway

Use multiple signals to evaluate clustering. Combine quantitative metrics, visual inspection, parameter sensitivity, and domain understanding instead of choosing a model using one score alone.
