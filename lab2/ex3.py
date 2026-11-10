import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# Generate two clusters with different densities
X, cluster_labels = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    n_features=2,
    random_state=42
)

# Test different numbers of neighbors
neighbors_values = [5, 10, 20, 30]

for k in neighbors_values:

    # Fit KNN
    knn = KNN(contamination=0.07, n_neighbors=k)
    knn.fit(X)
    knn_labels = knn.predict(X)

    # Fit LOF
    lof = LOF(contamination=0.07, n_neighbors=k)
    lof.fit(X)
    lof_labels = lof.predict(X)

    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [
        (knn_labels, "KNN"),
        (lof_labels, "LOF")
    ]

    for ax, (predicted_labels, model_name) in zip(axes, models):

        # Inliers: label 0; outliers: label 1
        inliers = predicted_labels == 0
        outliers = predicted_labels == 1

        ax.scatter(
            X[inliers, 0],
            X[inliers, 1],
            color="steelblue",
            label="Inliers",
            alpha=0.7,
            s=30
        )

        ax.scatter(
            X[outliers, 0],
            X[outliers, 1],
            color="red",
            label="Outliers",
            alpha=0.8,
            s=40
        )

        ax.set_title(
            f"{model_name} (n_neighbors={k})\n"
            f"Detected outliers: {np.sum(outliers)}"
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Anomaly Detection with KNN and LOF (k={k})",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

