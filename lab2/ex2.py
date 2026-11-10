import numpy as np
import matplotlib.pyplot as plt

from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
from matplotlib.colors import ListedColormap

# Reproducibility
np.random.seed(42)

# Generate synthetic dataset
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    n_features=2,
    contamination=0.1,
    size='same',
    density='same',
    dist=0.25,
    random_state=42
)

# Different numbers of neighbors to experiment with
neighbors_values = [3, 5, 10, 20, 50, 100]

# Colors: blue = inliers, red = outliers
cmap = ListedColormap(['#3498db', '#e74c3c'])

# Store balanced accuracy results
results = []

for k in neighbors_values:

    # Train KNN model
    model = KNN(
        contamination=0.1,
        n_neighbors=k
    )

    model.fit(X_train)

    # Predict labels
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate balanced accuracy
    train_accuracy = balanced_accuracy_score(
        y_train, y_train_pred
    )

    test_accuracy = balanced_accuracy_score(
        y_test, y_test_pred
    )

    results.append((k, train_accuracy, test_accuracy))

    # Create four subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plots = [
        (X_train, y_train, "Ground truth - Training data"),
        (X_train, y_train_pred, "Predicted labels - Training data"),
        (X_test, y_test, "Ground truth - Test data"),
        (X_test, y_test_pred, "Predicted labels - Test data")
    ]

    for ax, (X, labels, title) in zip(axes.flat, plots):

        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            cmap=cmap,
            alpha=0.7,
            s=30,
            vmin=0,
            vmax=1
        )

        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        ax.legend(
            handles=[
                plt.Line2D(
                    [], [], marker='o', linestyle='',
                    color='#3498db', label='Inliers'
                ),
                plt.Line2D(
                    [], [], marker='o', linestyle='',
                    color='#e74c3c', label='Outliers'
                )
            ],
            loc='best'
        )

    fig.suptitle(
        f"KNN Anomaly Detection (n_neighbors={k})\n"
        f"Train Balanced Accuracy={train_accuracy:.3f} | "
        f"Test Balanced Accuracy={test_accuracy:.3f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# Print balanced accuracy results
print("\nBalanced Accuracy Results")
print("-" * 55)
print(f"{'Neighbors':<15}{'Train Accuracy':<20}{'Test Accuracy':<20}")

for k, train_acc, test_acc in results:
    print(f"{k:<15}{train_acc:<20.4f}{test_acc:<20.4f}")

