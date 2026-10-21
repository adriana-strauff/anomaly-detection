import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

# Generate the dataset
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42
)

# Separate normal samples and outliers in the training set
normal = y_train == 0
outliers = y_train == 1

# Plot training samples
plt.figure(figsize=(8, 6))
plt.scatter(
    X_train[normal, 0],
    X_train[normal, 1],
    c="blue",
    label="Normal samples",
    alpha=0.7
)
plt.scatter(
    X_train[outliers, 0],
    X_train[outliers, 1],
    c="red",
    label="Outliers",
    alpha=0.8
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Dataset: Normal Samples and Outliers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()