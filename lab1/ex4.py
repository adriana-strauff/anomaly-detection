
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


# Dataset parameters
N_SAMPLES = 1000
CONTAMINATION = 0.10
N_FEATURES = 2

rng = np.random.default_rng(42)

# Choose the mean vector and covariance matrix
mu = np.array([2.0, -1.0])

Sigma = np.array([
    [2.0, 0.8],
    [0.8, 1.5]
])

# Generate normal samples from N(0, I)
n_outliers = int(N_SAMPLES * CONTAMINATION)
n_normal = N_SAMPLES - n_outliers

X_standard = rng.multivariate_normal(
    mean=np.zeros(N_FEATURES),
    cov=np.eye(N_FEATURES),
    size=n_normal
)

# Apply Cholesky decomposition: Sigma = L @ L.T
L = np.linalg.cholesky(Sigma)

# Transform standard normal samples: y = Lx + mu
X_normal = X_standard @ L.T + mu

# Generate anomalies from a different distribution
X_outliers = rng.uniform(
    low=-8,
    high=8,
    size=(n_outliers, N_FEATURES)
)

# Combine normal samples and anomalies
X = np.vstack([X_normal, X_outliers])
y_true = np.concatenate([
    np.zeros(n_normal, dtype=int),
    np.ones(n_outliers, dtype=int)
])

# Shuffle the dataset
indices = rng.permutation(N_SAMPLES)
X = X[indices]
y_true = y_true[indices]

# Compute a Z-score for each feature
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

z_scores = np.abs((X - mean) / std)

# Use the maximum absolute Z-score per sample
sample_scores = np.max(z_scores, axis=1)

# Threshold for the specified contamination rate
threshold = np.quantile(
    sample_scores,
    1 - CONTAMINATION
)

# Predict anomalies
y_pred = (sample_scores >= threshold).astype(int)

# Evaluate predictions
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

balanced_acc = balanced_accuracy_score(y_true, y_pred)

print("Number of samples:", N_SAMPLES)
print("Number of features:", N_FEATURES)
print("Actual contamination:", y_true.mean())
print("Mean vector:", mu)
print("Covariance matrix:\n", Sigma)
print("Cholesky factor L:\n", L)
print("Z-score threshold:", threshold)

print("\nConfusion matrix:")
print(cm)
print("TN:", tn)
print("TP:", tp)
print("FN:", fn)
print("FP:", fp)
print("Balanced accuracy:", round(balanced_acc, 4))
