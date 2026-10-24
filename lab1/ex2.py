import matplotlib.pyplot as plt

from pyod.models.knn import KNN
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    auc
)

from ex1 import load_dataset


# Load the dataset from ex 1
X_train, X_test, y_train, y_test = load_dataset()

CONTAMINATION = 0.1

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
print("Training contamination:", y_train.mean())
print("Testing contamination:", y_test.mean())


# Train the KNN model and make predictions
model = KNN(contamination=CONTAMINATION)
model.fit(X_train)

# Predict normal samples (0) and outliers (1)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Continuous anomaly scores for ROC curves
scores_train = model.decision_function(X_train)
scores_test = model.decision_function(X_test)


# Confusion matrix and balanced accuracy
def evaluate_model(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\n--- {dataset_name} ---")
    print("Confusion matrix:")
    print(cm)
    print(f"TN (True Negatives): {tn}")
    print(f"TP (True Positives): {tp}")
    print(f"FN (False Negatives): {fn}")
    print(f"FP (False Positives): {fp}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    return balanced_acc


train_bacc = evaluate_model(
    y_train, y_pred_train, "Training Data"
)

test_bacc = evaluate_model(
    y_test, y_pred_test, "Testing Data"
)


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, scores_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN on Testing Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Change contamination and compare the metrics
contamination_values = [0.01, 0.05, 0.10, 0.20]

results = []

for contamination in contamination_values:
    model = KNN(contamination=contamination)
    model.fit(X_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_bacc = balanced_accuracy_score(y_train, pred_train)
    test_bacc = balanced_accuracy_score(y_test, pred_test)

    # Calculate ROC-AUC using continuous anomaly scores
    test_scores = model.decision_function(X_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
    test_auc = auc(fpr_test, tpr_test)

    results.append({
        "contamination": contamination,
        "train_bacc": train_bacc,
        "test_bacc": test_bacc,
        "test_auc": test_auc
    })


# Print comparison table
print("\n--- Effect of Contamination ---")
print(
    f"{'Contamination':>14} | "
    f"{'Train BAcc':>10} | "
    f"{'Test BAcc':>9} | "
    f"{'Test AUC':>8}"
)
print("-" * 55)

for result in results:
    print(
        f"{result['contamination']:14.2f} | "
        f"{result['train_bacc']:10.4f} | "
        f"{result['test_bacc']:9.4f} | "
        f"{result['test_auc']:8.4f}"
    )


# Plot balanced accuracy for each contamination rate
contaminations = [
    result["contamination"] for result in results
]

train_accuracies = [
    result["train_bacc"] for result in results
]

test_accuracies = [
    result["test_bacc"] for result in results
]

plt.figure(figsize=(8, 6))

plt.plot(
    contaminations,
    train_accuracies,
    "o-",
    label="Training Balanced Accuracy"
)

plt.plot(
    contaminations,
    test_accuracies,
    "s-",
    label="Testing Balanced Accuracy"
)

plt.xlabel("Contamination Rate")
plt.ylabel("Balanced Accuracy")
plt.title("Effect of Contamination on KNN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
