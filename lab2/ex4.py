import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization


# Load the cardio dataset
data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()

# Calculate the contamination rate
contamination = np.mean(y == 1)

# Split the data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalize the data
X_train, X_test = standardizer(X_train, X_test)

# Define the number of neighbors
neighbors_values = np.linspace(30, 120, 10, dtype=int)

# Choose the model
model_type = "KNN"  # Change to "LOF" to use LOF

if model_type == "KNN":
    Model = KNN
else:
    Model = LOF

# Store the training and test scores
train_scores = []
test_scores = []

# Fit the models and calculate balanced accuracy
print(f"\n{model_type} results:")
print("Neighbors\tTrain BA\tTest BA")

for k in neighbors_values:
    model = Model(contamination=contamination, n_neighbors=k)
    model.fit(X_train)

    y_train_pred = model.labels_
    y_test_pred = model.predict(X_test)

    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)

    print(f"{k}\t\t{train_ba:.4f}\t\t{test_ba:.4f}")

    train_scores.append(model.decision_scores_)
    test_scores.append(model.decision_function(X_test))

# Convert the scores into arrays
train_scores = np.column_stack(train_scores)
test_scores = np.column_stack(test_scores)

# Normalize the scores
train_scores, test_scores = standardizer(train_scores, test_scores)

# Combine the scores using average and maximization
average_train = average(train_scores)
average_test = average(test_scores)

max_train = maximization(train_scores)
max_test = maximization(test_scores)

# Calculate the threshold and balanced accuracy
def evaluate_results(train_scores, test_scores, strategy):
    threshold = np.quantile(train_scores, 1 - contamination)

    y_train_pred = (train_scores >= threshold).astype(int)
    y_test_pred = (test_scores >= threshold).astype(int)

    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)

    print(f"\n{strategy} strategy:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train BA: {train_ba:.4f}")
    print(f"Test BA: {test_ba:.4f}")


# Evaluate both strategies
evaluate_results(average_train, average_test, "Average")
evaluate_results(max_train, max_test, "Maximization")
