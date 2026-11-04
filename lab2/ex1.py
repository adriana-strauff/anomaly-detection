
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Parameters
n = 100
a = 2
b = 3
mu = 0

noise_variances = [0.1, 1, 5]

# Generate four types of data
def generate_data(sigma2, n=100):

    # Regular observations
    x_regular = np.random.normal(0, 1, n)
    y_regular = a*x_regular + b + np.random.normal(
        mu, np.sqrt(sigma2), n
    )

    # High variance on x
    x_high = np.random.normal(0, 5, n)
    y_high = a*x_high + b + np.random.normal(
        mu, np.sqrt(sigma2), n
    )

    # High variance on y
    x_vertical = np.random.normal(0, 1, n)
    y_vertical = a*x_vertical + b + np.random.normal(
        mu, 5, n
    )

    # High variance on both x and y
    x_both = np.random.normal(0, 5, n)
    y_both = a*x_both + b + np.random.normal(
        mu, 5, n
    )

    return [
        (x_regular, y_regular, "Regular"),
        (x_high, y_high, "High variance X"),
        (x_vertical, y_vertical, "High variance Y"),
        (x_both, y_both, "High variance X and Y")
    ]


# Calculate leverage scores
def calculate_leverage(x):

    X = np.column_stack((np.ones(len(x)), x))

    H = X @ np.linalg.pinv(X.T @ X) @ X.T

    return np.diag(H)


# Plot results
fig, axes = plt.subplots(
    len(noise_variances), 4,
    figsize=(18, 12)
)

for row, sigma2 in enumerate(noise_variances):

    datasets = generate_data(sigma2)

    for col, (x, y, title) in enumerate(datasets):

        leverage = calculate_leverage(x)

        # Select the 5 observations with highest leverage
        highest = np.argsort(leverage)[-5:]

        ax = axes[row, col]

        ax.scatter(x, y, alpha=0.5, label="Observations")

        ax.scatter(
            x[highest],
            y[highest],
            color="red",
            s=70,
            label="Highest leverage"
        )

        ax.set_title(
            f"{title}\nNoise variance = {sigma2}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

