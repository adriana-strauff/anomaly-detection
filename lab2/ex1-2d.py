import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

n = 100
a, b, c = 2, 3, 5
mu = 0

noise_variances = [0.1, 1, 5]

def generate_data_2d(sigma2):

    # Regular
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = a*x1 + b*x2 + c + np.random.normal(mu, np.sqrt(sigma2), n)
    regular = (x1, x2, y)

    # High variance on X1
    x1 = np.random.normal(0, 5, n)
    x2 = np.random.normal(0, 1, n)
    y = a*x1 + b*x2 + c + np.random.normal(mu, np.sqrt(sigma2), n)
    high_x1 = (x1, x2, y)

    # High variance on Y
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = a*x1 + b*x2 + c + np.random.normal(mu, 5 * np.sqrt(sigma2), n)
    high_y = (x1, x2, y)

    # High variance on both X variables
    x1 = np.random.normal(0, 5, n)
    x2 = np.random.normal(0, 5, n)
    y = a*x1 + b*x2 + c + np.random.normal(mu, 5 * np.sqrt(sigma2), n)
    high_both = (x1, x2, y)

    return [
        (regular, "Regular"),
        (high_x1, "High variance X1"),
        (high_y, "High variance Y"),
        (high_both, "High variance X1 and X2")
    ]


def calculate_leverage_2d(x1, x2):

    X = np.column_stack([
        np.ones(len(x1)),
        x1,
        x2
    ])

    H = X @ np.linalg.pinv(X.T @ X) @ X.T

    return np.diag(H)


fig = plt.figure(figsize=(18, 12))

for row, sigma2 in enumerate(noise_variances):

    datasets = generate_data_2d(sigma2)

    for col, (data, title) in enumerate(datasets):

        x1, x2, y = data

        leverage = calculate_leverage_2d(x1, x2)

        highest = np.argsort(leverage)[-5:]

        ax = fig.add_subplot(
            len(noise_variances),
            4,
            row*4 + col + 1,
            projection="3d"
        )

        ax.scatter(x1, x2, y, alpha=0.4)

        ax.scatter(
            x1[highest],
            x2[highest],
            y[highest],
            color="red",
            s=70,
            label="Highest leverage"
        )

        ax.set_title(
            f"{title}\nNoise variance={sigma2}"
        )

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("Y")

        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
