import numpy as np
import matplotlib.pyplot as plt

def ackley_function(x1, x2):
    part1 = -0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
    part2 = 0.5 * (np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2))
    return -20.0 * np.exp(part1) - np.exp(part2) + np.e + 20.0

def generate_training_data(n_samples=200, low=-2.0, high=2.0, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(low, high, size=(n_samples, 2))
    y = []
    for i in range(n_samples):
        y_val = ackley_function(X[i, 0], X[i, 1])
        y.append(y_val)
    y = np.array(y).reshape(-1, 1)
    return X, y

class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, seed=42):
        np.random.seed(seed)
        self.lr = lr

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))

        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.Z1 = X @ self.w1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.w2 + self.b2
        y_hat = self.Z2
        return y_hat

    def backward(self, X, y, y_hat):
        m = X.shape[0]
        dZ2 = (y_hat - y) / m
        dW2 = self.A1.T @ dZ2
        dB2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.w2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1
        dB1 = np.sum(dZ1, axis=0, keepdims=True)

        self.w2 -= self.lr * dW2
        self.b2 -= self.lr * dB2
        self.w1 -= self.lr * dW1
        self.b1 -= self.lr * dB1

    def train(self, X, y, epochs=1000, batch_size=32, print_loss=True):
        n_samples = X.shape[0]

        for e in range(epochs):
            permutation = np.random.permutation(n_samples)

            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_hat_batch = self.forward(X_batch)

                self.backward(X_batch, y_batch, y_hat_batch)

            y_hat_full = self.forward(X)
            loss = np.mean((y_hat_full - y) ** 2)

            if print_loss and e % 200 == 0:
                print(f"[Epoka {e}] MSE (na całym zbiorze): {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    X_train, y_train = generate_training_data(n_samples=200)

    mlp = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1, lr=0.01)

    mlp.train(X_train, y_train, epochs=2000, batch_size=32, print_loss=True)


    def test_approximation(mlp, step=0.02):
        x_vals = np.arange(-2.0, 2.0 + step, step)
        y_vals = np.arange(-2.0, 2.0 + step, step)

        grid_points = []
        for xv in x_vals:
            for yv in y_vals:
                grid_points.append([xv, yv])
        grid_points = np.array(grid_points)

        real_values = ackley_function(grid_points[:, 0], grid_points[:, 1])
        approx_values = mlp.predict(grid_points).reshape(-1)

        mse = np.mean((approx_values - real_values) ** 2)
        print(f"Średni błąd kwadratowy (MSE) na siatce testowej: {mse:.6f}")

        return grid_points, real_values, approx_values


    grid_points, real_values, approx_values = test_approximation(mlp, step=0.02)

    def plot_results(grid_points, real_values, approx_values, X_train, y_train):
        x1 = grid_points[:, 0]
        x2 = grid_points[:, 1]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sc1 = plt.scatter(x1, x2, c=real_values, cmap='viridis', s=5)
        plt.colorbar(sc1)
        plt.scatter(X_train[:, 0], X_train[:, 1], c='red', marker='x', s=50, label='Węzły uczące')
        plt.title("Rzeczywiste wartości Ackley'a")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

        plt.subplot(1, 2, 2)
        sc2 = plt.scatter(x1, x2, c=approx_values, cmap='viridis', s=5)
        plt.colorbar(sc2)
        plt.scatter(X_train[:, 0], X_train[:, 1], c='red', marker='x', s=50, label='Węzły uczące')
        plt.title("Aproksymacja sieci neuronowej")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

        plt.tight_layout()
        plt.show()


    plot_results(grid_points, real_values, approx_values, X_train, y_train)
