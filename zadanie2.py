import numpy as np
import matplotlib.pyplot as plt


# 1. Funkcja Ackley'a
def ackley_function(x1, x2):
    part1 = -0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
    part2 = 0.5 * (np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2))
    return -20.0 * np.exp(part1) - np.exp(part2) + np.e + 20.0


# 2. Generowanie zbioru treningowego
def generate_training_data(n_samples=200, low=-2.0, high=2.0, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(low, high, size=(n_samples, 2))
    y = []
    for i in range(n_samples):
        y_val = ackley_function(X[i, 0], X[i, 1])
        y.append(y_val)
    y = np.array(y).reshape(-1, 1)
    return X, y


# 3. Definicja prostej sieci neuronowej (MLP)
class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, seed=42):
        np.random.seed(seed)
        self.lr = lr

        # Parametry warstwy 1 (wejście -> ukryta)
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))

        # Parametry warstwy 2 (ukryta -> wyjście)
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
        """
        Trening z wykorzystaniem mini-batch:
          - batch_size określa liczbę przykładów w każdej porcji (mini-batch).
        """
        n_samples = X.shape[0]

        for e in range(epochs):
            # Mieszamy indeksy próbek (shuffle)
            permutation = np.random.permutation(n_samples)

            # Mieszamy X i y tą samą permutacją
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Pętla przez mini-batche
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward dla mini-batcha
                y_hat_batch = self.forward(X_batch)

                # Backward i aktualizacja wag
                self.backward(X_batch, y_batch, y_hat_batch)

            # -- Po zakończeniu jednej epoki (wszystkie mini-batche) można ocenić błąd
            # Forward na całym zbiorze treningowym (opcjonalne, ale pomaga w monitorowaniu)
            y_hat_full = self.forward(X)
            loss = np.mean((y_hat_full - y) ** 2)

            if print_loss and e % 200 == 0:
                print(f"[Epoka {e}] MSE (na całym zbiorze): {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


# 4. Trenowanie sieci
if __name__ == "__main__":
    # Zakładamy, że mamy X_train i y_train (np. 200 punktów)
    X_train, y_train = generate_training_data(n_samples=200)

    mlp = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1, lr=0.01)

    # Trening z mini-batchami (rozmiar batch = 32), przez np. 2000 epok
    mlp.train(X_train, y_train, epochs=2000, batch_size=32, print_loss=True)


    # 5. Testowanie na siatce punktów
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


    # 6. Wizualizacja
    def plot_results(grid_points, real_values, approx_values, X_train, y_train):
        x1 = grid_points[:, 0]
        x2 = grid_points[:, 1]

        plt.figure(figsize=(12, 5))

        # Rzeczywista Ackley
        plt.subplot(1, 2, 1)
        sc1 = plt.scatter(x1, x2, c=real_values, cmap='viridis', s=5)
        plt.colorbar(sc1)
        plt.scatter(X_train[:, 0], X_train[:, 1], c='red', marker='x', s=50, label='Węzły uczące')
        plt.title("Rzeczywiste wartości Ackley'a")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

        # Aproksymacja
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
