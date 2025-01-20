import numpy as np
import time
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Xavier initialization
        limit_in = np.sqrt(6.0 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit_in, limit_in, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        limit_out = np.sqrt(6.0 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(-limit_out, limit_out, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        # x jest wejściem do ReLU, jego pochodna:
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, x):
        self.input = x
        self.hidden_input = np.dot(self.input, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward(self, y_true):
        # gradient wyjścia
        output_error = self.final_output - y_true  # [N,10]

        # gradient warstwy ukrytej
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_input)

        # aktualizacja wag wyjściowych
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= self.learning_rate * np.sum(output_error, axis=0, keepdims=True)

        # aktualizacja wag wejściowych
        self.weights_input_hidden -= self.learning_rate * np.dot(self.input.T, hidden_error)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, x_train, y_train, epochs, batch_size=64, patience=20):
        best_loss = float('inf')
        no_improve_count = 0
        n_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Losowe tasowanie danych
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            start_time = time.time()
            # Trenowanie mini-batchami
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # forward + backward
                output = self.forward(x_batch)
                self.backward(y_batch)

            # Oblicz stratę na całym zbiorze treningowym
            train_output = self.forward(x_train)
            loss = self.cross_entropy(y_train, train_output)
            predictions_train = self.predict(x_train)
            y_train_labels = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(predictions_train == y_train_labels) * 100

            duration = time.time() - start_time
            print(f"Epoka {epoch+1}/{epochs}, Strata: {loss:.6f}, Dokładność tren.: {train_accuracy:.2f}%, Czas: {duration:.2f}s")

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Zatrzymanie w epokach {epoch+1} z powodu braku poprawy.")
                    break

    def predict(self, x):
        output = self.forward(x)
        predictions = np.argmax(output, axis=1)
        return predictions

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        buffer = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
        return data

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

def prepare_data(train_images_path, train_labels_path, test_images_path, test_labels_path):
    x_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    x_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)

    # Normalizacja
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0

    # One-hot encoding etykiet
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    # Ograniczenie zbioru dla szybszych testów
    x_train = x_train[:60000]
    y_train_onehot = y_train_onehot[:60000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    y_test_onehot = y_test_onehot[:1000]

    return x_train, y_train_onehot, x_test, y_test, y_test_onehot

if __name__ == "__main__":
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    epochs = 20  # Można zwiększyć np. do 50-100 by uzyskać lepsze wyniki

    train_images_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'
    test_images_path = 't10k-images.idx3-ubyte'
    test_labels_path = 't10k-labels.idx1-ubyte'

    x_train, y_train, x_test, y_test, y_test_onehot = prepare_data(
        train_images_path, train_labels_path, test_images_path, test_labels_path
    )

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    nn.train(x_train, y_train, epochs)

    predictions = nn.predict(x_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Dokładność na danych testowych: {accuracy:.2f}%")

    # Wyświetlanie losowego obrazu z bazy testowej i przewidywanie
    idx = np.random.randint(0, len(x_test))
    random_image = x_test[idx].reshape(28, 28)
    plt.imshow(random_image, cmap='gray')
    plt.title("Losowy przykład z bazy testowej")
    plt.show()

    # Przewidywanie dla pojedynczego obrazu
    prediction_single = nn.predict(x_test[idx].reshape(1, -1))
    print(f"Przewidziana cyfra: {prediction_single[0]}")




