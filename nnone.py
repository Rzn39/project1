import random
import math

# --- Activation Functions ---
def relu(x):
    return [max(0, xi) for xi in x]

def relu_derivative(x):
    return [1 if xi > 0 else 0 for xi in x]

# --- Initialize Weights ---
def initialize_weights(input_size, output_size):
    return [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(input_size)]

def initialize_biases(size):
    return [0.0 for _ in range(size)]

# --- Dot Product ---
def dot_product(vec, mat):
    return [sum(vec[i] * mat[i][j] for i in range(len(vec))) for j in range(len(mat[0]))]

# --- Vector Addition ---
def add_vectors(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

# --- Adam Optimizer ---
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = [[0.0 for _ in layer] for layer in parameters]
        self.v = [[0.0 for _ in layer] for layer in parameters]

    def update(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            for j in range(len(params[i])):
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grads[i][j]
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (grads[i][j] ** 2)

                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)

                params[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)

# --- Neural Network ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [initialize_weights(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [initialize_biases(size) for size in layer_sizes[1:]]

        # Initialize Adam optimizers
        self.optimizers = [AdamOptimizer(layer) for layer in self.weights]

    def forward(self, x):
        self.zs = []
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = add_vectors(dot_product(self.activations[-1], w), b)
            self.zs.append(z)
            a = relu(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y_true):
        # Mean squared error derivative
        loss_grad = [(a - y) * 2 for a, y in zip(self.activations[-1], y_true)]
        grads_w = []
        grads_b = []

        delta = [loss_grad[i] * relu_derivative(self.zs[-1])[i] for i in range(len(loss_grad))]

        for l in reversed(range(len(self.weights))):
            a_prev = self.activations[l]
            w = self.weights[l]

            grad_w = [[a_prev[i] * delta[j] for j in range(len(delta))] for i in range(len(a_prev))]
            grad_b = delta

            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)

            if l > 0:
                delta_new = []
                for i in range(len(a_prev)):
                    error = sum(delta[j] * w[i][j] for j in range(len(delta)))
                    delta_new.append(error * relu_derivative(self.zs[l - 1])[i])
                delta = delta_new

        # Update weights and biases using Adam
        for i in range(len(self.weights)):
            self.optimizers[i].update(self.weights[i], grads_w[i])
            self.biases[i] = [b - 0.001 * gb for b, gb in zip(self.biases[i], grads_b[i])]

    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            out = self.forward(x)
            loss = sum((o - y[i]) ** 2 for i, o in enumerate(out)) / len(out)
            self.backward(y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- User Input ---
def get_user_defined_network():
    input_neurons = 8
    output_neurons = 8

    print("Enter the number of hidden layers:")
    num_layers = int(input(">> "))

    hidden_layers = []
    for i in range(num_layers):
        print(f"Enter size of hidden layer {i + 1}:")
        size = int(input(">> "))
        hidden_layers.append(size)

    return input_neurons, hidden_layers, output_neurons

# --- Example Usage ---
if __name__ == "__main__":
    # Get architecture
    input_size, hidden_layers, output_size = get_user_defined_network()

    # Create the model
    model = SimpleNeuralNetwork(input_size, hidden_layers, output_size)

    # Dummy data
    x = [random.uniform(0, 1) for _ in range(8)]
    y = [random.uniform(0, 1) for _ in range(8)]

    # Train model
    model.train(x, y, epochs=1000)

    # Test output
    print("Input:", x)
    print("Output:", model.forward(x))
