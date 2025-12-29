import torch
import matplotlib.pyplot as plt

class XorNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        torch.matmul_seed(42)

        self.W1 = torch.randn(input_size, hidden_size)
        self.b1 = torch.randn(1, hidden_size)

        self.W2 = torch.randn(hidden_size, output_size)
        self.b2 = torch.randn(1, output_size)

        self.loss_history = []

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def sigmoid_prieme(self, x):
        s = self.sigmoid(x)
        return s * (1-s)
    
    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2 
    
    def backward(self, X, y, output, learning_rate):
        output_error = output - y
        output_delta = output_error * self.sigmoid_prime(self.z2)

        hidden_error = torch.matmul(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_prime(self.z1)

        W2_grad = torch.matmul(X.T, hidden_delta)
        b2_grad = torch.sum(output_delta, axis=0, keepdim=True)

        W1_grad = torch.matmul(X.T, hidden_delta)
        b1_grad = torch.sum(hidden_delta, axis=0, keepdim=True)

        self.W2 -= learning_rate * W2_grad
        self.b2 -= learning_rate * b2_grad
        self.W1 -= learning_rate * W1_grad
        self.b1 -= learning_rate * b1_grad

    def train(self, X, y, epochs=1000, lr=0.05):
        for i in range(epochs):
            output = self.forward(X)

            self.backward(X, y, output, lr)

            loss = torch.mean((y - output)**2)
            self.loss_history.append(loss.item())

            if i % 1000 == 0:
                print(f"Epoch {i} | Loss: {loss:.5f}")


# Data setup the Xor problem 
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)

y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Initialize
nn = XorNetwork(input_size=2, hidden_size=3, output_size=1)

print("Training XOR Solver...")
nn.train(X, y, epochs=1000, lr=0.05)

print("\n --- Final Predictions ---")
with torch.no_grad():
    pred = nn.forward(X)
    for i in range(4):
        print(f"Input: {X[i].numpy()} | Target: {y[i].item()} | Prediction: {preds[i].item():.4f}")

# --- VISUALIZATION ---
plt.plot(nn.loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.savefig("loss_curve.png")
print("\nLoss curve saved to loss_curve.png")