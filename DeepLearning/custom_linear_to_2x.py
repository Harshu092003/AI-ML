import torch
import torch.nn as nn

# Training data
X = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
Y = torch.tensor([[2.0],[4.0],[6.0],[8.0],[10.0]])

# Custom Model Class
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,1))   # trainable parameter
        self.bias = nn.Parameter(torch.randn(1))       # trainable parameter

    def forward(self, x):
        return x * self.weight + self.bias   # y = wx + b

# Create model
model = LinearModel()

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(2000):
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)

    optimizer.zero_grad() # Reset gradients
    loss.backward() # Backpropagation and compute gradients
    optimizer.step() # Update parameters

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# Test
test = torch.tensor([[10.0]])
prediction = model(test)

print("\nPrediction for x=10:", prediction.item())
print("Expected:", 20)

# Show learned parameters
print("\nLearned Weight:", model.weight.item())
print("Learned Bias:", model.bias.item())
