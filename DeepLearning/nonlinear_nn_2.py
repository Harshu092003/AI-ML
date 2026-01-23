#  Sigmoid Activation Function :
# An activation function that converts any value into a number between 0 and 1, often used to represent probability.

# Adam (Adaptive Moment Estimation) : 

# An optimizer that automatically adjusts learning rates using past gradients, making training faster and more stable.

import torch
import torch.nn as nn
import torch.optim as optim

# Data: temperature → hot (1) or not (0)
X = torch.tensor([[10.0],[15.0],[20.0],[25.0],[30.0],[35.0]])
y = torch.tensor([[0.0],[0.0],[0.0],[1.0],[1.0],[1.0]])

class TempClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layer(x))

model = TempClassifier()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

test = torch.tensor([[12.0],[22.0],[32.0]])
out = model(test)

print("\nPredictions:")
for i, v in enumerate(test):
    prob = out[i].item()
    print(f"{v.item()}°C → Hot Probability: {prob:.3f} → {'HOT' if prob>0.5 else 'NOT HOT'}")
