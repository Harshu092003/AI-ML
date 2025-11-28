#  learn neural network 

import torch

# ---- Step 1: Create Tensors ----
x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Input
y = torch.tensor([2.0, 4.0, 6.0, 8.0])  # Label (y = 2x relation)

# ---- Step 2: Model parameter (weight) ----
# requires_grad=True means PyTorch will compute gradients for this variable
w = torch.tensor(1.0, requires_grad=True)

# ---- Step 3: Define Loss Function ----
loss_fn = torch.nn.MSELoss()

# ---- Step 4: Define Optimizer (learning rate = 0.01) ----
optimizer = torch.optim.SGD([w], lr=0.01)

# ---- Step 5: Training Loop ----
for epoch in range(100):
    # Forward pass: predicted value
    y_pred = w * x  
    
    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass: calculate gradient
    loss.backward()

    # Update parameter using optimizer
    optimizer.step()

    # Reset gradients to zero (VERY IMPORTANT)
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, Weight: {w.item()}")

print("\nFinal learned weight:", w.item())
