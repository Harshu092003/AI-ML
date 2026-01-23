# Autoencoders are a special type of neural networks that learn to compress data into a compact form and then reconstruct 
# it to closely match the original input. They consist of an:
#     Encoder that captures important features by reducing dimensionality.
#     Decoder that rebuilds the data from this compressed representation. 

# The model trains by minimizing reconstruction error using loss functions like Mean Squared Error or Binary Cross-Entropy. 
# These are applied in tasks such as noise removal, error detection and feature extraction where capturing efficient data representations is important.
# ReLU (Rectified Linear Unit) :
# An activation function that outputs 0 for negative values and the same value for positive values, making training fast and efficient.
import torch
import torch.nn as nn

# =========================
# 1. Training Data (4 features)
# =========================
X = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
    [4.0, 5.0, 6.0, 7.0],
    [5.0, 6.0, 7.0, 8.0],
])

# =========================
# 2. Autoencoder Model
# =========================
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: compress 4 → 2
        self.encoder = nn.Sequential(
            nn.Linear(4, 2), # compress 4 to 2 values
            nn.ReLU()
        )

        # Decoder: reconstruct 2 → 4
        self.decoder = nn.Sequential(
            nn.Linear(2, 4)
        )

    def forward(self, x):
        encoded = self.encoder(x)   # compressed representation
        decoded = self.decoder(encoded)  # reconstructed output
        return encoded, decoded

# =========================
# 3. Model, Loss, Optimizer
# =========================
model = AutoEncoder()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 4. Training Loop
# =========================
epochs = 3000

for epoch in range(epochs):
    encoded, output = model(X)
    loss = loss_fn(output, X)  # reconstruction loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# =========================
# 5. Testing
# =========================
test = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
encoded_test, reconstructed = model(test)

print("\n--- TEST SAMPLE ---")
print("Original Input      :", test)
print("Encoded (Compressed):", encoded_test)
print("Reconstructed Output:", reconstructed)
