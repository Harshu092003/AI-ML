
# LSTM (Long Short-Term Memory):
# Learns from sequences and remembers important past information
# short term : storing recent data points
# long term  : retaining important info over longer sequences

# Inside LSTM (3 gates):
# 1. Forget gate  -> removes unnecessary old memory
# 2. Input gate   -> adds important new information
# 3. Output gate  -> produces output using stored memory


import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM layer
        # input_size  = number of features per time step (e.g., 1 value at each step)
        # hidden_size = number of features LSTM learns internally (memory size)
        # num_layers  = stacked LSTM layers (2 layers = deeper understanding)
        # batch_first = input shape will be (batch = [[1.0], [2.0], [3.0], [4.0]], time_steps = 4 , features = 1 (only float values not other features))
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=20,
            num_layers=2,
            batch_first=True
        )

        # Fully connected (dense) layer
        # Converts LSTM output (20 features) into final prediction (1 value)
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        # x shape: (batch, time_steps, features)
        # Example: (3, 4, 1)

        # Pass input through LSTM
        # out shape: (batch, time_steps, hidden_size)
        # _ contains (hidden_state, cell_state) which we don't need here
        out, _ = self.lstm(x)

        # Take output from the LAST time step
        # out[:, -1, :] shape: (batch, hidden_size)
        # This represents the summary of the entire sequence
        last_time_step = out[:, -1, :]

        # Pass summary through fully connected layer
        # Output shape: (batch, 1)
        return self.fc(last_time_step)


model = LSTMModel()

# Data (NORMALIZED)
# Divide by 10 for stability
X = torch.tensor([
    [[1.0], [2.0], [3.0], [4.0]],
    [[2.0], [3.0], [4.0], [5.0]],
    [[3.0], [4.0], [5.0], [6.0]],
    [[4.0], [5.0], [6.0], [7.0]],
    [[5.0], [6.0], [7.0], [8.0]],
]) / 10.0

Y = torch.tensor([
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
]) / 10.0

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

# Test (NORMALIZED)
test = torch.tensor([[[4.0], [5.0], [6.0], [7.0]]]) / 10.0
prediction = model(test)

# De-normalize output
prediction = prediction * 10

print("\nPrediction:", prediction.item())
print("Expected: 8")
