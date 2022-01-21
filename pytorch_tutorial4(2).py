# A typical Pytorch Pipeline consists of these 3 steps
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer 
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradient 
#       - update weights

import torch
import torch.nn as nn
# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32) # Convert it to 2D array
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# Forward Function is replaced with model
# model = nn.Linear(input_size, output_size) # Accepts only 2D array

# Create own Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim): 
            super(LinearRegression, self).__init__()
            # define layers
            self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

# Loss Function is replaced with MSEloss()
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    #w.data = w.data - learning_rate * w.grad
    optimizer.step()
    
    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # Unpack the values
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')