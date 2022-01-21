# Implement Linear Regression using pytorch

import torch 

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# The forward and the loss function will be the same
# Calculate model prediction
def forward(x):
	return w * x

# Calculate loss = MSE 
def loss(y, y_predicted):
	return ((y_predicted - y) ** 2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training 
learning_rate = 0.01
n_iters = 61

for epoch in range(n_iters):
	# Forward Pass (same)
	y_pred = forward(X)
	# Calculate loss for each pass (same)
	l = loss(Y, y_pred)
	# Calculate gradient descent 
	l.backward() # Not as accurate as manual calculation, so we increase the no:of iterations in order to get better accuracy

	# Update Weights 
	with torch.no_grad():
		w -= learning_rate * w.grad
	# Zero Gradients 
	w.grad.zero_()

	if epoch % 10 == 0: 
		print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')