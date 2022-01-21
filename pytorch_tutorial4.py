# Linear Regression (Manually)
import numpy as np 

# f = w * x
# f = 2 * x

X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

# Initialize the weights 
w = 0.0

# Calculate model prediction
def forward(x):
	return w * x

# Calculate loss = MSE 
def loss(y, y_predicted):
	return ((y_predicted - y) ** 2).mean()


# Calculate gradient 
# y_predicted = w * x
# MSE = (1 / N) * (y_predicted - y) ** 2
# dMSE / dw = (1 / N) (2 * x) (w * x - y)
def gradient(x, y, y_predicted):
	return np.dot(2 * x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training 
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
	# Forward Pass
	y_pred = forward(X)
	# Calculate loss for each pass 
	l = loss(Y, y_pred)
	# Calculate gradient descent 
	dw = gradient(X, Y, y_pred)

	# Update Weights 
	w -= learning_rate * dw

	if epoch % 1 == 0: 
		print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
