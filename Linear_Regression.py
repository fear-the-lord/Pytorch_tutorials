# Just for practice of Linear Regression using Pytorch
# Import all the necessary dependencies
import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

# 0. Prepare the data
X_numpy, Y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
# Reshape the tensor
Y = Y.view(Y.shape[0], 1)
n_samples, n_features = X.shape

# 1. Design the model
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size) 

# 2. Define the loss and optimizer
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3. Training loop 
num_iters = 100
for epochs in range(num_iters):
	# Forward Pass
	y_predicted = model(X)
	# Loss
	l = loss(y_predicted, Y)
	# Backward Pass
	l.backward() 
	# Update weights
	optimizer.step()
	# Make weights zero 
	optimizer.zero_grad() 

	if (epochs + 1) % 10 == 0:
		[w, b] = model.parameters()
		print(f'epoch: {epochs + 1}, loss = {l.item():.3f}, w = {w[0][0].item():.3f}')


# Plot the result
predicted = model(X).detach().numpy() # Detach it from computational graph, makes requires_grad = False
plt.plot(X_numpy, Y_numpy, 'ro') # Plot the original data as red dots
plt.plot(X_numpy, predicted, 'b') # Regression Line as Blue
plt.show()