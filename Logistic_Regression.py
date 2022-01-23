import torch 
import torch.nn as nn 
import numpy as np 
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


# 0. Prepare the data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
n_samples, n_features = X.shape 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1) # Becomes a column vector 
Y_test = Y_test.view(Y_test.shape[0], 1)

# 1. Build the Model 
class LogisticRegression(nn.Module):
	def __init__(self, n_input_features):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(n_input_features, 1)

	def forward(self, x):
		y_predicted = torch.sigmoid(self.linear(x))
		return y_predicted

model = LogisticRegression(n_features)

# 2. Loss and optimizer 
learning_rate = 0.01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3. Training Loop 
n_iters = 1000
for epochs in range(n_iters):
	# Forward pass 
	y_predicted = model(X_train)
	# Loss
	l = loss(y_predicted, Y_train)
	# Backward Pass
	l.backward()
	# Update Weights 
	optimizer.step()
	# Zero gradients
	optimizer.zero_grad()

	if(epochs + 1) % 100 == 0:
		print(f'Epoch: {epochs + 1}, Loss = {l.item():.4f}')

# Evaluate the model
with torch.no_grad():
	y_predicted = model(X_test)
	y_predicted_cls = y_predicted.round()
	accuracy = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
	print(f'accuracy = {accuracy:.4f}')