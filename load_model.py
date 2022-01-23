import torch

FILE = 'model.pth'
# Load the model
model = torch.load(FILE)
model.eval() 

for param in model.parameters():
	print(param)