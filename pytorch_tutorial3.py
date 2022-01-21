# Implement Backpropagation 
# The whole concept consists of 3 steps
# 1. Forward Pass: Compute Loss
# 2. Compute Local gradients
# 3. Backward Pass: Compute dLoss / dWeights using Chain Rule

import torch 

x = torch.tensor(1.0) # x = 1
y = torch.tensor(2.0) # y = 2

w = torch.tensor(1.0, requires_grad = True) # w = 1

# Forward Pass to compute the loss 
y_hat = w * x
loss = (y_hat - y) ** 2 # Squared Error

print(loss) 

# Backward Pass (Calculated automatically)
loss.backward() # Start from loss
print(w.grad) # Should be -2


# Update the weights
# Next forward and backward pass
