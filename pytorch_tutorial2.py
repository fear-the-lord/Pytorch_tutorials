# Gradients are important for our model optimizations
# Pytorch provides the autograd package, which does the work for us

import torch 

# x = torch.randn(3, requires_grad = True) # Normal distribution 
# y = x + 2 # (x, 2) are the two inputs and y is the output
# # Calculating the value of y is the forward pass. 
# # y will have a grad_fn() what will calculate dy/dx and then add it to the back propagation step 
# # grad_fn = <AddBackward0>

# z = y * y * 2
# m = z 
# print(z) # grad_fn = <MulBackward0>
# z = z.mean() # grad_fn = <MeanBackward0>

# # z.backward() # Calculates dz/dx, this only works when z is a scalar value. 
# print(x.grad)

# v = torch.rand(3, dtype = torch.float32)
# m.backward(v)
# print(m)

# # There are 3 ways to stop calculating gradient for a variable
# # Way 1: 
# x.requires_grad_(False) # Affects the variable x in place
# # Way 2: 
# y = x.detach() # Creates a new tensor with the same value, but no gradient is required
# # Way 3: wrap in with
# with torch.no_grad():
# 	y = x + 2
# 	print(y)

# Let's see an example 
weights = torch.ones(4, requires_grad = True)

# for epochs in range(3):
# 	modified = (weights * 3).sum()
# 	# v = torch.rand(4, dtype = torch.float32)
# 	modified.backward()
# 	print(weights.grad)
# 	weights.grad.zero_() # Clear the gradient back to zero 

weights1 = torch.ones(4, requires_grad = True, dtype = torch.float32)
optimizer = torch.optim.SGD([weights1], lr = 0.01)
optimizer.step() # Same as backward()
print(optimizer)
optimizer.zero_grad() # Same as grad.zero_()
