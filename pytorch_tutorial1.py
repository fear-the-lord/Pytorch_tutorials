# Tensors are like arrays in pytorch, it can be of any dimension. 
import torch 
import numpy as np

##### CREATING TENSORS #####
# Tensor of size 2 * 2 containing all ones
x = torch.ones(2, 2)
print(x) # By default its data type is float32

# Tensor of size 2 * 2 containing all zeros
x = torch.zeros(2, 2)
print(x)

# Initialize an empty tensor, dimension 1, more like a scalar
x = torch.empty(1)
print(x)

# Create a random 3D tensor of size 2 x 2 x 3
x = torch.rand(2, 2, 3)
print(x)

# Create a tensor of type integer
x = torch.ones(2, 2, dtype = torch.int)
print(x.dtype) # Returns datatype of the tensor
print(x.size()) # Returns size of the tensor

# Creating a tensor from a list
x = torch.tensor([2.5, 0.1])
print(x)

##### OPERATIONS ON TENSOR #####

x = torch.rand(2, 2)
y = torch.rand(2, 2)
z = x + y # Element wise addition 
z = torch.add(x, y) # Does the same thing as z = x + y
print(z)

# Every variable having an underscore does an inplace operation
y.add_(x) # Inplace addition y = y + x
print(y)

z = x - y
z = torch.sub(x, y) # Element wise subtraction
print(z)

z = x * y # Multiplication operation
z = torch.mul(x, y)
y.mul_(x)

z = x / y # Division operation
z = torch.div(x, y)
y.div_(x)

##### SLICING OPERATIONS #####
x = torch.rand(5, 3)
print(x)
print(x[:, 0]) # Prints only the first column of all the rows 
print(x[1, :]) # Prints the first row of all the columns
print(x[1, 1]) # Prints the element at position (1, 1). This will actually return a tensor. 
# If our tensor has only only element, and we want the exact value of it, then we do
print(x[1, 1].item())

# RESHAPING A TENSOR (Number of elements must be same after a reshaping)
x = torch.rand(4, 4) # Creating a 4 x 4 tensor
y = x.view(16) # Re-shaping it as a 16 * 1 i.e. a single dimensional array with 16 elements 

# If we provide the first dimension as -1, and provide the 2nd dimension
# it automatically detects the first dimension
y = x.view(-1, 8)
print(y.size())

##### WORKING WITH NUMPY #####
a = torch.ones(5)
b = a.numpy()
print(type(b)) # It is converted to a numpy.ndarray
print(b)

# *Important* If we are working on CPU in pytorch and not GPU,
# then both the numpy array and the tensor will point to the same memory location. 
# Thus, modifying any one of the arrays inplace, will modify the other, 
# as they share the same memory space. 

a.add_(1) # Add 1 to the tensor a
print(a) # a as well as b changes from 1 to 2 
print(b)

# It happens the other way too
a = np.ones(5) # Create a numpy array of all 1s
print(a)
b = torch.from_numpy(a) # Convert from numpy to tensor
print(b)

a += 1 # Add 1 to the numpy array 
print(a) # Both the numpy array and tensor will change
print(b)

if torch.cuda.is_available(): 
	device = torch.device("cuda")
	x = torch.ones(5, device = device) # This creates a tensor and puts it on the GPU
	# Another way of creating a GPU tensor
	y = torch.ones(5) # Create the tensor normally 
	y = y.to(device) # Move it to the GPU 

	# This operation will be much faster now
	z = x + y 

	# This will give an error, since the GPU sensor(z) 
	# cannot be converted to a numpy array.
	# z.numpy()
	# So we need to move this to the CPU again
	z = z.to("cpu")
	# z = z.numpy()
	print(z)

# By default requires_grad = False
# This means that if we need to optimize 
# certain variable i.e. calculate gradient, then we need to set it True. 
x = torch.ones(5, requires_grad = True)
print(x)