'''
Common norms such as L1 norm, L2 norm, Frobenius_norm, etc.
'''

import torch

# Example vectors and matrices
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.5, 2.5, 3.5])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# An easy way using torch's api:
# # L1 Norm
# l1_norm = torch.norm(x, p=1)
# print("L1 Norm of x:", l1_norm.item())
#
# # L2 Norm
# l2_norm = torch.norm(x, p=2)
# print("L2 Norm of x:", l2_norm.item())
#
# # Infinity Norm
# infinity_norm = torch.norm(x, p=float('inf'))
# print("Infinity Norm of x:", infinity_norm.item())
#
# # Frobenius Norm (for matrices)
# frobenius_norm = torch.norm(A, p='fro')
# print("Frobenius Norm of A:", frobenius_norm.item())
#
# # Mean Squared Error (MSE)
# mse = torch.mean((x - y) ** 2)
# print("Mean Squared Error between x and y:", mse.item())

# L1 Norm
l1_norm = torch.sum(torch.abs(x))
print("L1 Norm of x:", l1_norm.item())

# L2 Norm
l2_norm = torch.sqrt(torch.sum(x ** 2))
print("L2 Norm of x:", l2_norm.item())

# Infinity Norm
infinity_norm = torch.max(torch.abs(x))
print("Infinity Norm of x:", infinity_norm.item())

# Frobenius Norm for matrix A
# It's exactly the matrix version of  L2 norm
frobenius_norm = torch.sqrt(torch.sum(A ** 2))
print("Frobenius Norm of A:", frobenius_norm.item())
