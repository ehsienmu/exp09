import torch
import numpy as np
#define a PyTorch Tensor usning Python List
a = torch.tensor([[2.,1.], [3.,4.], [4, 5]])
# compute mean, std, and var column-wise
m = torch.mean(a, axis = 0)
s = torch.std(a, axis = 0)
v = torch.var(a, axis = 0)
# print mean, std, and var
print("Column wise\nMean:{}\n std: {}\n Var: {}".format(m,s,v))
# compute mean, std, and var row-wise
m = torch.mean(a, axis = 1)
s = torch.std(a, axis = 1)
v = torch.var(a, axis = 1)
# print mean, std, and var
print("Row wise\nMean:{}\n std: {}\n Var: {}".format(m,s,v))