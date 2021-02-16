import torch


a=torch.tensor([1,1,0])
b=torch.tensor([0,1,0])

c=a&b

print(c)