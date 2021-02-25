import torch



a=torch.randn(4,2,100,200)

b=torch.norm(a,2,dim=1,keepdim=True)

print(b.shape)