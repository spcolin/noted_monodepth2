from layers import *
import torch

h=192
w=640
k=torch.tensor([[[371.2000,   0.0000, 320.0000,   0.0000],
         [  0.0000, 368.6400,  96.0000,   0.0000],
         [  0.0000,   0.0000,   1.0000,   0.0000],
         [  0.0000,   0.0000,   0.0000,   1.0000]]])
# inv_k=torch.inverse(k)
#
#
# bc=BackprojectDepth(1,h,w)
#
# print(bc.pix_coords[0,:,1000])
#
# depth=torch.rand(1,1,192,640)
# print(depth[0,0,1,360])
# cam_coord=bc(depth,inv_k).view(1,4,192,-1)
# print(cam_coord[0,:,1,360])

T21=torch.tensor([[   0.999926, -0.00983208 ,-0.00718345 , -0.0991151],
                [ 0.00962893   , 0.999481 , -0.0280589 , -0.0154967],
                [ 0.00745927 ,  0.0279877  ,  0.999502 , 0.00664393],
                  [          0     ,      0      ,     0     ,      1]])

a=torch.tensor([[   0.999926, -0.00983208 ,-0.00718345 ],
                [ 0.00962893   , 0.999481 , -0.0280589 ],
                [ 0.00745927 ,  0.0279877  ,  0.999502 ]])

b=torch.tensor([[-0.0991151],[ -0.0154967],[0.00664393]])

c=torch.tensor([[0.4077],[0.6138],[1.9113]])
print(a.shape)
print(b.shape)
print(c.shape)

print(a.mm(c)+b)
