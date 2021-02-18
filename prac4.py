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


a=torch.tensor([[ 1.0000e+00, -4.1999e-04,  5.2805e-04],
         [ 4.2026e-04,  1.0000e+00, -4.9384e-04],
         [-5.2783e-04,  4.9407e-04,  1.0000e+00]])
b=torch.tensor([[-6.9653e-04],[2.4728e-04],[-7.3439e-04]])

c=torch.tensor([[-0.0102],[0.0020],[0.2017]])
print(a.shape)
print(b.shape)
print(c.shape)

print(a.mm(c)+b)
