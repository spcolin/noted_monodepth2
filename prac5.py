import torch,torchvision
from PIL import Image
from layers import *

K=torch.tensor([[[535.4,0,320.1,0],
                [0,539.2,247.6,0],
                [0,0,1,0],
                [0,0,0,1]]],dtype=torch.float)


img1_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/rgb/1341845821.488014.png"
depth1_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/depth/1341845821.488036.png"

img2_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/rgb/1341845822.889712.png"
depth2_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/depth/1341845822.889734.png"

img3_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/rgb/1341845824.333783.png"
depth3_path="E:/TUM/rgbd_dataset_freiburg3_sitting_xyz/depth/1341845824.333796.png"


img1=Image.open(img1_path)
depth1=Image.open(depth1_path)

img2=Image.open(img2_path)
depth2=Image.open(depth2_path)

img3=Image.open(img3_path)
depth3=Image.open(depth3_path)

img1=torchvision.transforms.ToTensor()(img1).unsqueeze(0)
depth1=torchvision.transforms.ToTensor()(depth1).unsqueeze(0)/5000.0

img2=torchvision.transforms.ToTensor()(img2).unsqueeze(0)
depth2=torchvision.transforms.ToTensor()(depth2).unsqueeze(0)/5000.0

img3=torchvision.transforms.ToTensor()(img3).unsqueeze(0)
depth3=torchvision.transforms.ToTensor()(depth3).unsqueeze(0)/5000.0


T21=torch.tensor([[[   0.999926, -0.00983208 ,-0.00718345 , -0.0991151],
                [ 0.00962893   , 0.999481 , -0.0280589 , -0.0154967],
                [ 0.00745927 ,  0.0279877  ,  0.999502 , 0.00664393],
                  [          0     ,      0      ,     0     ,      1]]])
T23=torch.tensor([[[   0.999347 ,-0.00135086 , -0.0361122  ,  0.190935],
                [  0.0020131  ,   0.99979  , 0.0182602  , 0.0366029],
                [ 0.0360849 , -0.0183209  ,  0.999139 , 0.00397738],
                  [  0     ,      0      ,     0     ,      1]]])


backproject_depth = BackprojectDepth(1, 480, 640)
project_3d = Project3D(1, 480, 640)


cam_points = backproject_depth(depth2, torch.inverse(K))

# pix_coords23=project_3d(cam_points,K,T23)

print(depth1[0,0,400,400])
cam_points1 = backproject_depth(depth1, torch.inverse(K))
print(cam_points1.view(1,4,480,640)[0,:,400,400])
source_3d1=Coord_3d_trans(cam_points1,T21)
source_3d1=source_3d1.view(source_3d1.shape[0],3,480,-1)


pix_coords21 = project_3d(cam_points, K, T21)

warped_pre_3d1=F.grid_sample(
        source_3d1,
        pix_coords21,
        padding_mode="border")
print(cam_points.view(1,4,480,-1)[0,:,400,400])
print(source_3d1[0,:,376,381])
print(warped_pre_3d1[0,:,400,400])

