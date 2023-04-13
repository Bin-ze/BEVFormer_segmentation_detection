import torch
import cv2
import numpy as np

PATH = '/home/guozebin/work_code/BEVFormer/visual/gt000050_000.jpg'

#
img  = cv2.imread(PATH)
img1 = torch.from_numpy(img)
# test rot
img = torch.rot90(img1, k=-1, dims=[0,1])
cv2.imwrite('rot90.jpg', img.numpy())

img = torch.flip(img, dims=[1])

cv2.imwrite('flip.jpg', img.numpy())
