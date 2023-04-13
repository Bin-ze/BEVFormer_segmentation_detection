import os
import cv2
import numpy as np

path = '/root/work_code/BEVFormer/visual_res_base/'

visual_list = list(filter(lambda x: "camera" in x, os.listdir(path)))

for img in visual_list:
    det_img = path + img
    seg_img = path + img.split('_')[0] + '.png'

    det_img = cv2.imread(det_img)
    seg_img = cv2.imread(seg_img)
    seg_img = np.rot90(seg_img, 1, [0, 1])

    scale = det_img.shape[0] / seg_img.shape[0]
    width, height = int(scale * seg_img.shape[1]), det_img.shape[0]

    seg_img = cv2.resize(seg_img, (width, height), interpolation=cv2.INTER_LINEAR)

    # 拼接
    cancat_img = np.hstack((det_img, seg_img))
    cancat_img = cv2.resize(cancat_img, (cancat_img.shape[1]//2, cancat_img.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(f"visual/{img.split('_')[0]}.png", cancat_img)

