import os

import cv2
import mmcv
import numpy as np

from nuscenes.nuscenes import NuScenes

PATH_base = '/home/guozebin/work_code/BEVFormer/visual_res_base/'
PATH_small = '/home/guozebin/work_code/BEVFormer/visual_res_small/'
video_path = 'small_base_compare_demo.mp4'

if __name__ == '__main__':
    count = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/guozebin/work_code/BEVFormer/data/nuscenes', verbose=True)
    bevformer_results = mmcv.load('/home/guozebin/work_code/BEVFormer/val/work_dirs/'
                                  'bevformer_small_seg_det_300x300/Tue_Jan_31_16_25_12_2023/pts_bbox/results_nusc.json')
    sample_token_list = list(bevformer_results['results'].keys())[1000:2000]
    for id in range(0, 1000):

        if sample_token_list[id] + '.jpg' not in os.listdir(PATH_base):
            continue

        print(f"handle {PATH_base + sample_token_list[id] +'.jpg'}")
        print(f"handle {PATH_small + sample_token_list[id] +'.jpg'}")
        count += 1
        im_base = os.path.join(PATH_base, sample_token_list[id] + '.jpg')
        im_small = os.path.join(PATH_small, sample_token_list[id] + '.jpg')

        im_base = cv2.imread(im_base)
        # 添加指标值
        im_base = cv2.putText(im_base, "mAP: 41.9, NDS: 51.3, mIoU: 44.1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        im_small = cv2.imread(im_small)
        # 添加指标值
        im_small = cv2.putText(im_small, "mAP: 38.2, NDS: 48.7, mIoU: 40.4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # 添加一个黑色隔离带用于区分两个结果
        split = np.zeros([20, im_base.shape[1], 3])
        # 拼接
        im = np.vstack((im_small, split, im_base))
        if count == 1:
            fps, w, h = 5, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im.astype(np.uint8))

    print('Done!')
