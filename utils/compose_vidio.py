import os

import cv2
import mmcv

from nuscenes.nuscenes import NuScenes

PATH = '/home/guozebin/work_code/BEVFormer/visual_res_base'
video_path = 'seg_det_demo_best_v1.mp4'

if __name__ == '__main__':
    count = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/guozebin/work_code/BEVFormer/data/nuscenes', verbose=True)
    bevformer_results = mmcv.load(
        '/home/guozebin/work_code/BEVFormer/val/work_dirs/bevformer_small_seg_det_300x300/Tue_Jan_31_16_25_12_2023/pts_bbox/results_nusc.json')
    sample_token_list = list(bevformer_results['results'].keys())[1000:2000]
    for id in range(0, 1000):

        if sample_token_list[id] + '.jpg' not in os.listdir(PATH):
            continue

        print(f"handle {PATH + sample_token_list[id] +'.jpg'}")
        count += 1
        im = os.path.join(PATH, sample_token_list[id] + '.jpg')
        im = cv2.imread(im)

        if count == 1:
            fps, w, h = 5, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im)
    print('Done!')