import cv2
import numpy as np

path = '/root/work_code/BEVFormer/visual_rot/000681a060c04755a1537cf83b53ba57.png'

det_grid_conf = {
    'xbound': [-51.2, 51.2, 0.68],
    'ybound': [-51.2, 51.2, 0.68],
}

map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
}

def padding_seg_to_det(path):

    seg = cv2.imread(path)
    h, w, _ = seg.shape

    det_w = int((det_grid_conf['xbound'][1] - det_grid_conf['xbound'][0])/(map_grid_conf['xbound'][1] - map_grid_conf['xbound'][0]) * w)
    det_h = det_w

    new_img = np.zeros((det_h, det_w, 3))
    new_img = np.where(new_img == 0, 255, 0)
    new_img[det_h // 2 - h // 2: det_h // 2 + h//2, det_w // 2 - w // 2: det_w // 2 + w//2, :] = seg

    new_img = np.rot90(new_img, 1, [0, 1])


    return new_img


if __name__ == '__main__':

    img = padding_seg_to_det(path=path)
    cv2.imwrite("a.jpg", img)