import cv2
import numpy as np
from multiprocessing import Pool

def compute_disparity_block(args):
    imgL, imgR, y, window_radius, max_disp = args
    h, w = imgL.shape
    disparity_row = np.zeros(w, np.float32)
    for x in range(window_radius, w - window_radius):
        best_disp = 0
        min_sad = np.inf
        left_window = imgL[y - window_radius:y + window_radius + 1,
                           x - window_radius:x + window_radius + 1]
        for d in range(max_disp):
            if x - window_radius - d < 0:
                continue
            right_window = imgR[y - window_radius:y + window_radius + 1,
                                x - window_radius - d:x + window_radius + 1 - d]
            sad = np.sum(np.abs(left_window - right_window))
            if sad < min_sad:
                min_sad = sad
                best_disp = d
        disparity_row[x] = best_disp
    return y, disparity_row

def compute_disparity(imgL, imgR, window_size=5, max_disp=256):
    h, w = imgL.shape
    disparity_map = np.zeros((h, w), np.float32)
    window_radius = window_size // 2
    with Pool() as p:
        results = p.map(compute_disparity_block, [(imgL, imgR, y, window_radius, max_disp) for y in range(window_radius, h - window_radius)])
    for y, disparity_row in results:
        disparity_map[y, :] = disparity_row
    return disparity_map

if __name__ == '__main__':
    imgL = cv2.imread('imgs/aloe3_exp0.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('imgs/aloe4_exp0.png', cv2.IMREAD_GRAYSCALE)
    disparity_map = compute_disparity(imgL, imgR, window_size=15, max_disp=256)
    cv2.imshow('Disparity Map', disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
