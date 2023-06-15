import cv2
import numpy as np


def compute_disparity(save_dir_path, rectified_l, rectified_r, Q):

    window_size = 7
    min_disp = 0
    max_disp = 64
    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize= window_size,
        preFilterCap=63,
        uniquenessRatio = 15,
        speckleWindowSize = 10,
        speckleRange = 1,
        disp12MaxDiff = 20,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    left_matcher = stereo
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    l = 70000
    s = 1.2

    disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    disparity_filter.setLambda(l)
    disparity_filter.setSigmaColor(s)

    d_l = left_matcher.compute(rectified_l, rectified_r)
    d_r = right_matcher.compute(rectified_r, rectified_l)

    d_l = np.int16(d_l)
    d_r = np.int16(d_r)
    
    d_filter = disparity_filter.filter(d_l, rectified_l, None, d_r)

    disparity = cv2.normalize(d_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(save_dir_path + 'disparity_map.png', disparity)


save_dir_path = "./"
compute_disparity(save_dir_path,"aloe_left.png","aloe_right.png",0)