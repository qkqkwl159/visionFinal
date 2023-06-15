import cv2
import sys
import numpy as np

def getDisparity(imgLeft, imgRight, method="BM"):
    gray_left = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
    print(gray_left.shape)
    c, r = gray_left.shape
    if method == "BM":
        sbm = cv2.StereoBM_create(numDisparities=32, blockSize=17)
        sbm.setMinDisparity(16)
        sbm.setNumDisparities(32)
        sbm.setBlockSize(17)
        sbm.setDisp12MaxDiff(0)
        sbm.setUniquenessRatio(10)
        sbm.setSpeckleRange(32)
        sbm.setSpeckleWindowSize(100)
        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    elif method == "SGBM":
        sbm = cv2.StereoSGBM_create(minDisparity=1,
                                     numDisparities=32,
                                     blockSize=5,

                                     uniquenessRatio=5,
                                     speckleWindowSize=5,
                                     speckleRange=5,
                                     disp12MaxDiff=2,
                                     P1=8*3*5**2,
                                     P2=32*3*5**2,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_visual

imgLeft = cv2.imread("aloe_left.png")
imgRight = cv2.imread("aloe_right.png")
try:
    method = sys.argv[3]
except IndexError:
    method = "BM"

disparity = getDisparity(imgLeft, imgRight, method)
cv2.imshow("disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
