import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


def getDisparity(imgLeft, imgRight, method):
    gray_left = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
    print(gray_left.shape)
    c, r = gray_left.shape
    if method == "BM":
        print("BM : START")
        sbm = cv2.StereoBM_create()
        sbm.setMinDisparity(16)  
        #defalt 16
        sbm.setNumDisparities(32)
        #defalt 32
        sbm.setBlockSize(19)
        #defalt 17
        sbm.setDisp12MaxDiff(0)
        #defalt  0
        sbm.setUniquenessRatio(0)
        #defalt 10
        sbm.setSpeckleRange(128)
        #defalt 16
        sbm.setSpeckleWindowSize(8192)
        #defalt 100
        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif method == "SGBM":
        print("SGBM : START")
        sbm = cv2.StereoSGBM_create()
        sbm.setMinDisparity(16)  
        #defalt 16
        sbm.setNumDisparities(32)
        #defalt 32
        sbm.setBlockSize(17)
        #defalt 17
        sbm.setDisp12MaxDiff(0)
        #defalt  0
        sbm.setUniquenessRatio(0)
        #defalt 10
        sbm.setSpeckleRange(32)
        #defalt 16
        sbm.setSpeckleWindowSize(2048)
        #defalt 100
        '''mindisparity=1,
        numdisparities=16,
        blocksize=5,

        uniquenessratio=5,
        specklewindowsize=5,
        specklerange=5,
        disp12maxdiff=2,
        p1=8*3*5**2,
        p2=32*3*5**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        '''
        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_visual
#두 영상의 이미지가 너무 멀면 disparity가 생성이 되지않아 이상한 영상이 출력된다.
imgLeft = cv2.imread("./imgs/aloe3_exp0.png")
imgRight = cv2.imread("./imgs/aloe4_exp0.png")
try:
    method = sys.argv[3]
except IndexError:
    method = "SGBM"


disparity = getDisparity(imgLeft, imgRight, method)

plt.figure(figsize=(12,10))
plt.imshow(disparity,cmap='gray')
plt.title("SGBM_disparity")
plt.axis('off')
plt.show()
# cv2.imshow("disparity", disparity)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
