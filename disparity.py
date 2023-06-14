import numpy as np
import cv2
from matplotlib import pyplot as plt

# 스테레오 이미지 읽어오기
imgL = cv2.imread('view6.png',0)  # 왼쪽 이미지
imgR = cv2.imread('view0.png',0) # 오른쪽 이미지

# StereoBM 객체 생성
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

# disparity 계산
disparity = stereo.compute(imgL,imgR)

# 결과 출력
plt.imshow(disparity,'gray')
plt.show()
