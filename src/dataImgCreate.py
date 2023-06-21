import cv2
from matplotlib import pyplot as plt

left = cv2.imread("../imgs/aloe3_exp0.png",0)

right = cv2.imread("../imgs/aloe4_exp0.png",0)

plt.subplot(121)
plt.imshow(left,"gray")
plt.title("Left")
plt.axis("off")
plt.subplot(122)
plt.title("Right")
plt.imshow(right,"gray")
plt.axis("off")


plt.show()


