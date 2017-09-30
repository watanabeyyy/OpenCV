import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/4_5.bmp', cv2.IMREAD_COLOR)
bi = cv2.bilateralFilter(img, 51, 10, 10)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(bi, cv2.COLOR_BGR2RGB))
plt.title("bi")
plt.xticks([]),plt.yticks([])

plt.show()