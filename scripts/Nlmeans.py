import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('./img/4_5.bmp', cv2.IMREAD_COLOR)

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,3,21)

kernel_gradient_5x5 = np.array([
                            [ 1,  0,  0,  0,  1],
                            [ 0,  1,  1,  1,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 0, -1, -1, -1,  0],
                            [-1,  0,  0,  0, -1]
                            ], np.float32)
img_gradient_5x5 = cv2.filter2D(dst, -1, kernel_gradient_5x5)

ret, thresh_150 = cv2.threshold(img_gradient_5x5, 150, 255, cv2.THRESH_BINARY_INV)

size = 4

plt.subplot(size,1,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([]),plt.yticks([])
plt.subplot(size,1,2)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("NLMeans")
plt.xticks([]),plt.yticks([])
plt.subplot(size,1,3)
plt.imshow(cv2.cvtColor(img_gradient_5x5, cv2.COLOR_BGR2RGB))
plt.title("gradient")
plt.xticks([]),plt.yticks([])
plt.subplot(size,1,4)
plt.imshow(cv2.cvtColor(thresh_150, cv2.COLOR_BGR2RGB))
plt.title("binarize")
plt.xticks([]),plt.yticks([])

plt.show()