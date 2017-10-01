import cv2
import numpy as np
from matplotlib import pyplot as plt
import MD_cnt

org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)
img = MD_cnt.trim(org_img)
binarize = MD_cnt.hist_bin(img)
num, point = MD_cnt.cnt_disk(binarize, np.shape(binarize)[0])
print(num-1,"枚")

im_x = np.shape(img)[1]
im_y = 4096

sub_im_y = 64

num_sub_y = int(im_y / sub_im_y)

matching = np.zeros((im_y, im_x))

for j in range(num-1):
    left = int(point[j])
    right = int(point[j+1])+1
    for i in range(num_sub_y):
        p1 = img[(i * sub_im_y):((i + 1) * sub_im_y), left:right]
        if i == 0:
            p0 = img[((i + 1) * sub_im_y):((i + 2) * sub_im_y), left:right]
        else:
            p0 = img[((i - 1) * sub_im_y):(i * sub_im_y), left:right]
        p0 = p0 - np.mean(p0)
        p1 = p1 - np.mean(p1)
        norm = np.linalg.norm(p0) * np.linalg.norm(p1)
        zncc = np.sum(p0 * p1)
        if norm == 0:
            zncc = 0
        else:
            zncc = zncc / norm + 1
        matching[(i * sub_im_y):((i + 1) * sub_im_y), left:right] = zncc * 127

ret, binalize = cv2.threshold(matching, 200, 255, cv2.THRESH_BINARY)

size = 3

plt.subplot(1, size, 1)
plt.imshow(img)
plt.title("original")
plt.xticks([]), plt.yticks([])
plt.subplot(1, size, 2)
plt.imshow(matching)
plt.title("matching")
plt.xticks([]), plt.yticks([])
plt.subplot(1, size, 3)
plt.imshow(binalize)
plt.title("binalize")
plt.xticks([]), plt.yticks([])
plt.show()
