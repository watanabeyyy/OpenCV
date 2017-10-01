import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)

im_x = 2048
im_y = 4096

sub_im_x = 32
sub_im_y = 32

num_sub_x = int(im_x / sub_im_x)
num_sub_y = int(im_y / sub_im_y)

matching = np.zeros((im_y, im_x))

for j in range(num_sub_x):
    for i in range(num_sub_y):
        p1 = img[(i * sub_im_y):((i + 1) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
        if i == 0:
            p0 = img[((i + 1) * sub_im_y):((i + 2) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
            p2 = img[((i + 2) * sub_im_y):((i + 3) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
        elif i == num_sub_y - 1:
            p0 = img[((i - 2) * sub_im_y):((i - 1) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
            p2 = img[((i - 1) * sub_im_y):(i * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
        else:
            p0 = img[((i - 1) * sub_im_y):(i * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
            p2 = img[((i + 1) * sub_im_y):((i + 2) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)]
        p0 = p0 - np.mean(p0)
        p1 = p1 - np.mean(p1)
        p2 = p2 - np.mean(p2)
        zncc = np.sum(p0 * p1) / np.linalg.norm(p0) / np.linalg.norm(p1)
        zncc = zncc + np.sum(p1 * p2) / np.linalg.norm(p1) / np.linalg.norm(p2)
        zncc = zncc / 2
        matching[(i * sub_im_y):((i + 1) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)] = zncc * 255

ret, binalize = cv2.threshold(matching, 150, 255, cv2.THRESH_BINARY_INV)

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
