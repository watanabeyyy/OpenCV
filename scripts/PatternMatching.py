import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../img/4_5.bmp', cv2.IMREAD_GRAYSCALE)

im_x = 2048
im_y = 4096

sub_im_x = 32
sub_im_y = 4

num_sub_x = int(im_x / sub_im_x)
num_sub_y = int(im_y / sub_im_y)

matching = np.zeros((im_y, im_x))

normalize = sub_im_x * sub_im_y * 255

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
        matching1 = np.linalg.norm(p1 - p0)
        matching2 = np.linalg.norm(p1 - p2)
        matching[(i * sub_im_y):((i + 1) * sub_im_y), (j * sub_im_x):((j + 1) * sub_im_x)] = (
                                                                                             matching1 + matching2) / normalize

size = 2

plt.subplot(1, size, 1)
plt.imshow(img)
plt.title("original")
plt.xticks([]), plt.yticks([])
plt.subplot(1, size, 2)
plt.imshow(matching)
plt.title("matching")
plt.xticks([]), plt.yticks([])
plt.show()
