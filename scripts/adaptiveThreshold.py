import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)

org_img_x = np.shape(org_img)[1]

for i in range(org_img_x):
    if (org_img[100, i] > 10):
        trim_l = i - 5
        break
for i in range(org_img_x):
    if (org_img[100, org_img_x - (i + 1)] > 10):
        trim_r = org_img_x - (i + 1) + 5
        break
img = org_img[:, trim_l:trim_r]

img = cv2.equalizeHist(img)
binarize = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 50)


def cnt_disk(sub_bin, sub_bin_y):
    label = cv2.connectedComponentsWithStats(sub_bin)
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    # print(u"ブロブの個数:", n)
    # print(u"各ブロブの外接矩形の左上x座標", data[:, 0])
    # print(u"各ブロブの外接矩形の左上y座標", data[:, 1])
    # print(u"各ブロブの外接矩形の幅", data[:, 2])
    # print(u"各ブロブの外接矩形の高さ", data[:, 3])
    # print(u"各ブロブの面積", data[:, 4])
    # print(u"各ブロブの中心座標:\n", center)
    num = len(np.where(data[:, 3] > sub_bin_y - 10)[0])
    # print("枚数:", num-1)
    return num - 1


img_x = np.shape(img)[1]
img_y = np.shape(img)[0]
sub_bin_y = 256

num_sub_bin = int(img_y / sub_bin_y)
pre_cnt = np.zeros(num_sub_bin)

for i in range(num_sub_bin):
    sub_bin = binarize[i:(i + 1) * sub_bin_y - 1, :]
    pre_cnt[i] = cnt_disk(sub_bin, sub_bin_y)

print(stats.mode(pre_cnt))

size = 3

plt.subplot(1, size, 1)
plt.imshow(org_img)
plt.title("original")
plt.xticks([]), plt.yticks([])

plt.subplot(1, size, 2)
plt.imshow(img)
plt.title("trim->hist_equalize")
plt.xticks([]), plt.yticks([])

plt.subplot(1, size, 3)
plt.imshow(binarize)
plt.title("binalize")
plt.xticks([]), plt.yticks([])

plt.show()
