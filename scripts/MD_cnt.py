import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def trim(img):
    img_x = np.shape(img)[1]
    for i in range(img_x):
        if (img[100, i] > 10):
            trim_l = i - 10
            break
    for i in range(img_x):
        if (img[100, img_x - (i + 1)] > 10):
            trim_r = img_x - (i + 1) + 10
            break
    img = img[:, trim_l:trim_r]

    return img

def hist_bin(img):
    #img = cv2.equalizeHist(img)
    #binarize = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 50)

    img_flatten = img.flatten()

    # 各ビンを [0, 1), [1, 2), ..., [255, 256] とすることで、各輝度値の頻度を計算する。
    # np.histogram() のリファレンス参照。
    bins = np.arange(0, 257)
    hist, bin_edges = np.histogram(img_flatten, bins)
    bin_edges = bin_edges[:-1]

    # 頻度値の積算を計算する
    cumsum = np.cumsum(hist)

    # 対象物体が画像に占める割合が 50 %であるとわかっている場合
    p = 0.1
    for i in range(0, 256):
        pcent = cumsum[i] / cumsum[-1]
        if pcent > p:
            break

    _, binarize = cv2.threshold(img, i, 255, cv2.THRESH_BINARY_INV)

    maisu = 0

    for i in range(np.shape(binarize)[1]):
        num = len(np.where(binarize[:,i] == 255)[0])
        if (num > 0.3 * np.shape(binarize)[0]):
            binarize[:,i] = 255

    return binarize

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
    #print(u"各ブロブの中心座標:\n", center)
    num = len(np.where(data[:, 3] > sub_bin_y - 10)[0])
    #print(u"谷の中心座標:\n", center[np.where(data[:, 3] > sub_bin_y - 10)])
    # print("枚数:", num-1)

    return num - 1, center[np.where(data[:, 3] > sub_bin_y - 10)][:,0]


# img_x = np.shape(img)[1]
# img_y = np.shape(img)[0]
# sub_bin_y = 128
#
# num_sub_bin = int(img_y / sub_bin_y)
# num_sub_bin = 10
# pre_cnt = np.zeros(num_sub_bin)
#
# binarize = hist_bin(img)
#
# for i in range(num_sub_bin):
#     sub_bin = binarize[i:(i + 1) * sub_bin_y - 1, :]
#     pre_cnt[i] = cnt_disk((sub_bin), sub_bin_y)
#
# print(pre_cnt)
# print(stats.mode(pre_cnt))

org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)
img = trim(org_img)
binarize = hist_bin(img)
num , point = cnt_disk(binarize, np.shape(binarize)[0])

print("谷位置:",point)
print(num,"枚")

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
