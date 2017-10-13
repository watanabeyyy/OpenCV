import cv2
import numpy as np
from matplotlib import pyplot as plt
import MD_cnt


def match(img, temp, num, point):
    match_rate = np.zeros(num - 1)
    for j in range(num - 1):
        left = int(point[j])
        right = int(point[j + 1]) + 1
        p = img[:, left:right]
        p = p - np.mean(p)
        temp_resized = cv2.resize(temp, np.shape(p)[::-1])
        temp_resized = temp_resized - np.mean(temp_resized)
        norm = np.linalg.norm(temp_resized) * np.linalg.norm(p)
        zncc = np.sum(temp_resized * p)
        if norm == 0:
            zncc = 0
        else:
            zncc = zncc / norm
        match_rate[j] = zncc
    return match_rate


def check_orientation(img, num, point):
    left = int(point[0])
    right = int(point[1]) + 1
    temp_a = img[:, left:right]
    temp_b = temp_a[:, ::-1]

    rate_a = match(img, temp_a, num, point)
    rate_b = match(img, temp_b, num, point)
    ori = np.zeros(num - 1)
    bool_a = True
    if (rate_a[0] > rate_b[0]):
        bool_a = True
    else:
        bool_a = False
    for i in range(num - 1):
        if (rate_a[i] >= rate_b[i] and bool_a == True):
            ori[i] = 0
        elif (rate_a[i] <= rate_b[i] and bool_a == False):
            ori[i] = 0
        else:
            ori[i] = 1
        bool_a = not (bool_a)
    print(ori)


org_img = cv2.imread('../img/8_8.bmp', cv2.IMREAD_GRAYSCALE)
# トリミング⇒枚数カウント
img = MD_cnt.trim(org_img)
bin = MD_cnt.make_bin(img, th_bri=60, th_num=0.8)
num, point = MD_cnt.cnt_disk(bin, np.shape(bin)[0])
print(num - 1, "枚")

# 方向チェック
check_orientation(img, num, point)

# 欠陥抽出
im_x = np.shape(img)[1]
im_y = 4096

sub_im_y = 64

num_sub_y = int(im_y / sub_im_y)

matching = np.zeros((im_y, im_x))

for j in range(num - 1):
    left = int(point[j])
    right = int(point[j + 1]) + 1
    for i in range(num_sub_y):
        p1 = img[(i * sub_im_y):((i + 1) * sub_im_y), left:right]
        if i == 0:
            p0 = img[((i + 1) * sub_im_y):((i + 2) * sub_im_y), left:right]
        else:
            p0 = img[((i - 1) * sub_im_y):(i * sub_im_y), left:right]
        p0 = p0 - np.mean(p0)
        p1 = p1 - np.mean(p1)
        if (np.shape(p0) != np.shape(p1)):
            break
        norm = np.linalg.norm(p0) * np.linalg.norm(p1)
        zncc = np.sum(p0 * p1)
        if norm == 0:
            zncc = 0
        else:
            zncc = zncc / norm + 1
        matching[(i * sub_im_y):((i + 1) * sub_im_y), left:right] = zncc * 127

ret, binalize = cv2.threshold(matching, 200, 255, cv2.THRESH_BINARY)

# 可視化
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
