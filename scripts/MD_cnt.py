import cv2
import numpy as np
from matplotlib import pyplot as plt


def trim(img):
    img_x = np.shape(img)[1]
    for i in range(img_x):
        if (img[100, i] > 10):
            trim_l = i - 3
            break
    for i in range(img_x):
        if (img[100, img_x - (i + 1)] > 10):
            trim_r = img_x - (i + 1) + 2
            break
    img = img[:, trim_l:trim_r]

    return img


def make_bin(img, th_bri=50, th_num=0.4):
    binarize = np.zeros(np.shape(img)[1])
    c_num = np.shape(img)[1]
    for i in range(c_num):
        sub_im = img[:, i]
        num = len(np.where(sub_im < th_bri)[0])
        if (num > th_num * np.shape(img)[0]):
            binarize[i] = 0
        else:
            binarize[i] = 255
    return binarize


def cnt_disk(sub_bin, sub_bin_y):
    black = False
    pos = []
    for i in range(sub_bin_y):
        if (i == 0):
            if (sub_bin[i] == 0):
                black = True
                str = i
            else:
                black = False
        elif (i == sub_bin_y - 1):
            if (black):
                end = i - 1
                pos.append((str, end))
        else:
            if (not (black) and sub_bin[i] == 0):
                black = True
                str = i
            elif (black and sub_bin[i] == 255):
                black = False
                end = i - 1
                pos.append((str, end))
    num = len(pos)

    point = np.zeros(num)

    for i in range(num):
        point[i] = round((pos[i][0] + pos[i][1]) / 2)

    return num, point


if __name__ == "__main__":
    org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)
    img = trim(org_img)
    bin = make_bin(img)
    num, point = cnt_disk(bin, np.shape(bin)[0])

    print("谷位置:", point)
    print(num - 1, "枚")

    size = 2

    plt.subplot(1, size, 1)
    plt.imshow(org_img)
    plt.title("original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, size, 2)
    plt.imshow(img)
    plt.title("trim->hist_equalize")
    plt.xticks([]), plt.yticks([])

    plt.show()
