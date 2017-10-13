import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv2.imread('../img/2_00_5879.jpg', cv2.IMREAD_GRAYSCALE)
    binarize = np.zeros(np.shape(img))
    c_num = np.shape(img)[1]
    for i in range(c_num):
        if (i == 0):
            sub_im = img[:, i:i + 3]
        elif (i == c_num - 1):
            sub_im = img[:, i - 2:i + 1]
        else:
            sub_im = img[:, i - 1:i + 2]
        num = len(np.where(sub_im < 50)[0])
        print(num)
        if (num > 0.4 * 3 * np.shape(img)[0]):
            binarize[:, i] = 0
        else:
            binarize[:, i] = 255

    _, binarize = cv2.threshold(binarize, 127, 255, cv2.THRESH_BINARY_INV)

    size = 2

    plt.subplot(1, size, 1)
    plt.imshow(img)
    plt.title("original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, size, 2)
    plt.imshow(binarize)
    plt.title("binalize")
    plt.xticks([]), plt.yticks([])

    plt.show()
