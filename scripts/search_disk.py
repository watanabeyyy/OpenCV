import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img = cv2.imread('../img/コンマ05サンプル/22_8.bmp', cv2.IMREAD_GRAYSCALE)

    img = img[::400]
    size_x = np.shape(img)[1]
    size_y = np.shape(img)[0]

    map = np.zeros((size_y,size_x))

    win_size = 128
    for i in range(size_y):
        data = img[i, :]
        for j in range(size_x):
            if(j<win_size):
                map[i,j]=data[j]
            else:
                map[i,j] = np.sum(data[j-win_size:j])/win_size

    map = np.square(map - img)

    print(np.mean(img))

    plt.plot(map[0])
    plt.plot(img[0])
    plt.show()