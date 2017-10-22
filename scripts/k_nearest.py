import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img = cv2.imread('../img/コンマ05サンプル/22_8.bmp', cv2.IMREAD_GRAYSCALE)
    size_x = np.shape(img)[1]//8
    size_y = np.shape(img)[0]//8
    img = cv2.resize(img, (size_x, size_y))

    map = np.zeros((size_y,size_x))

    win_size = 32
    k = 10
    for i in range(size_x):
        data = img[:, i]
        print(i)
        for j in range(size_y):
            if(j<win_size):
                map[j,i] = 0
            else:
                train = data[j-win_size:j]
                test = train - data[j]
                anomaly_degree = np.sort(test)
                #print(anomaly_degree)
                anomaly_degree = anomaly_degree[::-1][:k]
                #print(anomaly_degree)
                map[j,i] = np.mean(np.square(anomaly_degree))
    ### Plot
    map = map / np.max(map) * 255
    map = np.uint8(map)
    ret, map = cv2.threshold(map, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.subplot(1, 2, 1)
    plt.gray()
    plt.title("sample")
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.gray()
    plt.title("anomaly_map")
    plt.imshow(map)
    plt.show()
