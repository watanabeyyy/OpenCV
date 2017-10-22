import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv2.imread('../img/コンマ05サンプル/22_8.bmp', cv2.IMREAD_GRAYSCALE)
    size_x = np.shape(img)[1]//8
    size_y = np.shape(img)[0]//8
    img = cv2.resize(img, (size_x, size_y))

    fftLen = size_y // 16
    win = np.hamming(fftLen)  # ハミング窓
    step = fftLen // 4

    l = size_y  # 入力信号の長さ
    M = (l - fftLen + step) // step  # スペクトログラムの時間フレーム数

    stft_map = []
    X = np.zeros([M, fftLen], dtype=complex)  # スペクトログラムの初期化(複素数型)

    for i in range(size_x):# STFT
        x = img[:,i]
        for m in range(M):
            start = step * m
            X[m, :] = np.fft.fft(x[start: start + fftLen] * win)
        stft_map = np.append(stft_map,abs(X[:,2]))#異常は2Hz成分

    stft_map = np.reshape(stft_map, (size_x, M))
    stft_map = stft_map.transpose((1, 0))
    stft_map = cv2.resize(stft_map,(size_x, size_y))

    threshold = np.mean(stft_map) + 1 * np.var(stft_map)

    stft_map = np.where(stft_map > threshold , 255,0)

    plt.subplot(1, 2, 1)
    plt.gray()
    plt.title("sample")
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.gray()
    plt.title("stft_map")
    plt.imshow(stft_map)
    plt.show()