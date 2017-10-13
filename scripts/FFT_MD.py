import cv2
import numpy as np
from matplotlib import pyplot as plt
import MD_cnt


if __name__ == "__main__":
    org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)

    dat = org_img[0,:]

    X = np.fft.fft(dat)  # FFT

    #freqList = np.fft.fftfreq(N, d=1.0 / fs)  # 周波数軸の値を計算

    amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル
    phaseSpectrum = [np.arctan2(int(c.imag), int(c.real)) for c in X]  # 位相スペクトル

    # 波形を描画
    plt.subplot(311)  # 3行1列のグラフの1番目の位置にプロット
    plt.plot(dat)
    plt.xlabel("time [sample]")
    plt.ylabel("amplitude")

    # 振幅スペクトルを描画
    plt.subplot(312)
    plt.plot(amplitudeSpectrum, marker='o', linestyle='-')
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")
    plt.show()