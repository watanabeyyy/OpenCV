import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sig

if __name__ == "__main__":
    org_img = cv2.imread('../img/56_6.bmp', cv2.IMREAD_GRAYSCALE)

    dat = org_img[0,:]

    z = org_img[0,:]
    y = np.linspace(1, 2048, 2048)

    z2 = sig.cwt(z, sig.ricker, y)

    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.text(0, 0, "CWT", fontsize="large")
    ax1draw = ax1.matshow(z2, cmap=plt.cm.gray)
    plt.colorbar(ax1draw)

    ax2.plot(z)
    ax2.text(0, 0, "Input data of CWT")

    plt.show()