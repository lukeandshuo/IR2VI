from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
def filtering(img):
    img = np.array(img, dtype=float)
    import time
    def plot(data, title):
        plot.i += 1
        plt.subplot(2,2,plot.i)
        plt.imshow(data)
        plt.gray()
        plt.title(title)
    plot.i = 0


    plot(img,'original')
    t_s = time.clock()
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])
    highpass_3x3 = ndimage.convolve(img, kernel)
    t_e = time.clock()
    print("time:",t_e-t_s)
    plot(highpass_3x3, 'Simple 3x3 Highpass')

    t_s = time.clock()
    lowpass = ndimage.gaussian_filter(img, 3)
    gauss_highpass = img - lowpass

    plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')


    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)

    plot(mag, r'sobel')
    plt.show()

if __name__ == '__main__':
    image = cv2.imread('./imgs/sensiac_V.png',0)
    filtering(image)
