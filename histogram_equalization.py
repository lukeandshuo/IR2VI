import cv2
import numpy as np
import matplotlib.pyplot as plt
import filtering
img1 = cv2.imread('./imgs/sensiac_IR.png',0)
#
# mean = np.mean(img1)
# std = np.std(img1)
# print(mean,std)
# norm = (img1-mean)/std
# plt.imshow(norm)
# plt.show()

# equ = cv2.equalizeHist(img1)

# filtering.filtering(equ)
# hist,bins = np.histogram(equ.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf*hist.max()/cdf.max()
# plt.plot(cdf_normalized,color='b')
# plt.hist(equ.flatten(),256,[0,256],color='r')
# plt.xlim([0,256])
# plt.show()

#
# img2 = cv2.imread('./imgs/sensiac_IR2.png',0)
#
#
# cv2.imshow("test",equ)
# cv2.waitKey(0)