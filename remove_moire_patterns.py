import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("moire_image.jpg", 0)  
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = rows//2 , cols//2
mask = np.ones((rows, cols), np.uint8)
r = 10
mask[crow-30-r:crow-30+r, ccol-30-r:ccol-30+r] = 0
mask[crow+30-r:crow+30+r, ccol+30-r:ccol+30+r] = 0
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title("Moire Reduced")
plt.show()