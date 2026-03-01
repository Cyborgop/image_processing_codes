import numpy as np
import cv2
from scipy.signal import convolve2d
import cv2

# Read image in grayscale
image = cv2.imread("input image file", cv2.IMREAD_GRAYSCALE)

# Gaussian blur
def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    gauss_kernel = np.outer(gauss, gauss)
    gauss_kernel /= np.sum(gauss_kernel)

    return convolve2d(image, gauss_kernel, mode='same', boundary='symm')

# Adaptive thresholding with odd window
def odd_window(image, kernel, C=7):
    img = apply_gaussian_blur(image)

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    height, width = img.shape

    pad_img = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='reflect'
    )

    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            window = pad_img[i:i+kh, j:j+kw]
            local_mean = np.mean(window)

            if img[i, j] > (local_mean - C):
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary

# Define odd window kernel (only size matters)
kernel = np.ones((11, 11))

# Apply adaptive thresholding
binary = odd_window(image, kernel, C=5)

# Display result
cv2.imshow(binary)


