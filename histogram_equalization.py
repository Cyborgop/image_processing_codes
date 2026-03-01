import numpy as np
import cv2

# Read grayscale image
image = cv2.imread("input image", cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape

# -----------------------------
# Step 1: Histogram
# -----------------------------
hist = np.zeros(256)

for i in range(rows):
    for j in range(cols):
        hist[image[i, j]] += 1

# -----------------------------
# Step 2: PDF (normalize)
# -----------------------------
pdf = hist / np.sum(hist)

# -----------------------------
# Step 3: CDF using cumsum
# -----------------------------
cdf = np.cumsum(pdf)

# -----------------------------
# Step 4: Create mapping
# -----------------------------
mapping = np.round(cdf * 255).astype(np.uint8)

# -----------------------------
# Step 5: Apply mapping
# -----------------------------
equalized = np.zeros_like(image, dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        equalized[i, j] = mapping[image[i, j]]

# Display
cv2.imshow("Equalized Image", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()