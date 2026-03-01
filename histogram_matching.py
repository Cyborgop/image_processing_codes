import numpy as np
import cv2
import cv2

# -----------------------------
# Load images (COLOR)
# -----------------------------
source = cv2.imread("/content/ChatGPT Image Feb 8, 2026, 06_21_11 PM.png")      # image to be modified
reference = cv2.imread("/content/faculty1.jpg")  # target histogram

if source is None or reference is None:
    raise ValueError("One or both images not loaded. Check paths.")

# -----------------------------
# Histogram matching for ONE channel
# -----------------------------
def match_histogram_channel(src_channel, ref_channel):

    # Step 1: Compute histograms
    src_hist = np.zeros(256)
    ref_hist = np.zeros(256)

    rows, cols = src_channel.shape
    for i in range(rows):
        for j in range(cols):
            src_hist[src_channel[i, j]] += 1

    rows, cols = ref_channel.shape
    for i in range(rows):
        for j in range(cols):
            ref_hist[ref_channel[i, j]] += 1

    # Step 2: Normalize histograms (PDF)
    src_pdf = src_hist / np.sum(src_hist)
    ref_pdf = ref_hist / np.sum(ref_hist)

    # Step 3: Compute CDFs
    src_cdf = np.cumsum(src_pdf)
    ref_cdf = np.cumsum(ref_pdf)

    # Step 4: Create intensity mapping
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        mapping[i] = np.argmin(diff)

    # Step 5: Apply mapping
    matched = np.zeros_like(src_channel, dtype=np.uint8)

    rows, cols = src_channel.shape
    for i in range(rows):
        for j in range(cols):
            matched[i, j] = mapping[src_channel[i, j]]

    return matched


# -----------------------------
# Split channels
# -----------------------------
src_b, src_g, src_r = cv2.split(source)
ref_b, ref_g, ref_r = cv2.split(reference)

# -----------------------------
# Match each channel
# -----------------------------
matched_b = match_histogram_channel(src_b, ref_b)
matched_g = match_histogram_channel(src_g, ref_g)
matched_r = match_histogram_channel(src_r, ref_r)

# -----------------------------
# Merge channels
# -----------------------------
matched_image = cv2.merge([matched_b, matched_g, matched_r])

# -----------------------------
# Display results
# -----------------------------
print("Source Image")
cv2.imshow(source)

print("Reference Image")
cv2.imshow(reference)

print("Histogram Matched Image")
cv2.imshow(matched_image)
