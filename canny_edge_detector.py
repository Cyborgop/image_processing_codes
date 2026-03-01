import numpy as np
import cv2
import cv2

# ---------- Load Grayscale Image ----------
image = cv2.imread(
    "/content/ChatGPT Image Feb 8, 2026, 06_21_11 PM.png",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError("Image not loaded")

cv2.imshow(image)


# ---------- 2D Convolution ----------
def conv2D(kernel, img):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    H, W = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')

    out = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)

    return out


# ---------- Gaussian Blur ----------
def gaussian_blur(image, kernel_size=7, sigma=1.5):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)
    return conv2D(kernel, image)


# ---------- Sobel Gradients ----------
def sobel_gradients(image):
    sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    Gx = conv2D(sobel_x, image)
    Gy = conv2D(sobel_y, image)

    return Gx, Gy


# ---------- Gradient Magnitude & Direction ----------
def gradient_mag_dir(Gx, Gy):
    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(Gy, Gx) * 180 / np.pi
    direction[direction < 0] += 180
    return magnitude, direction


# ---------- Non-Maximum Suppression ----------
def non_maximum_suppression(G, theta):
    H, W = G.shape
    nms = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = theta[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif (22.5 <= angle < 67.5):
                q, r = G[i-1, j+1], G[i+1, j-1]
            elif (67.5 <= angle < 112.5):
                q, r = G[i-1, j], G[i+1, j]
            else:
                q, r = G[i-1, j-1], G[i+1, j+1]

            if G[i, j] >= q and G[i, j] >= r:
                nms[i, j] = G[i, j]

    return nms


# ---------- DOUBLE THRESHOLDING ----------
def double_threshold(G):
    T_high = 0.25 * G.max()
    T_low  = 0.5 * T_high

    strong, weak = 255, 75
    edges = np.zeros(G.shape, dtype=np.uint8)

    strong_i, strong_j = np.where(G >= T_high)
    weak_i, weak_j = np.where((G >= T_low) & (G < T_high))

    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak

    return edges, weak, strong


# ---------- ITERATIVE HYSTERESIS (IMPORTANT FIX) ----------
def hysteresis(edges, weak, strong):
    H, W = edges.shape
    changed = True

    while changed:
        changed = False
        for i in range(1, H-1):
            for j in range(1, W-1):
                if edges[i, j] == weak:
                    if np.any(edges[i-1:i+2, j-1:j+2] == strong):
                        edges[i, j] = strong
                        changed = True

    edges[edges != strong] = 0
    return edges


# ---------- RUN FULL PIPELINE ----------
blur = gaussian_blur(image)
Gx, Gy = sobel_gradients(blur)
G, theta = gradient_mag_dir(Gx, Gy)
nms = non_maximum_suppression(G, theta)
thresh, weak, strong = double_threshold(nms)
final_edges = hysteresis(thresh, weak, strong)


# ---------- DISPLAY RESULTS ----------
print("Non-Maximum Suppression")
cv2.imshow(nms.astype(np.uint8))

print("After Double Thresholding")
cv2.imshow(thresh)

print("Final Edges After Hysteresis")
cv2.imshow(final_edges)
