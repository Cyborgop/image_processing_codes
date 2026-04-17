import cv2
import numpy as np

#input image
img = cv2.imread("Path to image",cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
img = img/255

#2d Convolution
def conv2d(img, kernel):
   height,width = img.shape
   h,w = kernel.shape
   hh=h//2
   wh=w//2
   padded=np.pad(img,((hh,hh),(wh,wh)),mode='constant')
   out = np.zeros_like(img, dtype=float)
   for i in range(height):
    for j in range(width):
      imgkernel=padded[i:i+h,j:j+w]
      out[i,j]=np.sum(imgkernel*kernel)
   return out

def build_gaussian_scale(image, octaves=4, scales=3):
    sigma = 1.6
    k = 2 ** (1 / scales)
    img_per_octave = scales + 3

    gaussian_pyramid = []
    base = image.copy()

    for _ in range(octaves):
        octave_images = []
        curr = base.copy()

        for s in range(img_per_octave):
            sigma_s = sigma * (k ** s)
            blurred = gaussian_blur(curr, sigma_s, 3)
            octave_images.append(blurred)

        gaussian_pyramid.append(octave_images)

        base = octave_images[scales][::2, ::2]

    return gaussian_pyramid

def DOG_pyramid(image):
    dog_pyramid = []
    for octave_image in build_gaussian_scale(image, 4, 3):
        dog_octave = []
        n = 0
        while n < len(octave_image) - 1:
            dog_octave.append(octave_image[n+1] - octave_image[n])
            n = n + 1
        dog_pyramid.append(dog_octave)
    return dog_pyramid
       

            
#create gaussian blur
def gaussian_blur(image,sigma,kernel):
    ax=np.linspace(-(kernel//2),(kernel//2),kernel)
    gauss = np.exp(-(ax**2)/(2*sigma**2))
    gauss_outer=np.outer(gauss,gauss)
    return conv2d(image,gauss_outer)

def maxima_minima(image):
    dog = DOG_pyramid(image)
    keypoints = []

    for o, octave in enumerate(dog):

        for s in range(1, len(octave)-1):

            prev_img = octave[s-1]
            curr_img = octave[s]
            next_img = octave[s+1]

            h, w = curr_img.shape

            for x in range(1, h-1):
                for y in range(1, w-1):

                    val = curr_img[x, y]

                    neighbors = []

                    neighbors.extend(prev_img[x-1:x+2, y-1:y+2].flatten())
                    neighbors.extend(curr_img[x-1:x+2, y-1:y+2].flatten())
                    neighbors.extend(next_img[x-1:x+2, y-1:y+2].flatten())

                    neighbors.remove(val)

                    if val > max(neighbors) or val < min(neighbors):
                        keypoints.append((o, s, x, y))

    return keypoints




def keypoint_localization(image, contrast_threshold=0.03, r=10):

    dog = DOG_pyramid(image)
    candidates = maxima_minima(image)
    refined_keypoints = []

    edge_threshold = ((r+1)**2)/r

    for o,s,x,y in candidates:

        dog_img = dog[o][s]
        val = dog_img[x,y]

        # contrast filtering
        if abs(val) < contrast_threshold:
            continue

        # second derivatives
        Dxx = dog_img[x+1,y] + dog_img[x-1,y] - 2*dog_img[x,y]
        Dyy = dog_img[x,y+1] + dog_img[x,y-1] - 2*dog_img[x,y]

        Dxy = (dog_img[x+1,y+1] - dog_img[x+1,y-1] -
               dog_img[x-1,y+1] + dog_img[x-1,y-1]) / 4

        trace = Dxx + Dyy
        det = Dxx*Dyy - (Dxy**2)

        if det <= 0:
            continue

        R = (trace**2)/det

        if R < edge_threshold:
            refined_keypoints.append((o,s,x,y))

    return refined_keypoints



def assign_orientation(image):

    gaussian = build_gaussian_scale(image,4,3)
    keypoints = keypoint_localization(image)

    final_keypoints = []

    for o,s,x,y in keypoints:

        img = gaussian[o][s]

        gx = img[x,y+1] - img[x,y-1]
        gy = img[x+1,y] - img[x-1,y]

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy,gx)

        final_keypoints.append((o,s,x,y,orientation,magnitude))

    return final_keypoints


# run full detector
sift_keypoints = assign_orientation(img)

print("Total keypoints:", len(sift_keypoints))

print(sift_keypoints[:10])
# convert image back to uint8 for drawing
img_vis = (img * 255).astype(np.uint8)

# convert to color so we can draw colored circles
img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

# draw keypoints
for o, s, x, y, orientation, magnitude in sift_keypoints:
    cv2.circle(img_vis, (y, x), 2, (0,255,0), 1)

# show result
cv2.imshow("SIFT Keypoints", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()



