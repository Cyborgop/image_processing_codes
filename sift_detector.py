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
    gauss= np.exp**((ax**2)/(2*np.pi*sigma**2))
    gauss_outer=np.outer(gauss,gauss)
    return conv2d(image,gauss_outer)





