import numpy as np

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


