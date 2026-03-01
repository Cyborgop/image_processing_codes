import numpy as np
import cv2
from math import floor, ceil

#insert image
img= cv2.imread("Path to image", cv2.IMREAD_GRAYSCALE)
#nearest interpolation 
def nearest_interpolation(img,htscalingfactor,wdscalingfactor):
  height,width=img.shape
  newht = int(height*htscalingfactor)
  newwd = int(width*wdscalingfactor)
  newimg=np.zeros(newht,newwd)
  for i in range(newht):
    for j in range(newwd):
      org_row=int(i/htscalingfactor)
      org_col=int(j/wdscalingfactor)
      newimg[i,j]=img[org_row,org_col]
  return newimg

#bilinear interpolation
def bilinear_interpolation(htscalingfactor,wdscalingfactor,img):
  height,width = img.shape
  newht=int(height*htscalingfactor)
  newwd=int(width*wdscalingfactor)
  newimg=np.zeros_like(newht,newwd)
  for i in range(newht):
    for j in range(newwd):
      org_rows=int(i/htscalingfactor)
      org_cols=int(j/wdscalingfactor)
      r1=floor(org_rows)
      c1=floor(org_cols)
      r2=ceil(min(org_rows,height-1))
      c2=ceil(min(org_cols,width-1))
      dx = org_rows - r1
      dy = org_cols - c1
      I00 = img[r1, c1]
      I01 = img[r1, c2]
      I10 = img[r2, c1]
      I11 = img[r2, c2]
      value = (1 - dx) * ((1 - dy) * I00 + dy * I01) + dx * ((1 - dy) * I10 + dy * I11)
      newimg[i, j] = value

    return newimg
  

  cv2.imshow(img)



