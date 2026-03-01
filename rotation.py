import numpy as np
from math import radians,cos,sin
def rotation(img,ang):
  height,width=img.shape
  radangle=radians(ang)
  newht=round(abs(height*cos(radangle)))+round(abs(width*sin(radangle)))
  newwd=round(abs(height*sin(radangle)))+round(abs(width*cos(radangle)))
  newimg=np.zeros(newht,newwd)
  ohtc=height//2
  owdc=width//2
  newhtc=newht//2
  newwdc=newwd//2
  for i in range(newht):
    for j in range(newwd):
       rows=(i-newhtc)*cos(radangle)+(j-newwdc)*sin(radangle)
       cols=-(i-newhtc)*sin(radangle)+(j-newwdc)*cos(radangle)
       rows= round(rows)+ohtc
       cols= round(cols)+owdc
       newimg[i,j]=img[rows,cols]
  return newimg