import numpy as np



def fft1d(X):
  N=len(X)
  if(N<=1):
    return X
  evenx=fft1d(X[0::2])
  oddx=fft1d(X[1::2])
  factor=np.exp((-2j*np.pi*np.arange(N))/N)
  final=np.concatenate((evenx+(factor[:N//2]*oddx)),(evenx+(factor[N//2:]*oddx)))
  return final

def ifft(X):
  N=len(X)
  if(N<=1):
    return X
  evenx=ifft(X[0::2])
  oddx=ifft(X[1::2])
  factor=np.exp((2j*np.pi*np.arange(N))/N)
  final=np.concatenate((evenx+(factor[:N//2]*oddx)),(evenx+(factor[N//2:]*oddx))/N)
  return final

def fft2d(img):
  fft_rows=np.array([fft1d(rows) for rows in img])
  fft_2d=np.array([fft1d(rows) for rows in fft_rows.T]).T
  return fft_2d

def ifft2d(img):
  ifft_rows=np.array([ifft(rows) for rows in img])
  ifft_2d=np.array([ifft(rows) for rows in ifft_rows.T]).T
  return ifft_2d

def apply_tx(img):
  rows,cols=img[:2]
  x,y=np.meshgrid(np.arange(cols),np.arange(rows))
  transformed=(-1)**(x+y)
  return transformed*img

