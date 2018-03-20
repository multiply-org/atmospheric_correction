#!/usr/bin/env python


import numpy as np
from scipy.fftpack.realtransforms import dct,idct


class fastDiff(object):
  '''
  fastDiff: class for fast 2nd O differentiation of a gridded dataset


  '''

  def __init__(self,y=None,gamma=1.0,thresh=1e-10,yshape=None,smoothOrder=1.0,axis=None):
    '''
    load gridded dataset y  and yshape (in case y is flattened)

    thresh:
      method has some rounding issues so threshold as thresh

    '''
    self.gamma = gamma
    self.yshape = yshape or ((y is not None) and y.shape) or None
    self.y = y
    self.axis = axis
    self.thresh = thresh
    self.smoothOrder = smoothOrder # not used at present 
    if self.yshape is not None:   
      self.dctFilter = self.diffFilter()

  def diff(self,y=None):
    '''
    differentiate y (or loaded y)
    '''
    if (y is not None):
      assert y.size == np.prod(self.yshape)
    else:
      y = self.y.reshape(self.yshape)
 
    DTDy = self.dctND(self.dctFilter * self.dctND(y,f=dct),f=idct)
    DTDy[np.logical_and(DTDy>=-self.thresh,DTDy<=self.thresh)] = 0.
    return DTDy

  def dctND(self,data,f=dct,axis=None):
    if axis is not None:
      nd = len(data.shape)
      for i in xrange(nd):
        data = f(data,norm='ortho',type=2,axis=axis[i])
      return data
    else:
      nd = len(data.shape)
      if nd == 1:
        return f(data,norm='ortho',type=2)
      elif nd == 2:
        return f(f(data,norm='ortho',type=2).T,norm='ortho',type=2).T
      elif nd ==3:
        return f(f(f(data,norm='ortho',type=2,axis=0)\
                         ,norm='ortho',type=2,axis=1)\
                         ,norm='ortho',type=2,axis=2)
      elif nd ==4:
        return f(f(f(f(data,norm='ortho',type=2,axis=0)\
                             ,norm='ortho',type=2,axis=1)\
                           ,norm='ortho',type=2,axis=2)\
                         ,norm='ortho',type=2,axis=3)

  def cost_der_cost(self,x,mask=None):
    '''
    this is the magic function for use in optimisation
    
    You can mask the array to ignore non masked pixels
    This might need some fixing at the boundaries eg by
    buffering the mask by 1 pixel
    '''
    J_ = -self.diff(y=x)
    if mask is not None:
        J_[~mask] = 0.0
    J = 0.5 * np.dot(x.flatten(),J_.flatten())
    return self.gamma*J,self.gamma*J_

  def Heaviside(self,N,s,truncate=None):
      '''
      Step function in frequency domain
      '''
      l = np.arange(N)
      omega = np.pi*l/N
      fn = -np.sqrt(2./N)*np.sin(s*omega)/(omega)
      fn[0] = (N-s)/np.sqrt(N)
      if truncate is not None:
          fn[truncate:] = 0
      return fn
  
  def Hfilter(self,a):
        '''
         DCT-II filter for Heaviside H(a)
        '''
        yshape = self.yshape
        axis = self.axis
        ndim = len(yshape)

        # sort axis
        axis,Lambda = self.sort_axis(axis,ndim,yshape)

        for c,i in enumerate(axis):
            # create a 1 x d array (so e.g. [1,1] for a 2D case
            siz0 = np.ones((1,ndim)).astype(int)[0]
            siz0[i] = yshape[i]
            N = yshape[i]
            this = self.Heaviside(N,a).reshape(siz0)
            Lambda = Lambda + this
        self.Hfilter_ = Lambda
        return (Lambda)

  def Heaviside_prime(self,N,s,truncate=None):
        l = np.arange(N)
        omega = np.pi*l/N
        fn = -np.sqrt(2./N)*np.cos(s*omega)
        #fn[0] = (N-s)/np.sqrt(N)
        if truncate is not None:
            fn[truncate:] = 0
        return fn

  def Hfilter_prime(self,a):
        '''
         DCT-II filter for Heaviside' H'(a) (impulse at a)
        '''
        yshape = self.yshape
        axis = self.axis
        ndim = len(yshape)
        #import pdb;pdb.set_trace()
        # sort axis
        axis,Lambda = self.sort_axis(axis,ndim,yshape)

        for c,i in enumerate(axis):
            # create a 1 x d array (so e.g. [1,1] for a 2D case
            siz0 = np.ones((1,ndim)).astype(int)[0]
            siz0[i] = yshape[i]
            N = yshape[i]
            this = Heaviside_prime(N,a).reshape(siz0)
            Lambda = Lambda + this
        self.Hfilter_prime_ = Lambda
        return (Lambda)

  def sort_axis(self,axis,ndim,yshape):
        if axis is None:
            axis = tuple(np.arange(ndim))
        axis = tuple(np.array(axis).flatten())
        # initialise Lambda
        Lambda = np.zeros(yshape).astype(float)

        # normalisation factor
        naxis = []
        for i in axis:
            # correct for -ves
            if i < 0:
              i = i + len(yshape) + 1
            naxis.append(i)
        axis = tuple(naxis)
        return axis,Lambda    

  def smoothFilter(self,gamma=None):
      '''
       DCT-II filter for 1/(1 + gamma DT D)
      '''
      gamma = gamma or self.gamma
      Lambda = self.diffFilter()
      return (1./(1+gamma*Lambda))

  def diffFilter(self):
      '''
       DCT-II filter for DT D
      '''
      yshape = self.yshape
      axis = self.axis
      ndim = len(yshape)

      # sort axis
      axis,Lambda = self.sort_axis(axis,ndim,yshape)

      for c,i in enumerate(axis):
          # create a 1 x d array (so e.g. [1,1] for a 2D case
          siz0 = np.ones((1,ndim)).astype(int)[0]
          siz0[i] = yshape[i]
          omega = np.pi*np.arange(yshape[i])/float(yshape[i])
          this = np.cos(omega).reshape(siz0)
          Lambda = Lambda + this
      Lambda = -(len(axis)-Lambda)
      return (2*Lambda)

import sys

def main(argv):
  case1()
  case2()
  case3()
  case4()
  case5()

def case1():
   '''
   generate test dataset: impulse response
   '''
   import pylab as plt

   x = np.arange(100)
   y= np.zeros(100)
   y[50] = 1.0

   dtd = fastDiff(y)
   dtdy = dtd.diff()

   plt.figure(figsize=(10,3))
   plt.title('case 1: impulse response')
   plt.plot(x,y,'k-',label='signal $y$: impulse')
   plt.plot(x,dtdy,'r--',label='$D^T D y$: impulse response')
   plt.legend(loc='best')
   plt.show()
   plt.savefig('images/case1.png')

def case2():
  # case 2 image
  from PIL import Image
  import urllib2
  import pylab as plt

  url='https://upload.wikimedia.org/wikipedia/en/0/04/TCF_centre.jpg'

  im = np.array(Image.open(urllib2.urlopen(url)).convert("L")).astype(float)
  im /= im.max()
 
  dtd = fastDiff(im,axis=(0,1))  
  dtdy = dtd.diff()
  dtd1 = fastDiff(im,axis=(0,))
  dtdy1 = dtd1.diff()
  dtd2 = fastDiff(im,axis=(1,))
  dtdy2 = dtd2.diff()

  fig = plt.figure(figsize=(15,15))
  a=fig.add_subplot(231)
  a.set_title('Before')
  imgplot = plt.imshow(im,interpolation='nearest',cmap='gray')
  plt.colorbar(orientation ='horizontal')

  a=fig.add_subplot(232)
  b=plt.imshow(dtdy,interpolation='nearest',cmap='gray')
  a.set_title('After')
  plt.colorbar(orientation ='horizontal',use_gridspec=True)

  a=fig.add_subplot(233)
  plt.imshow(dtdy1,interpolation='nearest',cmap='gray')
  a.set_title('After: axis 0')
  plt.colorbar(orientation ='horizontal')

  a=fig.add_subplot(234)
  plt.imshow(dtdy2,interpolation='nearest',cmap='gray')
  a.set_title('After: axis 1')
  plt.colorbar(orientation ='horizontal')

  a=fig.add_subplot(236) 
  a.set_title('DCT Filter')
  imgplot = plt.imshow(dtd.dctFilter,interpolation='nearest',cmap='gray')
  plt.colorbar(orientation ='horizontal')
  plt.show()
  plt.savefig('images/case2.png')

def case3():
  # case 2 image in colour
  from PIL import Image
  import urllib2
  import pylab as plt

  url='https://upload.wikimedia.org/wikipedia/en/0/04/TCF_centre.jpg'

  im = np.array(Image.open(urllib2.urlopen(url))).astype(np.short)
  im /= im.max()

  dtd = fastDiff(im,axis=(0,1))
  dtdy = dtd.diff()
  dtd1 = fastDiff(im,axis=(0,))
  dtdy1 = dtd1.diff()
  dtd2 = fastDiff(im,axis=(1,))
  dtdy2 = dtd2.diff()

  fig = plt.figure(figsize=(15,15))
  a=fig.add_subplot(231)
  a.set_title('Before')
  imgplot = plt.imshow(im,interpolation='nearest',cmap='gray')

  a=fig.add_subplot(232)
  f = dtdy
  b=plt.imshow((255*f/f.max()).astype(np.short),interpolation='nearest',cmap='gray')
  a.set_title('After')

  a=fig.add_subplot(233)
  f = dtdy1
  plt.imshow((255*f/f.max()).astype(np.short),interpolation='nearest',cmap='gray')
  a.set_title('After: axis 0')

  a=fig.add_subplot(234)
  f = dtdy2
  plt.imshow((255*f/f.max()).astype(np.short),interpolation='nearest',cmap='gray')
  a.set_title('After: axis 1')

  a=fig.add_subplot(236)
  a.set_title('negative DCT Filter')
  f = ((-(dtd.dctFilter)))
  f = (255*f/f.max()).astype(np.short)
  imgplot = plt.imshow(f,interpolation='nearest')
  plt.show()
  plt.savefig('images/case3.png')

def case4():
  '''
  Defining J =  x^T D^T D x
  so       J' = D^T D x
 
  we use the class to generate cost function and cost function derivatives
  for a differential operator
  '''
  from PIL import Image
  import urllib2
  import pylab as plt

  url='https://upload.wikimedia.org/wikipedia/en/0/04/TCF_centre.jpg'

  # 1D dataset
  im = np.array(Image.open(urllib2.urlopen(url)).convert("L")).astype(float)[50]
  im /= im.max()

  x = im
 
  # so J' = D^T D x
 
  # calculate Dx explicitly
  n = x.shape[0]
  D = np.eye(n) - np.diag(np.ones(n),-1)[:n,:n]
  D[0,0] = 0
  D = np.matrix(D)
  '''
  so D.T * D is 
     matrix([[ 1., -1.,  0., ...,  0.,  0.,  0.],
             [-1.,  2., -1., ...,  0.,  0.,  0.],
             [ 0., -1.,  2., ...,  0.,  0.,  0.],
             ..., 
             [ 0.,  0.,  0., ...,  2., -1.,  0.],
             [ 0.,  0.,  0., ..., -1.,  2., -1.],
             [ 0.,  0.,  0., ...,  0., -1.,  1.]])


  which is the same as our DCT filter
  '''
 
  dtd = fastDiff(x,axis=(0,))
  J_ = -dtd.diff()
  J_slow = np.array((D.T * D) * np.matrix(x).T).flatten()
  J = 0.5 * np.dot(x,J_)
  Jslow = np.array(0.5 * np.matrix(x) * (D.T * D) * np.matrix(x).T)[0,0]

  plt.figure(figsize=(10,4))
  plt.title('J_d = %.4f (DCT) = %.4f (explicit)'%(J,Jslow))
  plt.plot(x,'k-',label='x')
  plt.plot(J_,'r-',label="J'(x) (DCT)")
  plt.plot(J_slow,'g--',label="J'(x) (explicit)")

  plt.legend(loc='best')
  plt.show()
  plt.savefig('images/case4.png')

def cost_identity(x,xobs):
  J_ = x - xobs
  J =  0.5 * np.dot(J_,J_)
  return J,J_

def fun_dfun(x,xobs,diff):
  J,J_ = cost_identity(x,xobs)
  J1,J1_ = diff.cost_der_cost(x)
  return J + J1,J_ + J1_

def case5():
  '''
  Defining J =  x^T D^T D x
  so       J' = D^T D x

  we use the class to generate cost function and cost function derivatives
  for a differential operator
  '''
  from PIL import Image
  import urllib2
  import pylab as plt

  url='https://upload.wikimedia.org/wikipedia/en/0/04/TCF_centre.jpg'

  # 1D dataset
  im = np.array(Image.open(urllib2.urlopen(url)).convert("L")).astype(float)[50]
  im /= im.max()

  x = im


  '''
  so we have a dataset x

  and constraints:


  D x = 0 | gamma
  x = xobs

  from which we get:

  J = (1/2) x^T D^T D x + (1/2) (x - xobs)^2

  J' = D^T D x + (x - xobs)

  J'' = D^T D + I

  Here, we demonstrate how to use this class to form
  the call needed in minimize() for the differential operator

  1. define gamma
  2. set up the operator
  3. then J,J_ = diff.cost_der_cost(x)
 
  '''
  gamma = 10.

  # set up the differential operator, with an example of x
  # at this point
  diff = fastDiff(x,axis=(0,),gamma=gamma)
  # test the D cost function
  J,J_ = diff.cost_der_cost(x)

  # test combined 
  J,J_ = fun_dfun(x,x,diff) 

  from scipy.optimize import minimize

  res = minimize(fun_dfun, x, jac=True, method='Newton-CG',\
                options={'disp': True},\
                args=(x,diff))

  plt.figure(figsize=(10,4))
  plt.title('cost = %.3f for gamma = %.2f'%(res.fun,gamma))
  plt.plot(x,'k-',label='x')
  plt.plot(res.x,'r-',label='optimised x')
  plt.legend(loc='best')
  plt.show()
  plt.savefig('images/case5.png')




  
if __name__ == "__main__":
  main(sys.argv)
