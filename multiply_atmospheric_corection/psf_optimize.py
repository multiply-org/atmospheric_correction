#/usr/bin/env python 
import numpy as np
import sys
sys.path.insert(0, 'python')
from scipy import ndimage, signal, optimize
from multi_process import parmap
from spatial_mapping import cloud_dilation
from create_training_set import create_training_set
import scipy

class psf_optimize(object):
    def __init__(self, 
		 high_img,
		 high_indexs,
		 low_img,
                 qa,
                 cloud,
                 qa_thresh):
       self.high_img    = high_img
       self.Hx, self.Hy = high_indexs
       self.low_img     = low_img
       self.cloud       = cloud
       self.qa_thresh   = qa_thresh
       self.qa          = qa
       self.shape       = self.high_img.shape
       self.parameters  = ['xstd', 'ystd', 'angle', 'xs', 'ys']
       self.slop        = 0.95607605898444503
       self.off         = 0.0086119174434039214
    def _preprocess(self,):
     
        size = 2*int(round(1.96*39))# set the largest possible PSF size
        self.high_img[0,:]=self.high_img[-1,:]=self.high_img[:,0]=self.high_img[:,-1]= -9999
        self.bad_pixs = cloud_dilation( (self.high_img <= 0) | self.cloud  | (self.high_img >= 1), iteration=size/2)

        xstd, ystd = 29.75, 39
        ker = self.gaussian(xstd, ystd, 0)
        self.conved = signal.fftconvolve(self.high_img, ker, mode='same')

        l_mask = (~self.low_img.mask) & (self.qa<=self.qa_thresh)
        h_mask =  ~self.bad_pixs[self.Hx, self.Hy]
        self.lh_mask = l_mask & h_mask

    def gaussian(self, xstd, ystd, angle, norm = True):
        win = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        winx = int(round(win*(2**0.5)))
        winy = int(round(win*(2**0.5)))
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[center[0]-win/2: center[0]+win/2, center[1]-win/2:center[1]+win/2]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus 


    def gaus_optimize(self, p0):
        return optimize.fmin_l_bfgs_b(self.gaus_cost, p0, approx_grad=1, iprint=-1,
                                      bounds=self.bounds,maxiter=10, maxfun=10)         


    def shift_optimize(self, p0):
        return optimize.fmin(self.shift_cost, p0, full_output=1, maxiter=100, maxfun=150, disp=0)


    def gaus_cost(self, para):
        # cost for a final psf optimization
        xstd,ystd,angle, xs, ys = para 
        ker = self.gaussian(xstd,ystd,angle,True)                              
        conved = signal.fftconvolve(self.high_img, ker, mode='same')
        # mask bad pixels
        cos = self.cost(xs=xs, ys=ys, conved=conved)
        return cos


    def shift_cost(self, shifts):
        # cost with different shits
        xs, ys = shifts
        cos = self.cost(xs=xs, ys=ys, conved=self.conved)
        return cos


    def cost(self, xs=None, ys=None, conved = None):
        # a common cost function can be reused
        shifted_mask = np.logical_and.reduce(((self.Hx+int(xs)>=0),
                                              (self.Hx+int(xs)<self.shape[0]), 
                                              (self.Hy+int(ys)>=0),
                                              (self.Hy+int(ys)<self.shape[1])))
        mask = self.lh_mask & shifted_mask
        x_ind, y_ind = self.Hx + int(xs), self.Hy + int(ys)
        High_resolution_band, Low_resolution_band = conved[x_ind[mask], y_ind[mask]], self.low_img[mask]
        m_fed, s_fed = self.slop*Low_resolution_band+self.off, High_resolution_band
        try:
            r = scipy.stats.linregress(m_fed, s_fed)
            cost = abs(1-r.rvalue)
        except:
            cost = 100000000000.
        return cost


    def fire_shift_optimize(self,):
        #self.S2_PSF_optimization()
        self._preprocess()
        min_val = [-50,-50]
        max_val = [50,50]
        ps, distributions = create_training_set([ 'xs', 'ys'], min_val, max_val, n_train=50)
        self.shift_solved = parmap(self.shift_optimize, ps, nprocs=10)    
        self.paras, self.costs = np.array([i[0] for i in self.shift_solved]), \
                                           np.array([i[1] for i in self.shift_solved])

        xs, ys = self.paras[self.costs==self.costs.min()][0].astype(int)
        #print 'Best shift is ', xs, ys, 'with the correlation of', 1-self.costs.min()
        return xs, ys


    def fire_gaus_optimize(self,):
        xs, ys = self.fire_shift_optimize()
        if self.costs.min()<0.1:
            min_val = [12,12, -15,xs-2,ys-2]
            max_val = [50,50, 15, xs+2,ys+2]
            self.bounds = [12,50],[12,50],[-15,15],[xs-2,xs+2],[ys-2, ys+2]

            ps, distributions = create_training_set(self.parameters, min_val, max_val, n_train=50)
            print 'Start solving...'
            self.gaus_solved = parmap(self.gaus_optimize, ps, nprocs=5)
            result = np.array([np.hstack((i[0], i[1])) for i in  self.gaus_solved])
            print 'solved psf', dict(zip(self.parameters+['cost',],result[np.argmin(result[:,-1])]))
            return result[np.argmin(result[:,-1]),:]
        else:
            print 'Cost is too large, plese check!'
            return []
