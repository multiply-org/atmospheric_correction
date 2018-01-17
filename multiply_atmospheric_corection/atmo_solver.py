#/usr/bin/env python 
import sys
sys.path.insert(0, 'util')
import numpy as np
from glob import glob
try:
    import cPickle as pkl
except:
    import pickle as pkl
from scipy import optimize
from fastDiff import fastDiff
from multi_process import parmap

class solving_atmo_paras(object): 
    '''
    A simple implementation of dark dense vegitation method for the restieval of prior aod.
    '''
    def __init__(self,
                 boa, toa,
                 sza, vza,
                 saa, vaa,
                 aod_prior,
                 tcwv_prior,
                 tco3_prior,
                 elevation,
                 aod_unc,
                 tcwv_unc,
                 tco3_unc,
                 boa_unc,
                 Hx, Hy,
                 mask,
                 full_res,
                 aero_res,
                 emulators, 
                 band_indexs,
                 band_wavelength,
                 pix_res = 10.,
                 gamma   = 0.5,
                 alpha   = -1.6,# from nasa modis climatology
                 subsample = 1,
                 subsample_start = 0
                 ):
        
        self.boa             = boa
        self.toa             = toa
        self.sza             = np.cos(sza*np.pi/180.)
        self.vza             = np.cos(vza*np.pi/180.)
        self.saa             = np.cos(saa*np.pi/180.)
        self.vaa             = np.cos(vaa*np.pi/180.)
        if self.sza.ndim == 3:
            self.sza, self.saa = self.sza[0], self.saa[0]
        self.raa             = np.cos((self.saa - self.vaa)*np.pi/180.)
        self.aod_prior       = aod_prior
        self.tcwv_prior      = tcwv_prior
        self.tco3_prior      = tco3_prior
        self.ele             = elevation
        self.aod_unc         = aod_unc
        self.tcwv_unc        = tcwv_unc
        self.tco3_unc        = tco3_unc
        self.boa_unc         = boa_unc
        self.Hx, self.Hy     = Hx, Hy
        self.mask            = mask
        self.full_res        = full_res
        self.aero_res        = aero_res
        self.emus            = np.array(emulators)
        self.band_indexs     = band_indexs
        self.gamma           = gamma
        self.alpha           = alpha
        self.band_weights    = (np.array(band_wavelength)/1000.)**self.alpha
        self.band_weights    = self.band_weights / self.band_weights.sum() 
        self.pix_res         = pix_res
        self.subsample       = subsample
        self.subsample_start = subsample_start
        self.b_m_pixs        = (self.aero_res/500.)**2
 
    def _pre_process(self,):
        self.block_size   = int(np.ceil(1. * self.aero_res / self.pix_res))
        self.num_blocks_x = int(np.ceil(self.full_res[0]/(self.block_size)))
        self.num_blocks_y = int(np.ceil(self.full_res[1]/(self.block_size)))
        #self.mask = self.mask.reshape(self.num_blocks, self.block_size, \
        #                              self.num_blocks, self.block_size).astype(int).sum(axis=(3,1))
        #self.mask = (self.mask/((1.*self.block_size)**2)) >= 0.5 
        try: 
            #import pdb; pdb.set_trace()
            zero_aod  = np.zeros((self.num_blocks_x, self.num_blocks_y))
            zero_tcwv = np.zeros((self.num_blocks_x, self.num_blocks_y))
            zero_tco3 = np.zeros((self.num_blocks_x, self.num_blocks_y))
            self.control_variables = np.zeros((self.boa.shape[0], 7, self.num_blocks_x, self.num_blocks_y))
            if self.vza.ndim == 2:
                for i, parameter in enumerate([self.sza, self.vza, self.raa, zero_aod, zero_tcwv, zero_tco3, self.ele]):
                    if parameter.shape != (self.num_blocks_x, self.num_blocks_y):
                        self.control_variables[:, i, :, :] = self._block_resample(parameter)
                    else:
                        self.control_variables[:, i, :, :] = parameter
            elif self.vza.ndim == 3:
                for j in range(len(self.vza)):
                    for i, parameter in enumerate([self.sza, self.vza[j], self.raa[j], zero_aod, zero_tcwv, zero_tco3, self.ele]):
                        if parameter.shape != (self.num_blocks_x, self.num_blocks_y):
                            self.control_variables[j, i, :, :] = self._block_resample(parameter)
                        else:
                            self.control_variables[j, i, :, :] = parameter
            else:
                raise IOError('Angles should be 2D arrays.')
        except:
            raise IOError('Check the shape of input angles and elevation.') 
        try:
            self.prior_uncs = np.zeros((3, self.num_blocks_x * self.num_blocks_y))
            for i, parameter in enumerate([self.aod_unc, self.tcwv_unc, self.tco3_unc]):
                if  parameter.shape != (self.num_blocks_x, self.num_blocks_y):
                    self.prior_uncs[i]     = self._block_resample(parameter).ravel()
                else:
                    self.prior_uncs[i]     = parameter.ravel()
        except:
            raise IOError('Check the shape of input uncertainties.')
        try:
            self.priors = np.zeros((3, self.num_blocks_x * self.num_blocks_y))
            for i, parameter in enumerate([self.aod_prior, self.tcwv_prior, self.tco3_prior]):
                if  parameter.shape != (self.num_blocks_x, self.num_blocks_y):
                    self.priors[i]   = self._block_resample(parameter).ravel()
                else:
                    self.priors[i]   = parameter.ravel()
        except:
            raise IOError('Check the shape of input uncertainties.')

        self.resample_hx = (1. * self.Hx / self.full_res[0] * self.num_blocks_x).astype(int)
        self.resample_hy = (1. * self.Hy / self.full_res[1] * self.num_blocks_y).astype(int)
        self.xap_emus    = self.emus[0][self.band_indexs]
        self.xbp_emus    = self.emus[1][self.band_indexs]
        self.xcp_emus    = self.emus[2][self.band_indexs]
        self.up_bounds   = self.xap_emus[0].inputs[:,3:6].max(axis=0)
        self.bot_bounds  = self.xap_emus[0].inputs[:,3:6].min(axis=0)
        self.bot_bounds[0] = 0.001
        #self.uncs        = self.uncs  [:, self.resample_hx, self.resample_hx]
        #self.priors      = self.priors[:, self.resample_hx, self.resample_hx]
        y                = np.zeros((self.num_blocks_x, self.num_blocks_y))
        self.diff        = fastDiff(y,axis=(0,),gamma = self.gamma) 
       
    def _block_resample(self, parameter):
        hx = np.repeat(range(self.num_blocks_x), self.num_blocks_y)
        hy = np.tile  (range(self.num_blocks_y), self.num_blocks_x)
        x_size, y_size = parameter.shape
        resample_x = (1.* hx / self.num_blocks_x*x_size).astype(int)
        resample_y = (1.* hy / self.num_blocks_y*y_size).astype(int)
        resampled_parameter = parameter[resample_x, resample_y].reshape(self.num_blocks_x, self.num_blocks_y)
        return resampled_parameter

    def _helper(self, inp):
        H, _, dH = inp[0].predict(inp[1].T, do_unc=True)
        H, dH    = np.array(H).reshape(self.num_blocks_x, self.num_blocks_y), \
                   np.array(dH)[:,3:6].reshape(self.num_blocks_x, self.num_blocks_y, 3)
        return np.hstack([H[self.resample_hx, self.resample_hy][..., None], dH[self.resample_hx, self.resample_hy, :]])

    def _obs_cost(self, p, is_full = True):
        p = np.array(p).reshape(3, -1)
        X = self.control_variables.reshape(self.boa.shape[0], 7, -1)
        X[:, 3:6, :] = np.array(p)
        xap_H,  xbp_H,  xcp_H  = [], [], []
        xap_dH, xbp_dH, xcp_dH = [], [], []
          
        for i in range(len(self.xap_emus)):
            H, dH   = self.xap_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(self.num_blocks_x, self.num_blocks_y), \
                      np.array(dH)[:,3:6].reshape(self.num_blocks_x, self.num_blocks_y, 3)
            xap_H. append(H [self.resample_hx,self.resample_hy])
            xap_dH.append(dH[self.resample_hx,self.resample_hy,:])

            H, dH   = self.xbp_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(self.num_blocks_x, self.num_blocks_y), \
                      np.array(dH)[:,3:6].reshape(self.num_blocks_x, self.num_blocks_y, 3)
            xbp_H. append(H [self.resample_hx,self.resample_hy])
            xbp_dH.append(dH[self.resample_hx,self.resample_hy,:])

            H, dH   = self.xcp_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(self.num_blocks_x, self.num_blocks_y), \
                      np.array(dH)[:,3:6].reshape(self.num_blocks_x, self.num_blocks_y, 3)
            xcp_H. append(H [self.resample_hx,self.resample_hy])
            xcp_dH.append(dH[self.resample_hx,self.resample_hy,:])
        #import pdb;pdb.set_trace()
        xap_H,  xbp_H,  xcp_H  = np.array(xap_H),  np.array(xbp_H),  np.array(xcp_H)
        xap_dH, xbp_dH, xcp_dH = np.array(xap_dH), np.array(xbp_dH), np.array(xcp_dH)
        y        = xap_H * self.toa - xbp_H
        sur_ref  = y / (1 + xcp_H * y)
        diff     = sur_ref - self.boa
        full_J   = np.nansum(0.5 * self.band_weights[...,None] * (diff)**2 / self.boa_unc**2, axis=0)
        J        = np.zeros(self.full_res)
        J[self.Hx, self.Hy] = full_J
        J = np.nansum(J.reshape(self.num_blocks_x, self.block_size, \
                                self.num_blocks_y, self.block_size).sum(axis=(3,1))*self.mask)
        dH       = -1 * (-self.toa[...,None] * xap_dH + xcp_dH * (xbp_H[...,None] - xap_H[...,None] * self.toa[...,None])**2 + \
                         xbp_dH) /(self.toa[...,None] * xap_H[...,None] * xcp_H[...,None] - xbp_H[...,None] * xcp_H[...,None] + 1)**2
        full_dJ  = [ self.band_weights[...,None] * dH[:,:,i] * diff / (self.boa_unc**2) for i in range(3)]

        if is_full:
            dJ = np.nansum(np.array(full_dJ), axis=(1,))
            J_ = np.zeros((3,) + self.full_res)
            J_[:, self.Hx, self.Hy] = dJ
            J_ = np.nansum(J_.reshape(3, self.num_blocks_x, self.block_size, \
                                         self.num_blocks_y, self.block_size), axis=(4,2))
            J_[:, ~self.mask] = 0
            J_ = J_.reshape(3, -1)
        else:
            J_ = np.nansum(np.array(full_dJ), axis=(1, 2))
        return J, J_

    def _obs_cost_test(self, p, is_full = True):
        p = np.array(p).reshape(3, -1)
        X = self.control_variables.reshape(self.boa.shape[0], 7, -1)
        X[:, 3:6, :] = np.array(p)
        xap_H,  xbp_H,  xcp_H  = [], [], []
        xap_dH, xbp_dH, xcp_dH = [], [], []
        emus = list(self.xap_emus) + list(self.xbp_emus) + list(self.xcp_emus)
        Xs   = list(X)        + list(X)        + list(X)
        inps = list(zip(emus, Xs))
        #self._helper(inps[0])
        ret = np.array(parmap(self._helper, inps))
        xap_H,  xbp_H,  xcp_H  = ret[:, :, 0] .reshape(3, self.boa.shape[0], len(self.resample_hx))
        xap_dH, xbp_dH, xcp_dH = ret[:, :, 1:].reshape(3, self.boa.shape[0], len(self.resample_hx), 3)
        y        = xap_H * self.toa - xbp_H
        sur_ref  = y / (1 + xcp_H * y) 
        diff     = sur_ref - self.boa
        full_J   = np.nansum(0.5 * self.band_weights[...,None] * (diff)**2 / self.boa_unc**2, axis=0)
        J        = np.zeros(self.full_res)
        J[self.Hx, self.Hy] = full_J
        J = np.nansum(J.reshape(self.num_blocks_x, self.block_size, \
                                self.num_blocks_y, self.block_size).sum(axis=(3,1))*self.mask)
        #dH       = -1 * (-self.toa[...,None] * xap_dH + xcp_dH * (xbp_H[...,None] - xap_H[...,None] * self.toa[...,None])**2 + \
        #                 xbp_dH) /(self.toa[...,None] * xap_H[...,None] * xcp_H[...,None] - xbp_H[...,None] * xcp_H[...,None] + 1)**2
        dH       = -1 * (-self.toa[...,None] * xap_dH - \
                         2 * self.toa[...,None] * xap_H[...,None] * xbp_H[...,None] * xcp_dH + \
                         self.toa[...,None]**2 * xap_H[...,None]**2 * xcp_dH + \
                         xbp_dH + \
                         xbp_H[...,None]**2 * xcp_dH) / \
                         (self.toa[...,None] * xap_H[...,None] * xcp_H[...,None] - \
                          xbp_H[...,None] * xcp_H[...,None] + 1)**2 
        full_dJ  = [ self.band_weights[...,None] * dH[:,:,i] * diff / (self.boa_unc**2) for i in range(3)]
        
        if is_full:
            dJ = np.nansum(np.array(full_dJ), axis=(1,))
            J_ = np.zeros((3,) + self.full_res)
            J_[:, self.Hx, self.Hy] = dJ
            J_ = np.nansum(J_.reshape(3, self.num_blocks_x, self.block_size, \
                                         self.num_blocks_y, self.block_size), axis=(4,2))
            J_[:, ~self.mask] = 0
            J_ = J_.reshape(3, -1)
        else:
            J_ = np.nansum(np.array(full_dJ), axis=(1, 2))
        return J, J_
         
    def _smooth_cost(self, p, is_full=True):
        p = np.array(p).reshape(3, -1)
        aod, tcwv, tco3 = np.array(p).reshape(3, self.num_blocks_x, self.num_blocks_y)
        J_aod,  J_aod_  = self.diff.cost_der_cost(aod,  self.mask)
        J_tcwv, J_tcwv_ = self.diff.cost_der_cost(tcwv, self.mask)
        J_tco3, J_tco3_ = self.diff.cost_der_cost(tco3, self.mask)
        J, full_dJ      = J_aod + J_tcwv + J_tco3, np.array([J_aod_, J_tcwv_, J_tco3_])
        if is_full:
            J_ = np.array(full_dJ).reshape(3, -1)
        else:
            J_ = np.array(full_dJ).reshape(3, -1).sum(axis=(1,))
        return J, J_

    def _new_smooth_cost(self, p, is_full=True):
        p = np.array(p).reshape(3, -1)
        aod, tcwv, tco3 = np.array(p).reshape(3, self.num_blocks_x, self.num_blocks_y)
        J_aod,  J_aod_  = self._fit_smoothness(aod,  self.mask, 1. / self.gamma)
        J_tcwv, J_tcwv_ = self._fit_smoothness(tcwv, self.mask, 1. / self.gamma)
        J_tco3, J_tco3_ = self._fit_smoothness(tco3, self.mask, 1. / self.gamma)
        J, full_dJ      = J_aod + J_tcwv + J_tco3, np.array([J_aod_, J_tcwv_, J_tco3_])
        if is_full:
            J_ = np.array(full_dJ).reshape(3, -1)
        else:
            J_ = np.nansum(np.array(full_dJ).reshape(3, -1), axis=(1,))
        return J, J_

    def _fit_smoothness (self,  x, mask, sigma_model):
        # Build up the 8-neighbours
        hood = np.array ( [  x[:-2, :-2], x[:-2, 1:-1], x[ :-2, 2: ], \
                         x[ 1:-1,:-2], x[1:-1, 2:], \
                         x[ 2:,:-2], x[ 2:, 1:-1], x[ 2:, 2:] ] )
        j_model = 0
        der_j_model = np.zeros_like(x)
        for i in [1,3,4,6]:
            dif        = hood[i,:,:] - x[1:-1,1:-1] 
            dif[~mask[1:-1,1:-1]] = 0
            j_model = j_model + 0.5 * np.sum(dif **2)/sigma_model**2
            der_j_model[1:-1,1:-1] = der_j_model[1:-1,1:-1] - dif/sigma_model**2
        
        return j_model, 2 * der_j_model
    

    def _prior_cost(self, p, is_full=True):
        p                 = np.array(p).reshape(3, -1)
        J                 = 0.5 * (p - self.priors)**2/self.prior_uncs**2
        full_dJ           = (p - self.priors)/self.prior_uncs**2
        J      [:, ~self.mask.ravel()] = 0
        full_dJ[:, ~self.mask.ravel()] = 0
        if is_full:
            J_ = np.array(full_dJ)
        else:
            J_ = np.nansum(np.array(full_dJ), axis=(1,))
        J = np.array(J).sum()
        return J, J_

    def _cost(self, p):
        print('-------------------------------------------------------------------------------')
        print('Means:   ', list(np.array(p).reshape(3, -1)[:, self.mask.ravel()].mean(axis=-1)))
        obs_J, obs_J_       = self._obs_cost_test(p)
        prior_J, prior_J_   = self._prior_cost(p)
        #smooth_J, smooth_J_ = self._smooth_cost(p)
        smooth_J, smooth_J_ = self._new_smooth_cost(p)
        J = obs_J/self.b_m_pixs + prior_J + smooth_J
        J_ = (obs_J_/self.b_m_pixs +  prior_J_ + smooth_J_).ravel()
        print('costs:   ', [obs_J/self.b_m_pixs, prior_J, smooth_J])
        print('J_prime: ', list(((obs_J_/self.b_m_pixs)[:,self.mask.ravel()] +  prior_J_[:, self.mask.ravel()] + smooth_J_[:, self.mask.ravel()]).sum(axis=1)))
        print('-------------------------------------------------------------------------------')
        return J, J_
        
    def _optimization(self,):
        self._pre_process()
        p0  = self.priors
        bot = np.zeros_like(p0)
        up  = np.zeros_like(p0)
        bot = np.ones(p0.shape) * self.bot_bounds[...,None] 
        up  = np.ones(p0.shape) * self.up_bounds [...,None]
        p0  = p0.ravel()
        bot = bot.ravel()
        up  = up.ravel()
        bounds  = np.array([bot, up]).T 
        psolve = optimize.fmin_l_bfgs_b(self._cost, p0, approx_grad = 0, iprint = 1, m=20,\
                                        maxiter=500, pgtol = 1e-3,factr=1e6, bounds = bounds,fprime=None)
        return psolve

if __name__ == '__main__':
    sza  = np.ones((23,23))
    vza  = np.ones((23,23))
    vaa  = np.ones((23,23))
    saa  = np.ones((23,23))
    ele  = np.ones((61,61))
    aod  = np.ones((61,61))
    tcwv = np.ones((61,61))
    tco3 = np.ones((61,61))
    aod_unc  = np.ones((61,61))
    tcwv_unc = np.ones((61,61))
    tco3_unc = np.ones((61,61))
    sza[:]  = 30.
    vza[:]  = 10.
    vaa[:]  = 100.
    saa[:]  = 150.
    ele[:]  = 0.02
    aod[:]  = 0.45
    tcwv[:] = 2.3
    tco3[:] = 0.3
    aod_unc[:]  = 0.5
    tcwv_unc[:] = 0.2
    tco3_unc[:] = 0.2
    toa      = np.random.rand(6, 50000)
    y        = toa * 2.639794 -  0.038705
    boa      = y/(1+0.068196*y)
    boa_unc  = np.ones(50000) * 0.05
    Hx       = np.random.choice(10980, 50000) 
    Hy       = np.random.choice(10980, 50000)
    full_res = (10980, 10980) 
    aero_res = 3050
    emus_dir = '/home/ucfafyi/DATA/Multiply/emus/'
    sensor   = 'msi'
    xap_emu  = glob(emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(sensor))[0]     
    xbp_emu  = glob(emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(sensor))[0]
    xcp_emu  = glob(emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(sensor))[0]
    f        = lambda em: pkl.load(open(em, 'rb'))
    emus     = parmap(f, [xap_emu, xbp_emu, xcp_emu])
    band_indexs     = [1, 2, 3, 7, 11, 12]
    band_wavelength = [469, 555, 645, 869, 1640, 2130]
    mask = np.zeros((10980, 10980)).astype(bool)   
    mask[1, 1] = True
    aero = solving_atmo_paras(boa, toa,
                              sza, vza,
                              saa, vaa,
                              aod,
                              tcwv,
                              tco3,
                              ele,
                              aod_unc,
                              tcwv_unc,
                              tco3_unc,
                              boa_unc,
                              Hx, Hy,
                              mask,
                              full_res,
                              aero_res,
                              emus,
                              band_indexs,
                              band_wavelength)
    solved = aero._optimization()
