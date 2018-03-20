#/usr/bin/env python 
import os
import sys
import math
import logging
sys.path.insert(0, 'util')
import numpy as np
from glob import glob
try:
    import cPickle as pkl
except:
    import pickle as pkl
from scipy import optimize, interpolate
from fastDiff import fastDiff
from multi_process import parmap

#turn off multithreading
#os.environ['OPENBLAS_NUM_THREADS'] = '1' 
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class solving_atmo_paras(object): 
    '''
    A simple implementation of dark dense vegitation method for the restieval of prior aot.
    '''
    def __init__(self,
                 boa, toa,
                 sza, vza,
                 saa, vaa,
                 aot_prior,
                 tcwv_prior,
                 tco3_prior,
                 elevation,
                 aot_unc,
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
        self.aot_prior       = aot_prior
        self.tcwv_prior      = tcwv_prior
        self.tco3_prior      = tco3_prior
        self.ele             = elevation
        self.aot_unc         = aot_unc
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
 
        self.logger = logging.getLogger('MultiGrid solver')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.propagate = False
        self.logger.info('MultiGrid solver in process...')

    def _grid_conversion(self, array, new_shape):
        rs, cs = array.shape
        x, y   = np.arange(rs), np.arange(cs)
        kx = 3 if rs > 3 else 1                       
        ky = 3 if cs > 3 else 1                       
        f      = interpolate.RectBivariateSpline(x, y, array, kx=kx, ky=ky, s=0)
        nx, ny = new_shape
        nx, ny = 1. * np.arange(nx) / nx * rs, 1. * np.arange(ny) / ny * cs
        znew   = f(nx, ny)
        return znew

    def _base_grid_resample(self, array):
        rs, cs = array.shape
        self.resample_hx = (1. * self.Hx / self.full_res[0] * rs).astype(int)
        self.resample_hy = (1. * self.Hy / self.full_res[1] * cs).astype(int)
        znew   = array[self.resample_hx, self.resample_hy]
        return znew

    def _multi_grid_solver(self,):

        self.logger.propagate = False
        bx, by  = np.array(self.full_res) * 10. / self.aero_res
        level_x = math.log(bx, 2)
        level_y = math.log(by, 2)
        level   = int(min(level_x, level_y))
        scale_factors = 1. / 2**np.arange(level)[::-1]
        shapes        = (np.array([bx, by])[..., None] * scale_factors).astype(int).T[3:]

        self.xap_emus    = self.emus[0][self.band_indexs]
        self.xbp_emus    = self.emus[1][self.band_indexs]
        self.xcp_emus    = self.emus[2][self.band_indexs]
        self.up_bounds   = self.xap_emus[0].inputs[:,3:6].max(axis=0)
        self.bot_bounds  = self.xap_emus[0].inputs[:,3:6].min(axis=0) 
        self.bot_bounds[:] = 0.
        
        self.logger.info('Total %d level of grids are going to be used.'% (len(shapes)))
        for i, shape in enumerate(shapes):
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.logger.info(bcolors.BLUE + 'Optimizing at grid level %d' % (i+1) + bcolors.ENDC)
            self.num_blocks_x, self.num_blocks_y = shape
            self.b_m_pixs                        = (self.aero_res / 500.)**2
            self.control_variables = np.zeros((self.boa.shape[0], 7,shape[0], shape[1]))

            if self.vza.ndim == 2:
                for i, parameter in enumerate([self.sza, self.vza, self.raa, self.aot_prior, self.tcwv_prior, self.tco3_prior, self.ele]):
                    self.control_variables[:, i, :, :] = self._grid_conversion(parameter, shape)

            elif self.vza.ndim == 3:
                for j in range(len(self.vza)):
                    for i, parameter in enumerate([self.sza, self.vza[j], self.raa[j], self.aot_prior, self.tcwv_prior, self.tco3_prior, self.ele]):
                        self.control_variables[j, i, :, :] = self._grid_conversion(parameter, shape)

            self.prior_uncs = np.zeros((3, shape[0], shape[1]))
            for i, parameter in enumerate([self.aot_unc, self.tcwv_unc, self.tco3_unc]):
                self.prior_uncs[i] = self._grid_conversion(parameter, shape)

            self._coarse_num = np.zeros(self.full_res)
            self._coarse_num[self.Hx, self.Hy] = 1
            subs = [np.array_split(sub, self.num_blocks_y, axis=1) for sub in np.array_split(self._coarse_num, self.num_blocks_x, axis=0)]           
            self._coarse_num = np.zeros((self.num_blocks_x, self.num_blocks_y))
            for i in range(self.num_blocks_x):
                for j in range(self.num_blocks_y):
                    self._coarse_num[i,j]        = subs[i][j].sum()
            self._coarse_mask = self._coarse_num > 0
            #subs = [np.array_split(sub, self.num_blocks_y, axis=1) for sub in np.array_split(self.mask, self.num_blocks_x, axis=0)]           
            #self._coarse_mask = np.zeros((self.num_blocks_x, self.num_blocks_y))
            #for i in range(self.num_blocks_x):
            #    for j in range(self.num_blocks_y):
            #        self._coarse_mask[i,j] = subs[i][j].sum()
            #self._coarse_mask = self._coarse_mask > 0.    
            self.priors  = self.control_variables[0, [3, 4, 5], :, :].reshape(3, -1)
            p0  = self.priors
            bot = np.zeros_like(p0)
            up  = np.zeros_like(p0)
            bot = np.ones(p0.shape) * self.bot_bounds[...,None] 
            up  = np.ones(p0.shape) * self.up_bounds [...,None]
            p0  = p0.ravel()
            bot = bot.ravel()
            up  = up.ravel()
            bounds  = np.array([bot, up]).T 
            psolve = optimize.fmin_l_bfgs_b(self._cost, p0, approx_grad = 0, iprint = -1, m=20,\
                                            maxiter=500, pgtol = 1e-3,factr=1e6, bounds = bounds,fprime=None)
            self._obs_cost_test(psolve[0], do_unc = True)     
            
            self.logger.info(bcolors.GREEN + psolve[2]['task'] + bcolors.ENDC)
            self.logger.info(bcolors.GREEN + 'Iterations: %d'%psolve[2]['nit'] + bcolors.ENDC)  
            self.logger.info(bcolors.GREEN + 'Function calls: %d'%psolve[2]['funcalls'] +bcolors.ENDC)
            unc = (np.nansum([self.obs_unc.reshape(3, -1), 1. / self.prior_uncs**2 + self.gamma**2], axis = 0)) ** -0.5
            self.aot_prior, self.tcwv_prior, self.tco3_prior = psolve[0].reshape(3, self.num_blocks_x, self.num_blocks_y)
            self.aot_unc,   self.tcwv_unc,   self.tco3_unc   = unc.reshape(3, self.num_blocks_x, self.num_blocks_y)
        post_solved = np.array([self.aot_prior, self.tcwv_prior, self.tco3_prior]) 
        post_unc    = np.array([self.aot_unc,   self.tcwv_unc,   self.tco3_unc]) 
        return [post_solved, post_unc]

    def _helper(self, inp):
        H, _, dH = inp[0].predict(inp[1][:, self._coarse_mask.ravel()].T, do_unc=True)
        tmp1     = np.zeros((self.num_blocks_x, self.num_blocks_y))
        tmp2     = np.zeros((self.num_blocks_x, self.num_blocks_y, 3))
        tmp1[self._coarse_mask] = H
        tmp2[self._coarse_mask, :] = np.array(dH)[:,3:6]
        tmp1 = self._base_grid_resample(tmp1)[..., None]
        tmp2 = np.array([self._base_grid_resample(tmp2[:,:,i]) for i in range(3)]).transpose(1,0)
        return np.hstack([tmp1, tmp2])

    def _obs_cost_test(self, p, is_full = True, do_unc=False):
        p = np.array(p).reshape(3, -1)
        X = self.control_variables.reshape(self.boa.shape[0], 7, -1)
        X[:, 3:6, :] = np.array(p)
        xap_H,  xbp_H,  xcp_H  = [], [], []
        xap_dH, xbp_dH, xcp_dH = [], [], []
        emus = list(self.xap_emus) + list(self.xbp_emus) + list(self.xcp_emus)
        Xs   = list(X)             + list(X)             + list(X)
        inps = list(zip(emus, Xs))
        if p.shape[-1] > 5000:
            ret = np.array(parmap(self._helper, inps))
        else:
            ret = np.array(map(self._helper, inps))
        xap_H,  xbp_H,  xcp_H  = ret[:, :, 0] .reshape(3, self.boa.shape[0], len(self.Hx))
        xap_dH, xbp_dH, xcp_dH = ret[:, :, 1:].reshape(3, self.boa.shape[0], len(self.Hx), 3)
        y        = xap_H * self.toa - xbp_H
        sur_ref  = y / (1 + xcp_H * y) 
        diff     = sur_ref - self.boa
        full_J   = np.nansum(0.5 * self.band_weights[...,None] * (diff)**2 / self.boa_unc**2, axis=0)
        J        = np.zeros(self.full_res)
        dH       = -1 * (-self.toa[...,None] * xap_dH - \
                         2 * self.toa[...,None] * xap_H[...,None] * xbp_H[...,None] * xcp_dH + \
                         self.toa[...,None]**2 * xap_H[...,None]**2 * xcp_dH + \
                         xbp_dH + \
                         xbp_H[...,None]**2 * xcp_dH) / \
                         (self.toa[...,None] * xap_H[...,None] * xcp_H[...,None] - \
                          xbp_H[...,None] * xcp_H[...,None] + 1)**2 
        full_dJ  = [ self.band_weights[...,None] * dH[:,:,i] * diff / (self.boa_unc ** 2) for i in range(3)]
        
        if is_full:
            dJ = np.nansum(np.array(full_dJ), axis=(1,))
            J_ = np.zeros((3,) + self.full_res)
            J_[:, self.Hx, self.Hy] = dJ
            subs1 = [np.array_split(sub, self.num_blocks_y, axis=2) for sub in np.array_split(J_, self.num_blocks_x, axis=1)]

            J        = np.zeros(self.full_res)
            J[self.Hx, self.Hy] = full_J 
            subs2 = [np.array_split(sub, self.num_blocks_y, axis=1) for sub in np.array_split(J, self.num_blocks_x, axis=0)]

            J_ = np.zeros((3, self.num_blocks_x, self.num_blocks_y))
            J  = np.zeros((   self.num_blocks_x, self.num_blocks_y))            
            for i in range(self.num_blocks_x):
                for j in range(self.num_blocks_y):
                    J_[:, i,j] = np.nansum(subs1[i][j], axis=(1,2))
                    J [   i,j] = np.nansum(subs2[i][j], axis=(0,1))

            J_[:, ~self._coarse_mask] = 0
            J [   ~self._coarse_mask] = 0
            J_ = J_.reshape(3, -1)
            if do_unc:
                comb_unc              = np.nansum([self.band_weights[...,None] * (dH[:, :, i] ** 2) * (self.boa_unc ** -2)  for i in range(3)], axis = 1)
                comb_unc[comb_unc==0] = np.nan
                self.obs_unc          = np.zeros((3,) + self.full_res)
                self.obs_unc[:]       = np.nan
                self.obs_unc[:, self.Hx, self.Hy] = comb_unc
                subs = [np.array_split(sub, self.num_blocks_y, axis=2) for sub in np.array_split(self.obs_unc, self.num_blocks_x, axis=1)]
                self.obs_unc = np.zeros((3, self.num_blocks_x, self.num_blocks_y))
                for i in range(self.num_blocks_x):                                                                               
                    for j in range(self.num_blocks_y):                                                                           
                        self.obs_unc[:, i,j] = np.nanmean(subs[i][j], axis=(1,2))     
                self.obs_unc[:,~self._coarse_mask] = np.nan
                return self.obs_unc
        else:
            J  = np.nansum(np.array(full_J))
            J_ = np.nansum(np.array(full_dJ), axis=(1, 2))
        return J, J_

    def _new_smooth_cost(self, p, is_full=True):
        p = np.array(p).reshape(3, -1)
        aot, tcwv, tco3 = np.array(p).reshape(3, self.num_blocks_x, self.num_blocks_y)
        J_aot,  J_aot_  = self._fit_smoothness(aot,  1. / self.gamma)
        J_tcwv, J_tcwv_ = self._fit_smoothness(tcwv, 1. / self.gamma)
        J_tco3, J_tco3_ = self._fit_smoothness(tco3, 1. / self.gamma)
        J, full_dJ      = J_aot + J_tcwv + J_tco3, np.array([J_aot_, J_tcwv_, J_tco3_])
        if is_full:
            J_ = np.array(full_dJ).reshape(3, -1)
        else:
            J_ = np.nansum(np.array(full_dJ).reshape(3, -1), axis=(1,))
        return J, J_

    def _fit_smoothness (self,  x, sigma_model):
        grad = np.gradient(np.gradient(x, axis=0), axis=1)
        j_model     = 0.5 * grad **2 / sigma_model**2
        der_j_model = grad / sigma_model**2
        return j_model, der_j_model
    
    def _prior_cost(self, p, is_full=True):
        self.prior_uncs   = self.prior_uncs.reshape(3, -1)
        p                 = np.array(p).reshape(3, -1)
        J                 = 0.5 * (p - self.priors)**2 / self.prior_uncs**2
        full_dJ           = (p - self.priors)/self.prior_uncs**2
        J      [:, ~self._coarse_mask.ravel()] = 0
        full_dJ[:, ~self._coarse_mask.ravel()] = 0
        J                 = J.sum(axis=0)
        if is_full:
            return J.reshape(self.num_blocks_x, self.num_blocks_y), full_dJ
        else:
            J = np.array(J).sum()
            J_ = np.nansum(np.array(full_dJ), axis=(1,))
            return J, J_

    def _cost(self, p):
        print('---------------------------------------------------------------------------------------------------------')
        means = tuple(np.array(p).reshape(3, -1)[:, self._coarse_mask.ravel()].mean(axis=-1))
        self.logger.info('Means:    %-12.03f  %-12.03f  %-12.03f'%means)

        obs_J, obs_J_       = self._obs_cost_test(p)
        prior_J, prior_J_   = self._prior_cost(p)
        smooth_J, smooth_J_ = self._new_smooth_cost(p)
        J  = np.nansum(obs_J/self.b_m_pixs + prior_J + smooth_J)
        J_ = (obs_J_/self.b_m_pixs +  prior_J_ + smooth_J_).ravel()

        costs    = (np.nansum(obs_J/self.b_m_pixs), np.nansum(prior_J), np.nansum(smooth_J))
        J_primes = tuple(((obs_J_/self.b_m_pixs)[:,self._coarse_mask.ravel()] + \
                      prior_J_[:, self._coarse_mask.ravel()] + \
                      smooth_J_[:, self._coarse_mask.ravel()]).sum(axis=1))
        self.logger.info('costs:    %-12.03f  %-12.03f  %-12.03f'%costs)
        self.logger.info('J_primes: %-12.03f  %-12.03f  %-12.03f'%J_primes)
        print('---------------------------------------------------------------------------------------------------------')

        return J, J_

if __name__ == '__main__':
    sza  = np.ones((23,23))
    vza  = np.ones((23,23))
    vaa  = np.ones((23,23))
    saa  = np.ones((23,23))
    ele  = np.ones((61,61))
    aot  = np.ones((61,61))
    tcwv = np.ones((61,61))
    tco3 = np.ones((61,61))
    aot_unc  = np.ones((61,61))
    tcwv_unc = np.ones((61,61))
    tco3_unc = np.ones((61,61))
    sza[:]  = 30.
    vza[:]  = 10.
    vaa[:]  = 100.
    saa[:]  = 150.
    ele[:]  = 0.02
    aot[:]  = 0.45
    tcwv[:] = 2.3
    tco3[:] = 0.3
    aot_unc[:]  = 0.5
    tcwv_unc[:] = 0.2
    tco3_unc[:] = 0.2
    toa      = np.random.rand(6, 50000)
    y        = toa * 2.639794 -  0.038705
    boa      = y/(1+0.068196*y)
    boa_unc  = np.ones(50000) * 0.05
    Hx       = np.random.choice(10980, 50000) 
    Hy       = np.random.choice(10980, 50000)
    full_res = (10980, 10980) 
    aero_res = 600
    emus_dir = '/home/ucfafyi/DATA/Multiply/emus/'
    sensor   = 'MSI'
    xap_emu  = glob(emus_dir + '/isotropic_%s_emulators_*xap*.pkl'%(sensor))[0]     
    xbp_emu  = glob(emus_dir + '/isotropic_%s_emulators_*xbp*.pkl'%(sensor))[0]
    xcp_emu  = glob(emus_dir + '/isotropic_%s_emulators_*xcp*.pkl'%(sensor))[0]
    f        = lambda em: pkl.load(open(em, 'rb'))
    emus     = parmap(f, [xap_emu, xbp_emu, xcp_emu])
    band_indexs     = [1, 2, 3, 7, 11, 12]
    band_wavelength = [469, 555, 645, 869, 1640, 2130]
    mask = np.zeros((10980, 10980)).astype(bool)   
    mask[1, 1] = True
    aero = solving_atmo_paras(boa, toa,
                              sza, vza,
                              saa, vaa,
                              aot,
                              tcwv,
                              tco3,
                              ele,
                              aot_unc,
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
    solved = aero._multi_grid_solver()
