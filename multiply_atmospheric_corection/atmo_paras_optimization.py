#/usr/bin/env python
import numpy as np
import sys
sys.path.insert(0, 'python')
from emulation_engine import AtmosphericEmulationEngine
from grab_uncertainty import grab_uncertainty
from scipy import optimize 

class solving_atmo_paras(object):
    '''
    A class taking the toa, boa, initial [aot, water, ozone], [vza, sza, vaa, saa], elevation and emulators
    to do the atmospheric parameters retrival and do the atmopsheric correction.  
    '''
    def __init__(self, sensor,
                       emus_dir,
                       boa,toa, 
                       sza, vza,
                       saa, vaa, 
                       elevation,
                       boa_qa, boa_bands,
                       band_indexs, 
                       mask,prior,
                       brdf_std,
                       atmosphere = None, 
                       subsample  = None,
                       subsample_start = 0,
                       gradient_refl=True, 
                       bands=None):
        
        self.alpha         = -1.42 #angstrom exponent for continental type aerosols
        self.sensor        = sensor
        self.emus_dir      = emus_dir
        self.boa, self.toa = boa, toa
        self.atmosphere    = atmosphere
        self.sza, self.vza = sza, vza
        self.saa, self.vaa = saa, vaa
        self.elevation     = elevation
        self.boa_qa        = boa_qa
        self.boa_bands     = boa_bands
        self.band_weights  = (np.array(self.boa_bands)/1000.)**self.alpha
        self.band_indexs   =  band_indexs
        self.mask          = mask
        self.prior         = prior
        self.brdf_std      = brdf_std
        if subsample is None:
            self.subsample = 1
        else:
           self.subsample  = subsample
        self.subsample_sta = subsample_start
    
    def _load_emus(self):
        self.AEE = AtmosphericEmulationEngine(self.sensor, self.emus_dir)
        up_bounds   = self.AEE.emulators[0].inputs[:,4:7].max(axis=0)
        low_bounds  = self.AEE.emulators[0].inputs[:,4:7].min(axis=0)
        self.bounds = np.array([low_bounds, up_bounds]).T 

    def _load_unc(self):
        uc = grab_uncertainty(self.boa, self.boa_bands, self.boa_qa, self.brdf_std)
        self.boa_unc   = uc.get_boa_unc()
        self.aot_unc   = uc.aot_unc
        self.water_unc = uc.water_unc
        self.ozone_unc = uc.ozone_unc

    def _sort_emus_inputs(self,):

        assert self.boa.shape[1:] == self.mask.shape, 'mask should have the same shape as the last two axises of boa.'
        assert self.boa.shape      == self.toa.shape, 'toa and boa should have the same shape.'
        assert self.boa.shape      == self.boa_unc.shape, 'boa and boa_unc should have the same shape.'
        if self.atmosphere is not None:
            assert self.atmosphere.shape[0] == 3, 'Three parameters, i.e. AOT, water and Ozone are needed.'
            assert self.boa.shape[1:] == self.atmosphere.shape[1:], 'boa and atmosphere should have the same shape in the last two axises.'
        # make the boa and toa to be the shape of nbands * nsample
        # and apply the flattened mask and subsample 
        if self.mask.ndim == 2:
            flat_mask = self.mask[self.subsample_sta::self.subsample, \
                                  self.subsample_sta::self.subsample].flatten()
        elif self.mask.ndim == 1:
            flat_mask = self.mask[self.subsample_sta::self.subsample]  
        else:
            raise IOError('Wrong shape mask is given.')
    
        if  self.boa.ndim == 3:  
            flat_boa = self.boa[:,self.subsample_sta::self.subsample, \
                                  self.subsample_sta::self.subsample].reshape(self.boa.shape[0], -1)[..., flat_mask]
        elif self.boa.ndim == 2:
             flat_boa = self.boa[:,self.subsample_sta::self.subsample][..., flat_mask]
        else:
            raise IOError('Wrong shape BOA is given.')

        if self.toa.ndim == 3:
            flat_toa     = self.toa[:,self.subsample_sta::self.subsample, \
                                      self.subsample_sta::self.subsample].reshape(self.toa.shape[0], -1)[..., flat_mask]
        elif self.toa.ndim == 2:
            flat_toa = self.toa[:,self.subsample_sta::self.subsample][..., flat_mask]
        else:
            raise IOError('Wrong shape TOA is given.')
        if self.boa_unc.ndim == 3:
            flat_boa_unc = self.boa_unc[:,self.subsample_sta::self.subsample, \
                                          self.subsample_sta::self.subsample].reshape(self.toa.shape[0], -1)[..., flat_mask]
        elif self.boa_unc.ndim == 2:
             flat_boa_unc = self.boa_unc[:,self.subsample_sta::self.subsample][..., flat_mask]
        else:
            raise IOError('Wrong shape BOA uncertainty is given.')

        if self.atmosphere is not None:
            if self.atmosphere.ndim == 3:
                flat_atmos = self.atmosphere[:, self.subsample_sta::self.subsample, \
                                                self.subsample_sta::self.subsample].reshape(3, -1)[..., flat_mask]
            elif self.atmosphere.ndim == 2:
                flat_atmos = self.atmosphere[:, self.subsample_sta::self.subsample][..., flat_mask]

            self.flat_atmos = flat_atmos
        else:
            flat_atmos = np.array([])
            self.flat_atmos = flat_atmos   
        flat_angs_ele = []
        for i in [self.sza, self.vza, self.saa, self.vaa]:
            if isinstance(i, (float,int)):
                flat_angs_ele.append(i)
	    elif i.ndim ==3:
		assert i.shape[0] == self.boa.shape[0], 'check the shape of angles.'
		flat_i = i[:,self.subsample_sta::self.subsample, \
			     self.subsample_sta::self.subsample].reshape(i.shape[0], -1)[..., flat_mask]
	    elif (i.ndim == 2) & (i.shape == self.boa.shape[1:]):
		flat_i = i[self.subsample_sta::self.subsample, \
			   self.subsample_sta::self.subsample].flatten()[flat_mask]
	    elif (i.ndim == 2) & (i.shape != self.boa.shape[1:]):
		assert i.shape[0] == self.boa.shape[0], 'check the shape of angles.'
		flat_i = i[:, self.subsample_sta::self.subsample][..., flat_mask]
	    elif (i.ndim == 1) & (i.shape[0] == self.boa.shape[0]):
		flat_i = i
	    else:
		raise IOError('Wrong shape angles is given.')
	    flat_angs_ele.append([np.array(ang) for ang in \
				  np.ones(flat_boa.shape)*flat_i]) # make sure the type dose not change...

        if isinstance(self.elevation, (float, int)):
            flat_angs_ele.append(self.elevation)
          
        elif self.elevation.ndim == 2:
            flat_ele  = self.elevation[self.subsample_sta::self.subsample, \
                                       self.subsample_sta::self.subsample].flatten()[flat_mask]
            flat_angs_ele.append(flat_ele)
        elif self.elevation.ndim == 1:
            flat_ele = self.elevation[self.subsample_sta::self.subsample][flat_mask]
            flat_angs_ele.append(flat_ele)
       
        ## for the prior
        if np.array(self.prior).ndim == 1:
            self.flat_prior = np.array(self.prior) 
        elif np.array(self.prior).ndim == 2:
            assert self.prior.shape[1] == self.boa.shape[1], 'prior should have the same shape as the second axis of boa.'
            self.flat_prior = np.array(self.prior)[:, self.subsample_sta::self.subsample][..., flat_mask]
        elif np.array(self.prior).ndim == 3:
            assert self.prior.shape == self.boa.shape[1:], 'prior should have the same shape as the last two axises of boa.'
            self.flat_prior = self.prior[:, self.subsample_sta::self.subsample, \
                                            self.subsample_sta::self.subsample].reshape(3,-1)[..., flat_mask]

        return flat_mask, flat_boa, flat_toa, flat_boa_unc, flat_atmos, flat_angs_ele# [sza, vza, saa, vaa, elevation]        

    def obs_cost(self,is_full=False):

        flat_mask, flat_boa, flat_toa, flat_boa_unc, flat_atmos, [sza, vza, saa, vaa, elevation] = self._sort_emus_inputs()
        for i in [flat_mask, flat_boa, flat_toa, flat_boa_unc, flat_atmos, sza, vza, saa, vaa, elevation]:
            if np.array(i).size == 0:
                return 0., np.array([0.,0.,0.]) # any empty array result in earlier leaving the estimation
        H0, dH = self.AEE.emulator_reflectance_atmosphere(flat_boa, flat_atmos, sza, vza, saa, vaa, elevation, bands=self.band_indexs)
        H0, dH = np.array(H0), np.array(dH)
        diff = (H0 - flat_toa) # order is important!
        correction_mask = np.isfinite(diff)
        # correction mask to set 0 or larger number?
        diff[~correction_mask] = 0.
        dH[~correction_mask,:] = 0.
        J  = (0.5 * self.band_weights[...,None] * diff**2 / (self.band_weights[...,None].sum() * flat_boa_unc**2)).sum(axis=(0,1))
        full_dJ = [ self.band_weights[...,None] * dH[:,:,i] * diff/(self.band_weights[...,None].sum() * flat_boa_unc**2) for i in xrange(4,7)]
        if is_full:
            J_ = np.array(full_dJ).sum(axis=(1,))
        else:
            J_ = np.array(full_dJ).sum(axis=(1,2))
        
        return J, J_

    def prior_cost(self,is_full=False):
        # maybe need to update to per pixel basis uncertainty 
        # instead of using scaler values 0.5, 0.5, 0.001 
        uncs = np.array([self.aot_unc, self.water_unc, self.ozone_unc])[...,None]
        if self.flat_atmos.size == 0:
            return 0., np.array([0.,0.,0.])
        if self.flat_prior.ndim == 1:
            J = 0.5 * (self.flat_atmos - self.flat_prior[...,None])**2/uncs**2
            full_dJ = (self.flat_atmos - self.flat_prior[...,None])/uncs**2
        else:
            J = 0.5 * (self.flat_atmos - self.flat_prior)**2/uncs**2
            full_dJ = (self.flat_atmos - self.flat_prior)/uncs**2
        if is_full:
            J_ = np.array(full_dJ)
        else:
            J_ = np.array(full_dJ).sum(axis=(1,))

        J = np.array(J).sum()
        return J, J_

    def smooth_cost(self,):
        '''
        need to add first order regulization
        '''
        J  = 0
        J_ = np.array([0.,0.,0.])
        return J, J_

    def optimization(self,):
        '''
        An optimization function used for the retrieval of atmospheric parameters
        '''        
        p0     = self.prior 
        psolve1 = optimize.fmin_l_bfgs_b(self.fmin_l_bfgs_cost, p0, approx_grad=0, iprint=-1, \
                                         pgtol=1e-6,factr=1000, bounds=self.bounds,fprime=None)
        #psolve2 = optimize.fmin(self.fmin_cost, p0, full_output=True, maxiter=100, maxfun=150, disp=0)
        return psolve1#, psolve2
 
    def fmin_l_bfgs_cost(self,p):

        self.atmosphere     = np.array([i*np.ones(self.toa.shape[1:]) for i in p])

        obs_J, obs_J_       = self.obs_cost()
        prior_J, prior_J_   = self.prior_cost()
        smooth_J, smooth_J_ = self.smooth_cost()

        J = obs_J + prior_J + smooth_J
        J_ = obs_J_ +  prior_J_ + smooth_J_

        return J, J_

    def fmin_cost(self,p):

        self.atmosphere     = np.array([i*np.ones(self.toa.shape[1:]) for i in p])

        obs_J, obs_J_       = self.obs_cost()
        prior_J, prior_J_   = self.prior_cost()
        smooth_J, smooth_J_ = self.smooth_cost()

        J = obs_J + prior_J + smooth_J
        J_ = obs_J_ +  prior_J_ + smooth_J_

        return J

if __name__ == "__main__":
    boa = np.random.rand(4,100,100)
    boa[:] = 0.2
    toa = np.random.rand(4,100,100)
    toa[:] = 0.3
    aot = np.random.rand(100, 100)
    aot[:] = 0.3
    water = np.random.rand(100, 100)
    water[:] = 3.4
    ozone = np.zeros((100, 100))
    ozone[:] = 0.35
    atmosphere = np.array([aot,water,ozone])
    boa_qa = np.random.choice([0,1,255], size=(4,100,100))
    mask,prior = np.zeros((100, 100)).astype(bool), [0.2, 3, 0.3]
    mask[:50,:50] = True
    brdf_std = np.zeros_loke(boa)
    brdf_std[:] = 0.01
    atmo = solving_atmo_paras('MSI', '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',boa, \
                             toa,0.5,0.5,10,10,0.5, boa_qa, boa_bands=[645,869,469,555], \
                             band_indexs=[3,7,1,2], mask=mask, prior=prior, atmosphere=atmosphere, brdf_std = brdf_std)

    atmo._load_emus()
    atmo._load_unc()
    atmo._sort_emus_inputs()
    obs_J, obs_J_ = atmo.obs_cost()
    prior_J, prior_J_ = atmo.prior_cost()
    smooth_J, smooth_J_ = atmo.smooth_cost()
    J = obs_J + prior_J + smooth_J
    J_ = obs_J_ +  prior_J_ + smooth_J_
    
    print atmo.fmin_l_bfgs_cost([0.3, 3.2, 0.4],)
    print atmo.fmin_l_bfgs_cost([0.25, 3.2, 0.4],)
    atmo.optimization()

