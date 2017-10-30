#!/usr/bin/env python
import numpy as np

class grab_uncertainty(object):
    '''A class for the calculation of uncertainty and in the moment we only calculate modis brdf predicted boa uncertainty by taking the maxium of mod09 or 0.05*sur+0.005 divided by the magic conversion of brdf qa to uncertainty with magic number: magic = 0.618034 to get the uncertainty values for the modis predicted boa reflectance values. The AOT, water and Ozone uncertainty values are set to be constant value of 0.5, 0,5 and 0.01 respectively. No uncertainty value is used for the toa measurement. 
    '''
    def __init__(self, modis_boa = None, 
		       boa_band  = None, 
		       boa_qa    = None,
                       brdf_std  = None,
		       ):
        
        self.modis_boa = modis_boa
        self.boa_band  = boa_band
        self.boa_qa    = boa_qa
        self.brdf_std  = brdf_std
        mod09_band = [645,869,469,555,1240,1640,2130]
        mod09_band_unc = 0.0085, 0.0246, 0.0055, 0.0085, 0.0179, 0.0125, 0.0087
        self.mod09_band_unc_dict = dict(zip(mod09_band, mod09_band_unc))
        self.magic = 0.618034
        # using defult uncertainty values
        # for aot water and ozone from ECWMF
        # need to be unpdated later
        self.aot_unc = 0.5
        self.water_unc = 0.5
        self.ozone_unc = 0.5  
    def get_boa_unc(self,):
        if self.modis_boa is None:
            raise IOError('modis_boa should be specified.')
        if self.boa_qa is None:
            raise IOError('boa_qa should be specified.')
        if self.boa_band is None:
            raise IOError('boa_band should be specified.')
        assert self.modis_boa.shape == self.boa_qa.shape, "shapes do not match."
	assert len(self.boa_band)   == self.modis_boa.shape[0], "boa_band should match modis_boa."
        generalised_unc = 0.05*self.modis_boa +0.005	
        band_unc = np.zeros_like(self.modis_boa)		
        for i, band in enumerate(self.boa_band):
	    band_unc[i] = np.maximum(self.mod09_band_unc_dict[band], generalised_unc[i])
        brdf_unc = 0.05 * 1./(self.magic ** self.boa_qa) # the validation of BRDF has 0.05 uncertainty for QA=0
        self.boa_unc = np.sqrt(band_unc**2 + brdf_unc**2, self.brdf_std**2)
	return self.boa_unc

if __name__ == "__main__":
    uc = grab_uncertainty(modis_boa=np.random.rand(4,100,100),\
                          boa_band=[645,869,469,555], boa_qa = \
                          np.random.choice([0,1,255], size=(4,100,100)))
    uc.get_boa_unc()





