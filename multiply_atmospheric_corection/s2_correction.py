#/usr/bin/env python
import sys
sys.path.insert(0,'python')
import gdal
import numpy as np
from numpy import clip, uint8
from glob import glob
import logging
from Py6S import *
import cPickle as pkl
from multi_process import parmap
from grab_s2_toa import read_s2
#from aerosol_solver import solve_aerosol
from reproject import reproject_data
from emulation_engine import AtmosphericEmulationEngine

class atmospheric_correction(object):
    '''
    A class doing the atmospheric coprrection with the input of TOA reflectance
    angles, elevation and emulators of 6S from TOA to surface reflectance.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 s2_tile,
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = '/home/ucfafyi/DATA/Multiply/emus/',
                 reconstruct_s2_angle = False
                 ):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.s2_tile     = s2_tile
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.sur_refs     = {}
        self.reconstruct_s2_angle = reconstruct_s2_angle
	self.logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
	self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:       
	    ch = logging.StreamHandler()
	    ch.setLevel(logging.DEBUG)
	    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	    ch.setFormatter(formatter)
	    self.logger.addHandler(ch)


    def _load_inverse_emus(self, sensor):
	AEE = AtmosphericEmulationEngine(sensor, self.emus_dir)
        return AEE

    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.s2_sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.s2_sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.s2_sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.s2_sensor = 'msi'
        self.logger.info('Loading emulators.')
        #self.s2_inv_AEE = self._load_inverse_emus(self.s2_sensor)
        self._load_xa_xb_xc_emus()
        self.s2   = read_s2(self.s2_toa_dir, self.s2_tile, \
                            self.year, self.month, self.day, bands=None)
        self.logger.info('Reading in the reflectance.')
        all_refs = self.s2.get_s2_toa()
        self.logger.info('Reading in the angles')
        self.s2.get_s2_angles(self.reconstruct_s2_angle)
        all_angs = self.s2.angles
        self.sza,self.saa = all_angs['sza'], all_angs['saa']
        
        self.logger.info('Doing 10 meter bands')
        self._10meter_ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B02', 'B03', 'B04', 'B08']])
        if self.reconstruct_s2_angle:
            self._10meter_vza = np.array([all_angs['vza'][band]/100. for band
                                          in ['B02', 'B03', 'B04', 'B08']])
            self._10meter_vaa = np.array([all_angs['vaa'][band]/100. for band
                                          in ['B02', 'B03', 'B04', 'B08']])
        else:
            self._10meter_vza = np.array([np.repeat(np.repeat(all_angs['vza'][band], int(np.ceil(10980/23.)), \
                                          axis=0), int(np.ceil(10980/23.)), axis=1)[:10980, :10980]\
                                          for band in ['B02', 'B03', 'B04', 'B08']])

            self._10meter_vaa = np.array([np.repeat(np.repeat(all_angs['vaa'][band], int(np.ceil(10980/23.)), \
                                          axis=0), int(np.ceil(10980/23.)), axis=1)[:10980, :10980]\
                                          for band in ['B02', 'B03', 'B04', 'B08']])

        self._10meter_sza = np.repeat(np.repeat(self.sza, int(np.ceil(10980/23.)), \
                                      axis=0), int(np.ceil(10980/23.)), axis=1)[:10980, :10980]
        self._10meter_saa = np.repeat(np.repeat(self.saa, int(np.ceil(10980/23.)), \
                                      axis=0), int(np.ceil(10980/23.)), axis=1)[:10980, :10980]
        self.logger.info('Getting control variables for 10 meters bands.')
        self._10meter_aod, self._10meter_tcwv, self._10meter_tco3,\
                           self._10meter_ele = self.get_control_variables('B04')
        self._block_size = 3660
        self._num_blocks = 10980/self._block_size
        self._mean_size  = 60
        self._10meter_band_indexs = [1, 2, 3, 7]        
        self.rsr = [PredefinedWavelengths.S2A_MSI_02, PredefinedWavelengths.S2A_MSI_03, \
                    PredefinedWavelengths.S2A_MSI_04, PredefinedWavelengths.S2A_MSI_08 ]
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self._10meter_ref, self._10meter_sza, self._10meter_vza,\
                             self._10meter_saa, self._10meter_vaa, self._10meter_aod,\
                             self._10meter_tcwv, self._10meter_tco3, self._10meter_ele,\
                             self._10meter_band_indexs)     
        self.toa_rgb = clip(self._10meter_ref[[2,1,0], ...].transpose(1,2,0)*255/0.25, 0., 255.).astype(uint8)
        self.boa_rgb = clip(self.boa         [[2,1,0], ...].transpose(1,2,0)*255/0.25, 0., 255.).astype(uint8)
        self._save_rgb(self.toa_rgb, 'TOA_RGB.tif', self.s2.s2_file_dir+'/B04.jp2')
        self._save_rgb(self.boa_rgb, 'BOA_RGB.tif', self.s2.s2_file_dir+'/B04.jp2')


        del self._10meter_ref; del self._10meter_vza; del self._10meter_vaa;  del self._10meter_sza 
        del self._10meter_saa; del self._10meter_aod; del self._10meter_tcwv; del self._10meter_tco3; del self._10meter_ele
        
	self.sur_refs.update(dict(zip(['B02', 'B03', 'B04', 'B08'], self.boa)))
        self._save_img(self.boa, ['B02', 'B03', 'B04', 'B08']); del self.boa 
	  
        self.logger.info('Doing 20 meter bands')
        self._20meter_ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])

        if self.reconstruct_s2_angle:
            self._20meter_vza = np.array([all_angs['vza'][band] for band \
                                          in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).reshape(6, 5490, 2, 5490, 2).mean(axis=(4,2))
            self._20meter_vaa = np.array([all_angs['vaa'][band] for band \
                                          in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).reshape(6, 5490, 2, 5490, 2).mean(axis=(4,2))
        else:
            self._20meter_vza = np.array([np.repeat(np.repeat(all_angs['vza'][band], \
                                          int(np.ceil(5490/23.)), axis=0), int(np.ceil(5490/23.)), \
                                          axis=1)[:5490, :5490] for band in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])
            self._20meter_vaa = np.array([np.repeat(np.repeat(all_angs['vaa'][band], \
                                          int(np.ceil(5490/23.)), axis=0), int(np.ceil(5490/23.)), \
                                          axis=1)[:5490, :5490] for band in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])


        #self._20meter_vza = np.array([all_angs['vza'][band]/100. for band
        #                              in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])
        #self._20meter_vaa = np.array([all_angs['vaa'][band]/100. for band
        #                              in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])
        self._20meter_sza = np.repeat(np.repeat(self.sza, int(np.ceil(5490/23.)), \
                                      axis=0), int(np.ceil(5490/23.)), axis=1)[:5490, :5490]
        self._20meter_saa = np.repeat(np.repeat(self.saa, int(np.ceil(5490/23.)), \
                                      axis=0), int(np.ceil(5490/23.)), axis=1)[:5490, :5490]

        self.logger.info('Getting control variables for 20 meters bands.')
        self._20meter_aod, self._20meter_tcwv, self._20meter_tco3,\
                           self._20meter_ele = self.get_control_variables('B05')
        self._block_size = 1830
        self._num_blocks = 5490/self._block_size
        self._mean_size  = 30
        self._20meter_band_indexs = [4, 5, 6, 8, 11, 12]
        self.rsr = [PredefinedWavelengths.S2A_MSI_05, PredefinedWavelengths.S2A_MSI_06, \
                    PredefinedWavelengths.S2A_MSI_07, PredefinedWavelengths.S2A_MSI_09, \
                    PredefinedWavelengths.S2A_MSI_12, PredefinedWavelengths.S2A_MSI_13]

        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self._20meter_ref, self._20meter_sza, self._20meter_vza,\
                             self._20meter_saa, self._20meter_vaa, self._20meter_aod,\
                             self._20meter_tcwv, self._20meter_tco3, self._20meter_ele,\
                             self._20meter_band_indexs)

        del self._20meter_ref; del self._20meter_vza; del self._20meter_vaa;  del self._20meter_sza
        del self._20meter_saa; del self._20meter_aod; del self._20meter_tcwv; del self._20meter_tco3; del self._20meter_ele

        self.sur_refs.update(dict(zip(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], self.boa)))
        self._save_img(self.boa, ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']); del self.boa


        self.logger.info('Doing 60 meter bands')
        self._60meter_ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B01', 'B09', 'B10']])
        if self.reconstruct_s2_angle:
            self._60meter_vza = np.array([all_angs['vza'][band] for band \
                                          in ['B01', 'B09', 'B10']]).reshape(3, 1830, 6, 1830, 6).mean(axis=(4,2))
            self._60meter_vaa = np.array([all_angs['vaa'][band] for band \
                                          in ['B01', 'B09', 'B10']]).reshape(3, 1830, 6, 1830, 6).mean(axis=(4,2))
        else:
            self._60meter_vza = np.array([np.repeat(np.repeat(all_angs['vza'][band], int(np.ceil(1830/23.)), \
                                          axis=0), int(np.ceil(1830/23.)), axis=1)[:1830, :1830] \
                                          for band in ['B01', 'B09', 'B10']])

            self._60meter_vaa = np.array([np.repeat(np.repeat(all_angs['vaa'][band], int(np.ceil(1830/23.)), \
                                          axis=0), int(np.ceil(1830/23.)), axis=1)[:1830, :1830] \
                                          for band in ['B01', 'B09', 'B10']])
        #self._60meter_vza = np.array([all_angs['vza'][band]/100. for band
        #                              in ['B01', 'B09', 'B10']])
        #self._60meter_vaa = np.array([all_angs['vaa'][band]/100. for band
        #                              in ['B01', 'B09', 'B10']])
        self._60meter_sza = np.repeat(np.repeat(self.sza, int(np.ceil(1830/23.)), \
                                      axis=0), int(np.ceil(1830/23.)), axis=1)[:1830, :1830]
        self._60meter_saa = np.repeat(np.repeat(self.saa, int(np.ceil(1830/23.)), \
                                      axis=0), int(np.ceil(1830/23.)), axis=1)[:1830, :1830]

        self.logger.info('Getting control variables for 60 meters bands.')
        self._60meter_aod, self._60meter_tcwv, self._60meter_tco3,\
                           self._60meter_ele = self.get_control_variables('B09')
        self._block_size = 610
        self._num_blocks = 1830/self._block_size
        self._mean_size  = 10
        self._60meter_band_indexs = [0, 9, 10]
        self.rsr = [PredefinedWavelengths.S2A_MSI_01, PredefinedWavelengths.S2A_MSI_10, PredefinedWavelengths.S2A_MSI_11]

        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self._60meter_ref, self._60meter_sza, self._60meter_vza,\
                             self._60meter_saa, self._60meter_vaa, self._60meter_aod,\
                             self._60meter_tcwv, self._60meter_tco3, self._60meter_ele,\
                             self._60meter_band_indexs)

        del self._60meter_ref; del self._60meter_vza; del self._60meter_vaa;  del self._60meter_sza
        del self._60meter_saa; del self._60meter_aod; del self._60meter_tcwv; del self._60meter_tco3; del self._60meter_ele

        self.sur_refs.update(dict(zip(['B01', 'B09', 'B10'], self.boa)))
        self._save_img(self.boa, ['B01', 'B09', 'B10']); del self.boa
        del all_refs; del self.s2.selected_img; del all_angs; del self.s2.angles
        self.logger.info('Done!')

    def _save_rgb(self, rgb_array, name, source_image):
        g            = gdal.Open(source_image)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny = rgb_array.shape[:2]
        dst_ds = gdal.GetDriverByName('GTiff').Create(self.s2.s2_file_dir+\
                                '/%s'%name, ny, nx, 3, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(rgb_array[:,:,0])
        dst_ds.GetRasterBand(2).WriteArray(rgb_array[:,:,1])
        dst_ds.GetRasterBand(3).WriteArray(rgb_array[:,:,2])
        dst_ds.FlushCache()
        dst_ds = None

    def _save_img(self, refs, bands):
        g            = gdal.Open(self.s2.s2_file_dir+'/%s.jp2'%bands[0])
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(bands, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)
      
    def _save_band(self, band_ref, projection, geotransform):
        band, ref = band_ref
        nx, ny = ref.shape
        dst_ds = gdal.GetDriverByName('GTiff').Create(self.s2.s2_file_dir+\
                                '/%s_sur.tif'%band, ny, nx, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray(ref)
        dst_ds.FlushCache()                  
        dst_ds = None

    def get_control_variables(self, target_band):

	aod = reproject_data(self.s2.s2_file_dir+'/aot.tif', \
                             self.s2.s2_file_dir+'/%s.jp2'%target_band).data

        tcwv = reproject_data(self.s2.s2_file_dir+'/tcwv.tif', \
                              self.s2.s2_file_dir+'/%s.jp2'%target_band).data

        tco3 = reproject_data(self.s2.s2_file_dir+'/tco3.tif', \
                              self.s2.s2_file_dir+'/%s.jp2'%target_band).data
        ele = reproject_data(self.global_dem, self.s2.s2_file_dir+'/%s.jp2'%target_band).data
        mask = ~np.isfinite(ele)
        ele[mask] = np.interp(np.flatnonzero(mask), \
                              np.flatnonzero(~mask), ele[~mask]) # simple interpolation

        return aod, tcwv, tco3, ele


    def fire_correction(self, toa, sza, vza, saa, vaa, aod, tcwv, tco3, elevation, band_indexs):
        self._toa         = toa
        self._sza         = sza
        self._vza         = vza
        self._saa         = saa
        self._vaa         = vaa
        self._aod         = aod
        self._tcwv        = tcwv
        self._tco3        = tco3
        self._elevation   = elevation
        self._band_indexs = band_indexs
        rows              = np.repeat(np.arange(self._num_blocks), self._num_blocks)
        columns           = np.tile(np.arange(self._num_blocks), self._num_blocks)
        blocks            = zip(rows, columns)
        #self._s2_block_correction_emus_xa_xb_xc([0, 0])
        ret = parmap(self._s2_block_correction_emus_xa_xb_xc, blocks)
        #ret = parmap(self._s2_block_correction_6s, blocks)
        #ret               = parmap(self._s2_block_correction_emus, blocks) 
        self.boa = np.array([i[2] for i in ret]).reshape(self._num_blocks, self._num_blocks, toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(toa.shape[0], \
                             self._num_blocks*self._block_size, self._num_blocks*self._block_size)
                        
        del self._toa; del self._sza; del self._vza;  del self._saa
        del self._vaa; del self._aod; del self._tcwv; del self._tco3; del self._elevation

    def atm(self, p, RSR=None):
	aod, tcwv, tco3, sza, vza, raa , elevation = p
	path = '/home/ucfafyi/DATA/Multiply/6S/6SV2.1/sixsV2.1'
	s = SixS(path)
	s.altitudes.set_target_custom_altitude(elevation)
	s.altitudes.set_sensor_satellite_level()
	s.ground_reflectance = GroundReflectance.HomogeneousLambertian(GroundReflectance.GreenVegetation)
	s.geometry           = Geometry.User()
	s.geometry.solar_a   = 0
	s.geometry.solar_z   = sza
	s.geometry.view_a    = raa
	s.geometry.view_z    = vza
	s.aero_profile       = AeroProfile.PredefinedType(AeroProfile.Continental)
	s.aot550             = aod
	s.atmos_profile      = AtmosProfile.UserWaterAndOzone(tcwv, tco3)
	s.wavelength         = Wavelength(RSR)
	s.atmos_corr         = AtmosCorr.AtmosCorrLambertianFromReflectance(0.2)
	s.run()
	return s.outputs.coef_xap, s.outputs.coef_xbp, s.outputs.coef_xcp

    def _s2_block_correction_emus_xa_xb_xc(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        slice_x   = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y   = slice(j*self._block_size,(j+1)*self._block_size, 1)

        toa       = self._toa    [:,slice_x,slice_y]
        vza       = self._vza    [:,slice_x,slice_y]*np.pi/180.
        vaa       = self._vaa    [:,slice_x,slice_y]*np.pi/180.
        sza       = self._sza      [slice_x,slice_y]*np.pi/180.
        saa       = self._saa      [slice_x,slice_y]*np.pi/180.
        tcwv      = self._tcwv     [slice_x,slice_y]
        tco3      = self._tco3     [slice_x,slice_y]
        aod       = self._aod      [slice_x,slice_y]
        elevation = self._elevation[slice_x,slice_y]/1000.
        corfs = []
        for bi, band in enumerate(self._band_indexs):    
            p = [self._block_mean(i, self._mean_size).ravel() for i in [np.cos(sza), \
                 np.cos(vza[bi]), np.cos(saa - vaa[bi]), aod, tcwv, tco3, elevation]] 

            a = self.xap_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            b = self.xbp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            c = self.xcp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            a = np.repeat(np.repeat(a, self._mean_size, axis=0), self._mean_size, axis=1)
            b = np.repeat(np.repeat(b, self._mean_size, axis=0), self._mean_size, axis=1)
            c = np.repeat(np.repeat(c, self._mean_size, axis=0), self._mean_size, axis=1)
            y     = a * toa[bi] -b
            corf  = y / (1 + c*y)
            corfs.append(corf)
        boa = np.array(corfs)
        return [i, j, boa]

    def _block_mean(self, data, block_size):
        x_size, y_size = data.shape
        x_blocks       = x_size//block_size
        y_blocks       = y_size//block_size
        data           = data.copy().reshape(x_blocks, block_size, y_blocks, block_size)        
        small_data     = np.nanmean(data, axis=(3,1))
        return small_data


    def _s2_block_correction_emus_xa_xb_xc_(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        slice_x   = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y   = slice(j*self._block_size,(j+1)*self._block_size, 1)

        toa       = self._toa    [:,slice_x,slice_y]
        vza       = self._vza    [:,slice_x,slice_y]*np.pi/180.
        vaa       = self._vaa    [:,slice_x,slice_y]*np.pi/180.
        sza       = self._sza      [slice_x,slice_y]*np.pi/180.
        saa       = self._saa      [slice_x,slice_y]*np.pi/180.
        tcwv      = self._tcwv     [slice_x,slice_y]
        tco3      = self._tco3     [slice_x,slice_y]
        aod       = self._aod      [slice_x,slice_y]
        elevation = self._elevation[slice_x,slice_y]/1000.
        corfs = []
        for bi, band in enumerate(self._band_indexs):
            p = [np.cos(np.nanmean(sza.reshape())), np.cos(np.nanmean(vza[bi])), np.cos(np.nanmean([saa - \
                 vaa[bi]])), np.nanmean(aod), np.nanmean(tcwv), np.nanmean(tco3), np.nanmean(elevation)] 
            #p = [np.cos(sza).ravel(), np.cos(vza[bi]).ravel(), \
            #     np.cos(saa - vaa[bi]).ravel(), aod.ravel(), \
            #     tcwv.ravel(), tco3.ravel(), elevation.ravel()]
            a = self.xap_emus[band].predict(np.array([p,]))[0]
            b = self.xbp_emus[band].predict(np.array([p,]))[0]
            c = self.xcp_emus[band].predict(np.array([p,]))[0]
            y     = a * toa[bi] -b
            corf  = y / (1 + c*y)
            corfs.append(corf)
        boa = np.array(corfs)
        return [i, j, boa]

    def _s2_block_correction_emus(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i,j))
        slice_x   = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y   = slice(j*self._block_size,(j+1)*self._block_size, 1)

        toa       =      self._toa[:,slice_x,slice_y].reshape(self._toa.shape[0], -1)
        vza       = list(self._vza[:,slice_x,slice_y].reshape(self._vza.shape[0], -1)*np.pi/180.)
        vaa       = list(self._vaa[:,slice_x,slice_y].reshape(self._vaa.shape[0], -1))
        
        sza       = self._sza      [slice_x,slice_y].ravel()*np.pi/180.
        saa       = self._saa      [slice_x,slice_y].ravel()
        tcwv      = self._tcwv     [slice_x,slice_y].ravel()
        tco3      = self._tco3     [slice_x,slice_y].ravel()
        aod       = self._aod      [slice_x,slice_y].ravel()
        elevation = self._elevation[slice_x,slice_y].ravel()/1000.
        boa       = self.correction_engine(toa, sza, vza, saa, vaa, aod, tcwv, \
                                           tco3, elevation, self._band_indexs)
        self.corrected.append([i, j, boa])
        
    def correction_engine(self, toa, sza, vza, saa, vaa, aod, tcwv, tco3, elevation, band_indexs):
        atmos = np.array([aod, tcwv, tco3])
        boa,_ = self.s2_inv_AEE.emulator_reflectance_atmosphere(toa, atmos, sza,vza, \
                                                                 saa, vaa, elevation, bands=band_indexs)
        return np.array(boa)   

if __name__=='__main__':
    atmo_cor = atmospheric_correction(2017, 9, 4, '29SQB')
    atmo_cor.atmospheric_correction()
