#/usr/bin/env python
import sys
sys.path.insert(0,'python')
import gdal
import numpy as np
from numpy import clip, uint8
from glob import glob
import logging
import datetime 
from Py6S import *
import cPickle as pkl
from multi_process import parmap
from reproject import reproject_data
from modis_l1b_reader import MODIS_L1b_reader
from emulation_engine import AtmosphericEmulationEngine

class atmospheric_correction(object):
    '''
    A class doing the atmospheric coprrection with the input of TOA reflectance
    angles, elevation and emulators of 6S from TOA to surface reflectance.
    '''
    def __init__(self,
                 h,v,
                 doy, 
                 year, 
                 mod_l1b_dir = '/data/selene/ucfajlg/Ujia/MODIS_L1b/GRIDDED',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = '/home/ucfafyi/DATA/Multiply/emus/'):              
        
        self.year        = year
        self.doy         = doy
        self.h, self.v   = h, v
        self.mod_l1b_dir = mod_l1b_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.sur_refs    = {}
        self.date        = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
        self.month       = self.date.month
        self.day        = self.date.day
	self.logger = logging.getLogger('MODIS Atmospheric Correction')
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
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.modis_sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.modis_sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.modis_sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.modis_sensor = 'TERRA'
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        self.logger.info('Finding MODIS files.')
        modis_l1b        =  MODIS_L1b_reader(self.mod_l1b_dir, "h%02dv%02d"%(self.h,self.v),self.year)
        self.modis_files = [(i,modis_l1b.granules[i]) for i in modis_l1b.granules.keys() if i.date() == self.date.date()]
        self.logger.info('%d MODIS file(s) is(are) found for doy %04d-%03d.'%(len(self.modis_files), self.year, self.doy))
        for timestamp, modis_file in self.modis_files:
            self._doing_one_file(modis_file, timestamp)
            #break

    def _doing_one_file(self, modis_file, timestamp):
        self.logger.info('Doing %s.'%modis_file.b1.split('/')[-1].split('_EV_')[0])
        band_files  = [getattr(modis_file, 'b%d'%band) for band in range(1,8)]
        angle_files = [getattr(modis_file, ang) for ang in ['sza', 'vza', 'saa', 'vaa']]
        modis_toa   = []
        modis_angle = []
        f           = lambda fname: gdal.Open(fname).ReadAsArray()

        self.logger.info('Reading in MODIS TOA.')
        modis_toa   = parmap(f, band_files)

        self.logger.info('Reading in angles.')
        modis_angle = parmap(f, angle_files)

        scale  = np.array(modis_file.scale)
        offset = np.array(modis_file.offset)

        self.modis_toa    = np.array(modis_toa)*np.array(scale)[:,None, None] + offset[:,None, None]
        self.modis_angle  = np.array(modis_angle)/100.
        self.example_file = band_files[0]
        self.sen_time     = timestamp
        self.logger.info('Getting control variables')
        self.aod, self.tcwv, self.tco3, self.ele =  self.get_control_variables() 
        self._block_size  = 480
        self._num_blocks  = 2400/self._block_size
        self._mean_size   = 6
        self.band_indexs  = [0, 1, 2, 3, 4, 5, 6]
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self.modis_toa, self.modis_angle[0], self.modis_angle[1], self.modis_angle[2], \
                             self.modis_angle[3], self.aod, self.tcwv, self.tco3, self.ele, self.band_indexs) 

 
        
    def get_control_variables(self,):

        aod = reproject_data(self.mod_l1b_dir+'/atmo_paras/' + \
                             self.example_file.split('/')[-1].split('_EV_')[0]+'_EV_aod550.tif', self.example_file)
        aod.get_it()

        tcwv = reproject_data(self.mod_l1b_dir+'/atmo_paras/' + \
                              self.example_file.split('/')[-1].split('_EV_')[0]+'_EV_tcwv.tif', self.example_file)
        tcwv.get_it()

        tco3 = reproject_data(self.mod_l1b_dir+'/atmo_paras/' +\
                              self.example_file.split('/')[-1].split('_EV_')[0]+'_EV_tco3.tif', self.example_file)
        tco3.get_it()

        ele = reproject_data(self.global_dem, self.example_file)
        ele.get_it()
        mask = ~np.isfinite(ele.data)
        if mask.sum()>0:
           ele.data[mask] = np.interp(np.flatnonzero(mask), \
                                      np.flatnonzero(~mask), ele.data[~mask]) # simple interpolation

        return aod.data, tcwv.data, tco3.data, ele.data

    def _save_img(self, refs, bands):
        g            = gdal.Open(self.example_file)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(bands, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)
      
    def _save_band(self, band_ref, projection, geotransform):
        band, ref = band_ref
        nx, ny = ref.shape
        dst_ds = gdal.GetDriverByName('GTiff').Create(self.mod_l1b_dir+ '/sur_ref/'+\
                                self.example_file.split('/')[-1].split('_EV_')[0]+'_EV_500_SurRef_b%02d.tif'%band, ny, nx, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray(ref)
        dst_ds.FlushCache()                  
        dst_ds = None

    def _save_rgb(self, rgb, name):
        g            = gdal.Open(self.example_file)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny       = rgb.shape[:2]
        dst_ds = gdal.GetDriverByName('GTiff').Create(self.mod_l1b_dir+ '/sur_ref/'+\
                                self.example_file.split('/')[-1].split('_EV_')[0]+'_EV_500_%s_.tif'%name, ny, nx, 3, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(rgb[:,:,0])
        dst_ds.GetRasterBand(2).WriteArray(rgb[:,:,1])
        dst_ds.GetRasterBand(3).WriteArray(rgb[:,:,2])
        dst_ds.FlushCache()
        dst_ds = None

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
        ret = parmap(self._block_correction_emus_xa_xb_xc, blocks)
        self.sur_ref      = np.array([i[2] for i in ret]).reshape(self._num_blocks, self._num_blocks, toa.shape[0], \
                                      self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(toa.shape[0], \
                                      self._num_blocks*self._block_size, self._num_blocks*self._block_size)
                        
        self._save_img(self.sur_ref, [1, 2, 3, 4, 5, 6, 7])
        self.boa_rgb = clip(self.sur_ref[[0,3,2], ...].transpose(1,2,0)*255/0.25, 0,255.).astype(uint8)
        self.toa_rgb = clip(self._toa   [[0,3,2], ...].transpose(1,2,0)*255/0.25, 0,255.).astype(uint8)
        self._save_rgb(self.boa_rgb, 'BOA_RGB'); self._save_rgb(self.toa_rgb, 'TOA_RGB')
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

    def _block_correction_emus_xa_xb_xc(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        slice_x   = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y   = slice(j*self._block_size,(j+1)*self._block_size, 1)

        toa       = self._toa    [:,slice_x,slice_y]
        vza       = self._vza      [slice_x,slice_y]*np.pi/180.
        vaa       = self._vaa      [slice_x,slice_y]*np.pi/180.
        sza       = self._sza      [slice_x,slice_y]*np.pi/180.
        saa       = self._saa      [slice_x,slice_y]*np.pi/180.
        tcwv      = self._tcwv     [slice_x,slice_y]
        tco3      = self._tco3     [slice_x,slice_y]
        aod       = self._aod      [slice_x,slice_y]
        elevation = self._elevation[slice_x,slice_y]/1000.
        corfs = []
        for bi, band in enumerate(self._band_indexs):    
            p = [self._block_mean(item, self._mean_size).ravel() for item in \
                 [np.cos(sza), np.cos(vza), np.cos(saa - vaa), aod, tcwv, tco3, elevation]] 

            a = self.xap_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            b = self.xbp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            c = self.xbp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            a = np.repeat(np.repeat(a, self._mean_size, axis=0), self._mean_size, axis=1)
            b = np.repeat(np.repeat(b, self._mean_size, axis=0), self._mean_size, axis=1)
            c = np.repeat(np.repeat(c, self._mean_size, axis=0), self._mean_size, axis=1)
            y     = a * toa[bi] - b
            corf  = y / (1 + c * y)
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

if __name__=='__main__':
    atmo_cor = atmospheric_correction(17, 5, 247, 2017)
    atmo_cor.atmospheric_correction()
