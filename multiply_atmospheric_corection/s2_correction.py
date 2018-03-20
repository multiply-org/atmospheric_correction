#/usr/bin/env python
import os
import sys
sys.path.insert(0,'util')
import gdal
import psutil
import logging
import warnings
import numpy as np
from glob import glob
from numpy import clip, uint8
try:
    import cPickle as pkl
except:
    import pickle as pkl
from multi_process import parmap
from grab_s2_toa import read_s2
from reproject import reproject_data

warnings.filterwarnings("ignore")
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
                 acquisition = '0',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = './emus/',
                 satellite   = 'S2A',
                 reconstruct_s2_angle = True
                 ):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.s2_tile     = s2_tile
        self.acquisition = acquisition
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.satellite   = satellite
        self.sur_refs    = {}
        self.aero_res    = 600
        self.logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:       
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xap_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xbp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xcp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        if sys.version_info >= (3,0):
            f = lambda em: pkl.load(open(em, 'rb'), encoding = 'latin1')
        else:
            f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])    
  
    def _resample_view_angles(self,):
        f        = lambda fname: reproject_data(fname, \
                                                self.example_file, srcNodata = -32767, \
                                                outputType= gdal.GDT_Float32, xRes=self.aero_res, \
                                                yRes=self.aero_res, resample = gdal.GRIORA_NearestNeighbour).data / 100.
        ret      = np.array(parmap(f, self.va_files))
        self.vaa, self.vza = ret[:, 0], ret[:, 1]
        #for i,j in enumerate(self.vaa):
        #    mask = ~np.isfinite(self.vaa[i])
        #    self.vaa[i][mask] = np.interp(np.flatnonzero(mask), \
        #                                  np.flatnonzero(~mask), self.vaa[i][~mask]).astype(float)
        #for i,j in enumerate(self.vza):                                                           
        #    mask = ~np.isfinite(self.vza[i])                                                      
        #    self.vza[i][mask] = np.interp(np.flatnonzero(mask), \
        #                                  np.flatnonzero(~mask), self.vza[i][~mask]).astype(float)
    
    def  _get_control_variables(self,):

        fnames = [self.s2_file_dir+'/%s.tif'%i for i in ['aot_unc', 'tcwv_unc', 'tco3_unc']]
        f      = lambda fname: reproject_data(fname, \
                                              self.example_file, \
                                              outputType=gdal.GDT_Float32, xRes=self.aero_res, \
                                              yRes=self.aero_res, resample = gdal.GRIORA_NearestNeighbour).data 
        self.aot_unc, self.tcwv_unc, self.tco3_unc = parmap(f, fnames) 
       
        fnames = [self.s2_file_dir+'/%s.tif'%i for i in ['aot', 'tcwv', 'tco3']] + [self.sa_files, self.global_dem]
        f      = lambda fname: reproject_data(fname, \
                                              self.example_file, \
                                              outputType= gdal.GDT_Float32, xRes=self.aero_res, \
                                              yRes=self.aero_res, resample = gdal.GRIORA_Bilinear).data 
        self.aot, self.tcwv, self.tco3, [self.saa, self.sza], self.ele = parmap(f, fnames)
        self.saa, self.sza = self.saa/100., self.sza/100.

        mask = ~np.isfinite(self.ele)
        self.ele[mask] = np.interp(np.flatnonzero(mask), \
                                   np.flatnonzero(~mask), self.ele[~mask]).astype(float)

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.s2_sensor = 'MSI'
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        s2   = read_s2(self.s2_toa_dir, self.s2_tile, \
                       self.year, self.month, self.day, bands=None, acquisition = self.acquisition)
        self.s2_file_dir  = s2.s2_file_dir
        self.example_file = s2.s2_file_dir+'/B04.jp2'

        self.logger.info('Reading in the reflectance.')
        all_refs = s2.get_s2_toa()

        self.logger.info('Getting the angle files.')
        s2.get_s2_angles()                                                                                                                                        
        self.sa_files = self.s2_file_dir + '/angles/SAA_SZA.tif'

        self.logger.info('Getting control variables.')        
        self._get_control_variables()

        self.logger.info('Doing 10 meter bands')
        self._10_meter_bands = ['B02', 'B03', 'B04', 'B08']
        self.toa  = np.array([all_refs[band].astype(float)/10000. for band in self._10_meter_bands])
        self.mask = (self.toa > 0).all(axis=0)
        self.mask = self._block_mean(self.mask.astype(float), self.aero_res / 10) > 0 
 
        self.logger.info('Reading in view angles.')
        self.va_files = [self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%i for i in self._10_meter_bands]
        self._resample_view_angles()

        self._block_size = 3660
        self._num_blocks = int(10980/self._block_size)
        self._control_varibales_size = int(109800. / self.aero_res / self._num_blocks)
        self._mean_size  = 60
        self._band_indexs = [1, 2, 3, 7]        
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        boa, unc = self.fire_correction()
        
        self.scale   = 0.25 
        self.toa_rgb = clip(self.toa[[2,1,0], ...].transpose(1,2,0)*255/self.scale, 0., 255.).astype(uint8)
        self.boa_rgb = clip(boa     [[2,1,0], ...].transpose(1,2,0)*255/self.scale, 0., 255.).astype(uint8)
        self._save_rgb(self.toa_rgb, 'TOA_RGB.tif', self.example_file)
        self._save_rgb(self.boa_rgb, 'BOA_RGB.tif', self.example_file)
        
        gdal.Translate(self.s2_file_dir+'/TOA_overview.jpg', self.s2_file_dir+'/TOA_RGB.tif', \
                       format = 'JPEG', widthPct=50, heightPct=50, resampleAlg=gdal.GRA_Bilinear ).FlushCache()
        gdal.Translate(self.s2_file_dir+'/BOA_overview.jpg', self.s2_file_dir+'/BOA_RGB.tif', \
                       format = 'JPEG', widthPct=50, heightPct=50, resampleAlg=gdal.GRA_Bilinear ).FlushCache()
        
        #self.sur_refs.update(dict(zip(['B02', 'B03', 'B04', 'B08'], self.boa)))
        fnames = [i + '_sur' for i in self._10_meter_bands] + [i+'_sur_unc' for i in self._10_meter_bands]
        self._save_img(list(boa) + list(unc), fnames); del boa; del unc
        for band in self._10_meter_bands:
            all_refs[band] = None 
  
        self.logger.info('Doing 20 meter bands')
        self._20_meter_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.toa = np.array([all_refs[band].astype(float)/10000. for band in self._20_meter_bands])

        self.logger.info('Reading in view angles.')
        self.va_files = [self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%i for i in self._20_meter_bands]
        self._resample_view_angles()
        self._block_size = 1830     
        self._num_blocks = int(5490/self._block_size)                           
        self._mean_size  = 30        
        self._band_indexs = [4, 5, 6, 8, 11, 12]                                         
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        boa, unc = self.fire_correction()
        fnames = [i + '_sur' for i in self._20_meter_bands] + [i+'_sur_unc' for i in self._20_meter_bands]
        self._save_img(list(boa) + list(unc), fnames); del boa; del unc
        #self.sur_refs.update(dict(zip(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], self.boa)))
        for band in self._20_meter_bands:
            all_refs[band] = None

        self.logger.info('Doing 60 meter bands')
        self._60_meter_bands = ['B01', 'B09', 'B10']
        self.toa = np.array([all_refs[band].astype(float)/10000. for band in self._60_meter_bands])

        self.logger.info('Reading in view angles.')                                                 
        self.va_files = [self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%i for i in self._60_meter_bands]
        self._resample_view_angles()                                                                
        self._block_size = 610                                                                     
        self._num_blocks = int(1830/self._block_size)                                               
        self._mean_size  = 10
        self._band_indexs = [0, 9, 10]                                                    
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)         
        boa, unc = self.fire_correction()                                                           
        fnames = [i + '_sur' for i in self._60_meter_bands] + [i+'_sur_unc' for i in self._60_meter_bands]
        self._save_img(list(boa) + list(unc), fnames); del boa; del unc
        #self.sur_refs.update(dict(zip(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], self.boa)))      
        for band in self._60_meter_bands:
            all_refs[band] = None

        self.logger.info('Done!')

    def _save_rgb(self, rgb_array, name, source_image):
        g            = gdal.Open(source_image)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny       = rgb_array.shape[:2]
        outputFileName = self.s2_file_dir+'/%s'%name
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 3, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=JPEG"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(rgb_array[:,:,0])
        dst_ds.GetRasterBand(1).SetScale(self.scale)
        dst_ds.GetRasterBand(2).WriteArray(rgb_array[:,:,1])
        dst_ds.GetRasterBand(2).SetScale(self.scale)
        dst_ds.GetRasterBand(3).WriteArray(rgb_array[:,:,2])
        dst_ds.GetRasterBand(3).SetScale(self.scale)
        dst_ds.FlushCache()
        dst_ds = None

    def rgb_scale_offset(arr):
        arrmin = arr.mean() - 3*arr.std()
        arrmax = arr.mean() + 3*arr.std()
        arrlen = arrmax-arrmin
        scale  = arrlen/255.0
        offset = arrmin
        return offset, scale

    def _save_img(self, refs, bands):
        g            = gdal.Open(self.s2_file_dir+'/%s.jp2'%bands[0][:3])
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(bands, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)
      
    def _save_band(self, band_ref, projection, geotransform):
        band, ref = band_ref
        nx, ny = ref.shape
        outputFileName = self.s2_file_dir+'/%s.tif'%band
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        sur = ref * 10000
        sur[~(sur>=0)] = -9999
        sur = sur.astype(np.int16)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(1).WriteArray(sur)
        dst_ds.FlushCache()                  
        dst_ds = None

    def get_control_variables(self, target_band):
        fnames   = [self.s2_file_dir+'/%s.tif'%i for i in ['aot', 'tcwv', 'tco3', 'aot_unc', 'tcwv_unc', 'tco3_unc']]
        fnames.append(self.global_dem)
        f        = lambda fname: reproject_data(fname, self.s2_file_dir + '/B04.jp2', outputType= gdal.GDT_Float32, xRes=600, yRes=600).data
        aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, ele = parmap(f, fnames)
        mask = ~np.isfinite(ele)
        ele[mask] = np.interp(np.flatnonzero(mask), \
                              np.flatnonzero(~mask), ele[~mask])
        return aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, ele.astype(float)


    def fire_correction(self):
        rows              = np.repeat(np.arange(self._num_blocks), self._num_blocks)
        columns           = np.tile(np.arange(self._num_blocks), self._num_blocks)
        blocks            = list(zip(rows, columns))
        band_ram          = 2.5e10 / (60. / self._mean_size)
        av_ram = psutil.virtual_memory().available 
        procs = np.min([int(len(blocks) * (av_ram / band_ram) / self.toa.shape[0]), psutil.cpu_count(), len(blocks)])
        if procs == 0:
            raise MemoryError('To few RAM can be used and minimum RAM is 14GB')
        #self._s2_block_correction_emus_xa_xb_xc([0,0])
        ret = parmap(self._s2_block_correction_emus_xa_xb_xc, blocks, procs)
        boa = np.array([i[2] for i in ret]).reshape(self._num_blocks, self._num_blocks, self.toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(self.toa.shape[0], \
                             self._num_blocks*self._block_size, self._num_blocks*self._block_size)

        unc = np.array([i[3] for i in ret]).reshape(self._num_blocks, self._num_blocks, self.toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(self.toa.shape[0], \
                             self._num_blocks*self._block_size, self._num_blocks*self._block_size)                
        return boa, unc

    def _s2_block_correction_emus_xa_xb_xc(self, block):
        i, j        = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        slice_x     = slice(i*self._block_size,            (i+1)*self._block_size, 1)
        slice_y     = slice(j*self._block_size,            (j+1)*self._block_size, 1)
        slice_c_x   = slice(i*self._control_varibales_size,(i+1)*self._control_varibales_size, 1)
        slice_c_y   = slice(j*self._control_varibales_size,(j+1)*self._control_varibales_size, 1)

        toa         = self.toa  [:, slice_x,   slice_y  ]
        vza         = self.vza  [:, slice_c_x, slice_c_y]*np.pi/180.
        vaa         = self.vaa  [:, slice_c_x, slice_c_y]*np.pi/180.
        sza         = self.sza     [slice_c_x, slice_c_y]*np.pi/180.
        saa         = self.saa     [slice_c_x, slice_c_y]*np.pi/180.
        aot         = self.aot     [slice_c_x, slice_c_y]
        tcwv        = self.tcwv    [slice_c_x, slice_c_y]
        tco3        = self.tco3    [slice_c_x, slice_c_y]
        aot_unc     = self.aot_unc [slice_c_x, slice_c_y]
        tcwv_unc    = self.tcwv_unc[slice_c_x, slice_c_y]
        tco3_unc    = self.tco3_unc[slice_c_x, slice_c_y]
        elevation   = self.ele     [slice_c_x, slice_c_y]/1000.
        mask        = self.mask    [slice_c_x, slice_c_y]
        if mask.sum() != 0:
            xap, xap_dH = [], []
            xbp, xbp_dH = [], []
            xcp, xcp_dH = [], []
            xps = [xap, xbp, xcp]
            xhs = [xap_dH, xbp_dH, xcp_dH]
            for bi, band in enumerate(self._band_indexs):    
                p        = np.array([np.cos(sza), np.cos(vza[bi]), np.cos(saa - vaa[bi]), aot, tcwv, tco3, elevation])
                mp       = p[:, mask]
                for ei, emu in enumerate([self.xap_emus, self.xbp_emus, self.xcp_emus]):
                    temp1, temp2 = np.zeros_like(p[0]), np.zeros(p[0].shape + (3,))
                    H, _, dH     = emu[band].predict(mp.T, do_unc=True)
                    temp1[mask]  = H
                    temp2[mask]  = np.array(dH)[:, 3:6]
                    #temp1        = temp1.reshape(self._block_size//self._mean_size, self._block_size//self._mean_size)
                    temp1        = np.repeat(np.repeat(temp1, self._mean_size, axis=0), self._mean_size, axis=1)
                    if ei == 0:
                        temp1[temp1==0]  = 1.
                    #temp2        = temp2.reshape(self._block_size//self._mean_size, self._block_size//self._mean_size, 3)
                    temp2        = np.repeat(np.repeat(temp2, self._mean_size, axis=0), self._mean_size, axis=1)
                    xps[ei].append(temp1)
                    xhs[ei].append(temp2)

            xap_H, xbp_H, xcp_H    = np.array(xps)
            xap_dH, xbp_dH, xcp_dH = np.array(xhs)

            y            = xap_H * toa - xbp_H
            boa          = y / (1 + xcp_H * y)           

            dH           = -1 * (-toa[...,None] * xap_dH - \
                            2 * toa[...,None] * xap_H[...,None] * xbp_H[...,None] * xcp_dH + \
                            toa[...,None]**2 * xap_H[...,None]**2 * xcp_dH + \
                            xbp_dH + \
                            xbp_H[...,None]**2 * xcp_dH) / \
                            (toa[...,None] * xap_H[...,None] * xcp_H[...,None] - \
                            xbp_H[...,None] * xcp_H[...,None] + 1)**2
            
            aot_dH, tcwv_dH, tco3_dH = [ dH[:, :, :,i] for i in range(3)]
            toa_dH = xap_H / (xcp_H*(toa * xap_H - xbp_H) + 1)**2

            aot_unc  = np.repeat(np.repeat(aot_unc,  self._mean_size, axis=0), self._mean_size, axis=1)
            tcwv_unc = np.repeat(np.repeat(tcwv_unc, self._mean_size, axis=0), self._mean_size, axis=1) 
            tco3_unc = np.repeat(np.repeat(tco3_unc, self._mean_size, axis=0), self._mean_size, axis=1)

            unc = np.sqrt(aot_dH ** 2 * aot_unc**2 + tcwv_dH ** 2 * tcwv_unc**2 + tco3_dH ** 2 * tco3_unc**2 + toa_dH**2 * 0.015**2) 
        else:
            boa    = toa
            unc    = np.zeros_like(toa)
            unc[:] = -0.9999
        return [i, j, boa, unc]

    def _block_mean(self, data, block_size):
        x_size, y_size = data.shape
        x_blocks       = x_size//block_size
        y_blocks       = y_size//block_size
        data           = data.copy().reshape(int(x_blocks), int(block_size), int(y_blocks), int(block_size))
        small_data     = np.nanmean(data, axis=(3,1))
        return small_data

if __name__=='__main__':
    atmo_cor = atmospheric_correction(2016, 11, 16, '50SLG', s2_toa_dir='/data/nemesis/S2_data/')
    atmo_cor.atmospheric_correction()
