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
from grab_l8_toa import read_l8
from multi_process import parmap
from reproject import reproject_data
from scipy.interpolate import griddata

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
                 l8_tile,
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/LC08_L1TP_123034_20170710_20170725_01_T1/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = './emus/',
                 ):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.l8_tile     = l8_tile
        self.l8_toa_dir  = l8_toa_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.bands       = [1, 2, 3, 4, 5, 6, 7]
        self.aero_res    = 900
        self.logger = logging.getLogger('Landsat 8 Atmospheric Correction')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:           
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xap.pkl'%(self.sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xbp.pkl'%(self.sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_correction_xcp.pkl'%(self.sensor))[0]
        #f = lambda em: pkl.load(open(em, 'rb'))
        if sys.version_info >= (3,0):
            f = lambda em: pkl.load(open(em, 'rb'), encoding = 'latin1')
        else:
            f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.sensor = 'OLI'
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        l8   = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        self.l8_header = l8.header
        self.example_file = glob(self.l8_toa_dir + '/%s_[b, B]%d.[t, T][i, I][f, F]'%(l8.header, 1))[0]

        self.logger.info('Reading in the reflectance.')
        self.toa = l8._get_toa()

        self.logger.info('Reading in control_vriables.')
        self._get_control_variables(l8)
        self._sorting_data()

        self.shape = self.toa.shape[1:3]
        self._block_size = 3000
        self._num_blocks_x, self._num_blocks_y = int(np.ceil(1. * self.shape[0] / self._block_size)), int(np.ceil(1. * self.shape[1] / self._block_size))
        rows              = np.repeat(np.arange(self._num_blocks_x), self._num_blocks_y)
        columns           = np.tile  (np.arange(self._num_blocks_y), self._num_blocks_x)
        blocks            = list(zip(rows, columns))
        self.mask =  gdal.Open(glob(self.l8_toa_dir + '/%s_[b, B][q, Q][a, A].[T, t][I, i][F, f]'%l8.header)[0]).ReadAsArray() == 1

        self.logger.info('Doing correction.')
        av_ram = psutil.virtual_memory().available 
        block_ram = 16 * 1e9
        procs = np.min([int(av_ram / block_ram), psutil.cpu_count(), len(blocks)])
        if procs == 0:
            raise MemoryError('To few RAM can be used and minimum RAM is 16GB')

        ret = parmap(self._block_correction_emus_xa_xb_xc, blocks, procs)

        self.boa = np.array([i[2] for i in ret]).reshape(self._num_blocks_x, self._num_blocks_y, self.toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(self.toa.shape[0], \
                             self._num_blocks_x*self._block_size, self._num_blocks_y*self._block_size)[:, : self.shape[0], : self.shape[1]]
        
        self.unc = np.array([i[3] for i in ret]).reshape(self._num_blocks_x, self._num_blocks_y, self.toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(self.toa.shape[0], \
                             self._num_blocks_x*self._block_size, self._num_blocks_y*self._block_size)[:, : self.shape[0], : self.shape[1]]

        self.boa[:, self.mask] = np.nan
        self.toa[:, self.mask] = np.nan
        self.unc[:, self.mask] = np.nan
        self.scale   = 0.25 #np.percentile(self.boa[3][self.boa[3] > 0], 95)
        self.boa_rgb = np.clip(self.boa[[3,2,1]].transpose(1,2,0) * 255 / self.scale, 0, 255).astype(uint8)
        self.toa_rgb = np.clip(self.toa[[3,2,1]].data.transpose(1,2,0) * 255 / self.scale, 0, 255).astype(uint8)

        self.logger.info('Saving corrected results')
        self._save_rgb(self.toa_rgb, 'TOA_RGB', self.example_file)
        self._save_rgb(self.boa_rgb, 'BOA_RGB', self.example_file)
        fnames = ['b%s_sur'%i for i in self.bands] + ['b%s_sur_unc'%i for i in self.bands]
        self._save_img(list(self.boa) + list(self.unc), fnames)

        gdal.Translate(self.l8_toa_dir + '/%s_%s'%(self.l8_header, 'BOA_overview.jpg'), \
                       self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, 'BOA_RGB'), format = \
                       'JPEG', widthPct=50, heightPct=50, resampleAlg='cubic' ).FlushCache()

        gdal.Translate(self.l8_toa_dir + '/%s_%s'%(self.l8_header, 'TOA_overview.jpg'), \
                       self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, 'TOA_RGB'), format = \
                       'JPEG', widthPct=50, heightPct=50, resampleAlg='cubic' ).FlushCache()
        self.logger.info('Done!')

    def _get_control_variables(self, l8):

        fnames = [self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, i) for i in ['aot_unc', 'tcwv_unc', 'tco3_unc']] + l8.saa_sza
        f      = lambda fname: reproject_data(fname, \
                                              self.example_file, \
                                              xRes=self.aero_res, \
                                              yRes=self.aero_res, \
                                              srcNodata = [-32768, 0],\
                                              outputType=gdal.GDT_Float32,\
                                              resample = gdal.GRIORA_NearestNeighbour).data
        self.aot_unc, self.tcwv_unc, self.tco3_unc, sas = parmap(f, fnames)
        vas = np.array(parmap(f, l8.vaa_vza))
        self.vaa, self.vza = vas[:, 0] / 100., vas[:, 1] / 100.
        self.vaa[self.vaa<-180] = np.nan
        self.vza[self.vza<=0]   = np.nan
        self.saa, self.sza      = sas / 100.
        self.saa[self.saa<-180] = np.nan
        self.sza[self.sza<=0]   = np.nan        

        fnames = [self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, i) for i in ['aot', 'tcwv', 'tco3']] + [self.global_dem, ]
        f      = lambda fname: reproject_data(fname, \
                                              self.example_file, \
                                              outputType= gdal.GDT_Float32, xRes=self.aero_res, \
                                              yRes=self.aero_res, resample = gdal.GRIORA_Bilinear).data
        self.aot, self.tcwv, self.tco3, self.ele = parmap(f, fnames)
        self.ele = self.ele / 1000.

    def _fill_nan(self, array):                                                                                                                         
        x_shp, y_shp = array.shape
        mask  = ~np.isnan(array)
        valid = np.array(np.where(mask)).T
        value = array[mask]
        mesh  = np.repeat(range(x_shp), y_shp).reshape(x_shp, y_shp), \
                np.tile  (range(y_shp), x_shp).reshape(x_shp, y_shp)
        array = griddata(valid, value, mesh, method='nearest')
        return array

    def _sorting_data(self,):
        self.vaa = np.array(parmap(self._fill_nan, list(self.vaa)))
        self.vza = np.array(parmap(self._fill_nan, list(self.vza)))
        self.saa, self.sza, self.ele = parmap(self._fill_nan, [self.saa, self.sza, self.ele])
        self.vaa, self.vza, self.saa, self.sza = map(np.deg2rad, [self.vaa, self.vza, self.saa, self.sza])

    def _save_rgb(self, rgb_array, name, source_image):
        g            = gdal.Open(source_image)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny = rgb_array.shape[:2]
        outputFileName = self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, name)
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
        outputFileName = self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, band)
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)    
        dst_ds.SetProjection(projection) 
        sur = ref * 10000
        sur[np.isnan(sur)] = -9999
        sur = sur.astype(np.int16)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(1).WriteArray(sur)
        dst_ds.FlushCache()                  
        dst_ds = None

    def _block_helper(self, val_res, block):
        val, res = val_res
        i, j       = block
        res_scale  = 30. / res           
        slice_size = int(self._block_size * res_scale)
        slice_x    = slice(i * slice_size, (i+1) * slice_size, 1)
        slice_y    = slice(j * slice_size, (j+1) * slice_size, 1)
        if   val.ndim == 2:             
            temp    = np.zeros((slice_size, slice_size))
            temp[:] = np.nan            
            temp  [ : min((i+1) * slice_size, val.shape[-2]) - i * slice_size, \
                    : min((j+1) * slice_size, val.shape[-1]) - j * slice_size] = val[slice_x,slice_y]
        elif val.ndim == 3:             
            temp    = np.zeros((val.shape[0], slice_size, slice_size))
            temp[:] = np.nan            
            temp  [:, : min((i+1) * slice_size, val.shape[-2]) - i * slice_size, \
                      : min((j+1) * slice_size, val.shape[-1]) - j * slice_size] = val[:, slice_x,slice_y]

        return temp

    def _block_correction_emus_xa_xb_xc(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        vals = self.toa, self.mask, self.vza, self.vaa, self.sza, self.saa, self.aot, self.tcwv, self.tco3, self.aot_unc, self.tcwv_unc, self.tco3_unc, self.ele
        ress = [30., 30.] +  [self.aero_res, ] * (len(vals) - 1)
        val_ress = zip(vals, ress)
        f = lambda val_res: self._block_helper(val_res, block)
        toa, mask, vza, vaa, sza, saa, aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, ele = map(f, val_ress)
        p = [np.cos(sza), None, None, aot, tcwv, tco3, ele]
        raa = np.cos(saa - vaa)
        vza = np.cos(vza)
        c_size = int(self._block_size * 30. / self.aero_res)
        m_size = int(self.aero_res / 30.)
        mask = mask.astype(bool)
        if (~mask).sum() != 0:                    
            xap, xap_dH = [], []               
            xbp, xbp_dH = [], []               
            xcp, xcp_dH = [], []               
            xps = [xap, xbp, xcp]              
            xhs = [xap_dH, xbp_dH, xcp_dH]     
            for bi, band in enumerate(self.bands):    
                p[1:3] = vza[bi], raa[bi]         
                X = np.array(p).reshape(7, -1)
                for ei, emu in enumerate([self.xap_emus, self.xbp_emus, self.xcp_emus]):
                    H, _, dH     = emu[bi].predict(X.T, do_unc=True)
                    temp1 = H.reshape(c_size, c_size)           
                    temp2 = np.array(dH)[:, 3:6].reshape(c_size, c_size, 3)
                    temp1        = np.repeat(np.repeat(temp1, m_size, axis=0), m_size, axis=1)
                    temp2        = np.repeat(np.repeat(temp2, m_size, axis=0), m_size, axis=1)
                    xps[ei].append(temp1)      
                    xhs[ei].append(temp2)      
                                               
            xap_H, xbp_H, xcp_H    = np.array(xps)
            xap_dH, xbp_dH, xcp_dH = np.array(xhs)
                                               
            y   = xap_H * toa - xbp_H 
            boa = y / (1 + xcp_H * y)           
                                               
            dH  = -1 * (-toa[...,None] * xap_dH - \
                  2 * toa[...,None] * xap_H[...,None] * xbp_H[...,None] * xcp_dH + \
                  toa[...,None]**2 * xap_H[...,None]**2 * xcp_dH + \
                  xbp_dH + \
                  xbp_H[...,None]**2 * xcp_dH) / \
                  (toa[...,None] * xap_H[...,None] * xcp_H[...,None] - \
                  xbp_H[...,None] * xcp_H[...,None] + 1)**2
                                               
            aot_dH, tcwv_dH, tco3_dH = [ dH[:, :, :,i] for i in range(3)]
            toa_dH = xap_H / (xcp_H*(toa * xap_H - xbp_H) + 1)**2                                  

            aot_unc  = np.repeat(np.repeat(aot_unc,  m_size, axis=0), m_size, axis=1)
            tcwv_unc = np.repeat(np.repeat(tcwv_unc, m_size, axis=0), m_size, axis=1) 
            tco3_unc = np.repeat(np.repeat(tco3_unc, m_size, axis=0), m_size, axis=1)
            unc = np.sqrt(aot_dH ** 2 * aot_unc**2 + tcwv_dH ** 2 * tcwv_unc**2 + tco3_dH ** 2 * tco3_unc**2 + toa_dH**2 * 0.015**2) 
        else:                                  
            boa    = toa                       
            unc    = np.zeros_like(toa)        

        unc[:, mask] = -0.9999    
        boa[:, mask] = 0.
        return [i, j, boa, unc]
 
if __name__=='__main__':
    atmo_cor = atmospheric_correction(2017, 7, 10, (123, 34),)
    atmo_cor.atmospheric_correction()
