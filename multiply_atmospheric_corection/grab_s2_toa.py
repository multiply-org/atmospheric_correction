#!/usr/bin/env python
import gdal
import os
import sys
import copy
sys.path.insert(0, 'util')
sys.path.insert(0, './')
import xml.etree.ElementTree as ET
import numpy as np
import pickle as pkl
from multiprocessing import Pool
from glob import glob
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
from s2_Angle_resample import resample_s2_angles
from reproject import reproject_data
from multi_process import parmap
from skimage.morphology import disk, binary_dilation, binary_erosion

def read_s2_band(fname):
        g = gdal.Open(fname)
        if g is None:
            raise IOError
        else:
            return g.ReadAsArray()


class read_s2(object):
    '''
    A class reading S2 toa reflectance, taken the directory, date and bands needed,
    It will read in the cloud mask as well, if no cloud.tiff, then call the classification
    algorithm to get the cloud mask and save it.
    '''
    def __init__(self, 
                 s2_toa_dir,
                 s2_tile, 
                 year, month, day,
                 acquisition = '0',
                 bands   = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A']):
        self.s2_toa_dir  = s2_toa_dir
        self.s2_tile     = s2_tile
        self.year        = year
        self.month       = month
        self.day         = day
        self.bands       = bands # selected bands
        self.s2_bands    = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', \
                           'B08','B8A', 'B09', 'B10', 'B11', 'B12' #all bands
        self.s2_file_dir = os.path.join(self.s2_toa_dir, self.s2_tile[:-3],\
                                        self.s2_tile[-3], self.s2_tile[-2:],\
                                        str(self.year), str(self.month), str(self.day), acquisition)
        self.selected_img = None
        self.done         = False

    def _read_all(self, done = False):
        fname     = [self.s2_file_dir+'/%s.jp2'%i for i in self.s2_bands]
        pool      = Pool(processes=len(fname))
        ret       = pool.map(read_s2_band, fname)    
        self.imgs = dict(zip(self.s2_bands, ret))
        self.done = True

    def get_s2_toa(self,vrt = False):
        self._read_all(self.done)
        if self.bands is None:
            self.bands = self.s2_bands
        selc_imgs = [self.imgs[band] for band in self.bands] 
        return dict(zip(self.bands, selc_imgs))

    def get_s2_cloud(self,):
        if len(glob(self.s2_file_dir+'/cloud.tif'))==0:
            self._read_all(self.done)  
            cloud_bands  = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
            ratio        = [ 6,     1,     1,     2,     1,     2,     6,     6,     2,     2]
            
            refs         = [(self.imgs[band]/10000.).reshape(1830, 6//ratio[i], 1830,  6//ratio[i]) for i, band in enumerate(cloud_bands)]
            refs         = np.array([ref.sum(axis=(3,1)) / (ref>=0.0001).sum(axis=(3,1)) for ref in refs])
            classifier   = pkl.load(open('./data/sen2cloud_detector.pkl', 'rb'))
            mask         = np.all(refs >= 0.0001, axis=0)
            cloud_probs  = classifier.predict_proba(refs[:, mask].T)[:,1]
            cloud        = np.zeros((1830, 1830))
            cloud[mask]  = cloud_probs
            cloud_mask   = cloud > 0.90
            cloud_mask   = binary_erosion (cloud_mask, disk(2))
            self.cloud   = binary_dilation(cloud_mask, disk(3))
            self.cloud   = np.repeat(np.repeat(self.cloud,6, axis = 0), 6, axis = 1)
            g            = gdal.Open(self.s2_file_dir+'/B01.jp2')
            driver       = gdal.GetDriverByName('GTiff')
            g1           = driver.Create(self.s2_file_dir+'/cloud.tif', \
                                         g.RasterXSize, g.RasterYSize, 1, \
                                         gdal.GDT_Byte,  options=["TILED=YES", "COMPRESS=DEFLATE"])
            g1.SetGeoTransform(g.GetGeoTransform())
            g1.SetProjection  (g.GetProjection()  )
            g1.GetRasterBand(1).WriteArray((cloud * 100).astype(int))
            g1=None; g=None
        else:
            cloud = gdal.Open(self.s2_file_dir+\
                             '/cloud.tif').ReadAsArray()
            cloud_mask   = cloud > 89 #rounding issue                           
            cloud_mask   = binary_erosion (cloud_mask, disk(2))     
            self.cloud   = binary_dilation(cloud_mask, disk(3)) 
            self.cloud   = np.repeat(np.repeat(self.cloud,6, axis = 0), 6, axis = 1)
        try:
            mask = self.imgs['B04'] >= 1.
        except:
            mask = gdal.Open(self.s2_file_dir + '/B04.jp2').ReadAsArray() >= 1.
        self.cloud_cover = 1. * self.cloud.sum() / mask.sum()

        return self.cloud
        
    def get_s2_angles(self):
        if len(glob(self.s2_file_dir + '/angles/*.tif')) == 14:
            self.saa_sza = [self.s2_file_dir + '/angles/SAA_SZA.tif']
            self.vaa_vza = [self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%i for i in self.bands]
        else:
            resample_s2_angles(self.s2_file_dir)
            self.saa_sza = [self.s2_file_dir + '/angles/SAA_SZA.tif']
            self.vaa_vza = [self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%i for i in self.bands]
if __name__ == '__main__':
    s2 = read_s2('/store/S2_data/', '50SMH', \
                  2017, 10, 12, bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] )
    cm = s2.get_s2_cloud()
    s2.get_s2_angles()
