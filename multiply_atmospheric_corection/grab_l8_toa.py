#/usr/bin/env python
import os
import sys
sys.path.insert(0, 'util')
import gdal
import numpy as np
from glob import glob
import subprocess
from datetime import datetime
from multi_process import parmap

def gdal_reader(fname):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        return g.ReadAsArray()
 
def update_vaa(fname):
    # view azimuth angle is not usable for Landsat 8
    # and mean values are used
    ds   = gdal.Open(fname, gdal.GA_Update)
    data = ds.GetRasterBand(1).ReadAsArray()
    data = np.ma.array(data, mask = ((data > 18000) | (data < -18000)))
    data[data<0]     = data[data<0] + 18000
    data[~data.mask] = int(np.nanmean(data[~data.mask]))
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

class read_l8(object):
    '''
    read in the l8 datasets, toa and angles.
    '''
    def __init__(self,
                 toa_dir,
                 tile,
                 year,
                 month,
                 day,
                 bands = None,
                 angle_exe = './util/l8_angles/l8_angles'
                ):
        self.toa_dir   = toa_dir
        self.tile      = tile
        self.year      = year
        self.month     = month
        self.day       = day
        if bands is None:
            self.bands = np.arange(1, 8)
        else:
            self.bands = np.array(bands)
        self.angle_exe = os.path.abspath(angle_exe)
        composite      = glob(self.toa_dir + '/LC08_L1TP_%03d%03d_%04d%02d%02d_*_01_??_[B, b]1.[T, t][I, i][F, f]' \
                         % ( self.tile[0], self.tile[1], self.year, self.month, self.day))[0].split('/')[-1].split('_')[:-1]
        self.header    = '_'.join(composite)    
        self.toa_file  = [glob(self.toa_dir + '/%s_[b, B]%d.[T, t][I, i][F, f]'%(self.header, i))[0] for i in self.bands]
        self.mete_file =  glob(self.toa_dir + '/%s_[m, M][t, T][l, L].[t, T][x, X][t, T]'%self.header)[0]
        self.qa_file   =  glob(self.toa_dir + '/%s_[b, B][q, Q][a, A].[T, t][I, i][F, f]'%self.header)[0]
        try:
            self.saa_sza =  glob(self.toa_dir + '/%s_solar_B%02d.tif' %(self.header, 1))
            self.vaa_vza = [glob(self.toa_dir + '/%s_sensor_B%02d.tif'%(self.header, i))[0] for i in self.bands]
        except:
            ang_file     = glob(self.toa_dir + '/%s_[a, A][n, N][g, G].[t, T][x, X][t, T]'%self.header)[0]
            cwd = os.getcwd()
            if not os.path.exists(self.angle_exe):
                os.chdir('./util/l8_angles/') 
                subprocess.call('make') 
            os.chdir(cwd)             
            os.chdir(self.toa_dir)
            ang_file = ang_file.split('/')[-1]
            f        =  lambda band: subprocess.call([self.angle_exe, ang_file, 'BOTH', '1', '-f', '-32768', '-b', str(band)])
            parmap(f, np.arange(1, 8))
            os.chdir(cwd)
            self.saa_sza = []
            #self.vaa_vza = []
            f       = lambda fnames: gdal.Translate(fnames[0], fnames[1], creationOptions = ['COMPRESS=DEFLATE', 'TILED=YES']).FlushCache()
            fnames  = [(self.toa_dir + '/%s_sensor_B%02d.tif'%(self.header, i), \
                        self.toa_dir + '/%s_sensor_B%02d.img'%(self.header, i)) for i in range(1, 8)]
            fnames += [(self.toa_dir + '/%s_solar_B%02d.tif' %(self.header, 1), 
                        self.toa_dir + '/%s_solar_B%02d.img' %(self.header, 1)),]        
            sas     = [self.toa_dir  + '/%s_sensor_B%02d.img'%(self.header, i) for i in range(1, 8)]
            parmap(update_vaa, sas)
            parmap(f, fnames)
            #for j, i in enumerate(np.arange(1, 8)):
                #if j==0:
                    #gdal.Translate(self.toa_dir + '/%s_solar_B%02d.tif' %(self.header, i), \
                    #               self.toa_dir + '/%s_solar_B%02d.img' %(self.header, i), \
                    #               creationOptions = ['COMPRESS=LZW', 'TILED=YES']).FlushCache()
                #gdal.Translate(self.toa_dir + '/%s_sensor_B%02d.tif'%(self.header, i), \
                #               self.toa_dir + '/%s_sensor_B%02d.img'%(self.header, i), \
                #               creationOptions = ['COMPRESS=LZW', 'TILED=YES']).FlushCache()
             #   [os.remove(f) for f in glob(self.toa_dir + '/%s_sensor_B%02d.img*'%(self.header, i))]
             #   [os.remove(f) for f in glob(self.toa_dir + '/%s_solar_B%02d.img*'%(self.header, i))]
            [os.remove(f) for f in glob(self.toa_dir + '/%s_s*_*.img*'%self.header)]
            self.vaa_vza = [self.toa_dir + '/%s_sensor_B%02d.tif'%(self.header, i) for i in self.bands]
            self.saa_sza = [self.toa_dir + '/%s_solar_B%02d.tif' %(self.header, 1), ]
        try:
            scale, offset = self._get_scale()
        except:
            raise IOError('Failed read in scalling factors.')

    def _get_toa(self,):
        try:
            scale, offset = self._get_scale()
        except:
            raise IOError('Failed read in scalling factors.')
        bands_scale  = scale [self.bands-1]
        bands_offset = offset[self.bands-1] 
        toa          = np.array(parmap(gdal_reader, self.toa_file)).astype(float) * \
                                bands_scale[...,None, None] + bands_offset[...,None, None]
        qa_mask  = self._get_qa()
        sza      = self._get_angles()[1]
        toa      = toa / np.cos(np.deg2rad(sza))
        toa_mask = toa < 0
        mask     = qa_mask | toa_mask | sza.mask
        toa      = np.ma.array(toa, mask=mask)
        return toa

    def _get_angles(self,):
        saa, sza = np.array(parmap(gdal_reader, self.saa_sza)).astype(float).transpose(1,0,2,3)/100.
        vaa, vza = np.array(parmap(gdal_reader, self.vaa_vza)).astype(float).transpose(1,0,2,3)/100.
        saa = np.ma.array(saa, mask = ((saa > 180) | (saa < -180)))
        sza = np.ma.array(sza, mask = ((sza > 90 ) | (sza < 0   )))
        vaa = np.ma.array(vaa, mask = ((vaa > 180) | (vaa < -180)))
        vza = np.ma.array(vza, mask = ((vza > 90 ) | (vza < 0   )))
        saa.mask = sza.mask = vaa.mask = vza.mask = (saa.mask | sza.mask | vaa.mask | vza.mask)
        return saa, sza, vaa, vza

    def _get_scale(self,):
        scale, offset = [], []
        if sys.version_info >= (3,0):
            with open( self.mete_file, 'r', encoding='latin1') as f:
                for line in f:
                    if 'REFLECTANCE_MULT_BAND' in line:
                        scale.append(float(line.split()[-1]))
                    elif 'REFLECTANCE_ADD_BAND' in line:
                        offset.append(float(line.split()[-1]))
                    elif 'DATE_ACQUIRED' in line:
                        date = line.split()[-1]
                    elif 'SCENE_CENTER_TIME' in line:
                        time = line.split()[-1]
        else:
            with open( self.mete_file, 'rb') as f:
                for line in f:
                    if 'REFLECTANCE_MULT_BAND' in line:
                        scale.append(float(line.split()[-1]))
                    elif 'REFLECTANCE_ADD_BAND' in line:
                        offset.append(float(line.split()[-1]))
                    elif 'DATE_ACQUIRED' in line:
                        date = line.split()[-1]
                    elif 'SCENE_CENTER_TIME' in line:
                        time = line.split()[-1]
        datetime_str  = date + time
        self.sen_time = datetime.strptime(datetime_str.split('.')[0], '%Y-%m-%d"%H:%M:%S')
        return np.array(scale), np.array(offset)

    def _get_qa(self,):
        bqa = gdal_reader(self.qa_file)
        qa_mask = ~((bqa >= 2720) & (bqa <= 2732))
        return qa_mask
if __name__ == '__main__':
    l8 = read_l8('/home/ucfafyi/DATA/S2_MODIS/l_data/LC08_L1TP_192027_20170526_20170615_01_T1', (192, 27), 2017, 5, 26, bands=[2,3,4,5,6,7])
    #toa = l8._get_toa()
