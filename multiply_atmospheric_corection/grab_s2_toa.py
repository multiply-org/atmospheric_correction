#/usr/bin/env python
import gdal
import os
import sys
sys.path.insert(0, 'python')
import xml.etree.ElementTree as ET
import numpy as np
from multiprocessing import Pool
from glob import glob
from cloud import classification
import subprocess
import copy

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
                 bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A']):
        self.s2_toa_dir = s2_toa_dir
        self.s2_tile    = s2_tile
        self.year       = year
        self.month      = month
        self.day        = day
        self.bands      = bands # selected bands
        self.s2_bands   = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', \
                          'B08','B8A', 'B09', 'B10', 'B11', 'B12' #all bands
        self.s2_file_dir = os.path.join(self.s2_toa_dir, self.s2_tile[:-3],\
                                        self.s2_tile[-3], self.s2_tile[-2:],\
                                        str(self.year), str(self.month), str(self.day),'0')
        self.selected_img = None
        
    def get_s2_toa(self,vrt = False):
        if vrt:
	    # open the created vrt file with 10 meter, 20 meter and 60 meter 
	    # grouped togehter and use gdal memory map to open it
	    g = gdal.Open(self.s2_file_dir+'/10meter.vrt')
	    data= g.GetVirtualMemArray()
	    b2,b3,b4,b8 = data
	    g1 = gdal.Open(self.s2_file_dir+'/20meter.vrt')
	    data1 = g1.GetVirtualMemArray()
	    b5, b6, b7, b8a, b11, b12 = data1
	    g2 = gdal.Open(self.s2_file_dir+'/60meter.vrt')
	    data2 = g2.GetVirtualMemArray()
	    b1, b9, b10 = data2
	    img = dict(zip(self.s2_bands, [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b8a]))
	    if self.bands is not None:
	        imgs = {k: img[k] for k in self.bands}
	    else:
	        imgs = img
        else:
             if self.bands is None:
                 self.bands = self.s2_bands
             fname = [self.s2_file_dir+'/%s.jp2'%i for i in self.bands]
             pool = Pool(processes=len(fname))
             ret = pool.map(read_s2_band, fname)
             imgs = dict(zip(self.bands, ret))
        self.selected_img = copy.deepcopy(imgs)
        return self.selected_img

    def get_s2_cloud(self,):
        if glob(self.s2_file_dir+'/cloud.tiff')==[]:
            print 'loading Sentinel2 data...'
            needed_bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
            if self.selected_img is None:
                self.bands = needed_bands
                img = self.get_s2_toa(vrt=False)
            else:
                add_band   = [i for i in needed_bands if i not in self.selected_img.keys()]
                exist_band = [i for i in needed_bands if i in self.selected_img.keys()]
                if len(add_band) ==0:
                    img = {i:self.selected_img[i] for i in needed_bands}
                else:
                    fname = [self.s2_file_dir+'/%s.jp2'%i for i in add_band]
                    pool = Pool(processes=len(fname))
                    ret = pool.map(read_s2_band, fname)
                    add_img = dict(zip(add_band, ret))
                    img = {i:self.selected_img[i] for i in exist_band}
                    img.update(add_img)

            cl = classification(img = img)
            cl.Get_cm_p()
            g=None; g1=None
            self.cloud = cl.cm
            g = gdal.Open(self.s2_file_dir+'/B04.jp2')
            driver = gdal.GetDriverByName('GTiff')
            g1 = driver.Create(self.s2_file_dir+'/cloud.tiff', \
                               g.RasterXSize, g.RasterYSize, 1, gdal.GDT_Byte)

            projection   = g.GetProjection()
            geotransform = g.GetGeoTransform()
            g1.SetGeoTransform( geotransform )
            g1.SetProjection( projection )
            gcp_count = g.GetGCPs()
            if gcp_count != 0:
                g1.SetGCPs( gcp_count, g.GetGCPProjection() )
            g1.GetRasterBand(1).WriteArray(self.cloud)
            g1=None; g=None
            del cl
        else:
            self.cloud = cloud = gdal.Open(self.s2_file_dir+\
                                           '/cloud.tiff').ReadAsArray().astype(bool)
        self.cloud_cover = 1.*self.cloud.sum()/self.cloud.size

    def get_s2_angles(self,reconstruct = True, slic = None):


	tree = ET.parse(self.s2_file_dir+'/metadata.xml')
	root = tree.getroot()
	#Sun_Angles_Grid
	saa =[]
	sza =[]
	msz = []
	msa = []
	#Viewing_Incidence_Angles_Grids
	vza = {}
	vaa = {}
	mvz = {}
	mva = {}
	for child in root:
	    for j in child:
		for k in j.findall('Sun_Angles_Grid'):
		    for l in k.findall('Zenith'):
			for m in l.findall('Values_List'):
			    for x in m.findall('VALUES'):
				sza.append(x.text.split())

		    for n in k.findall('Azimuth'):
			for o in n.findall('Values_List'):
			    for p in o.findall('VALUES'):
				saa.append(p.text.split())

		for ms in j.findall('Mean_Sun_Angle'):
		    self.msz = float(ms.find('ZENITH_ANGLE').text)
		    self.msa = float(ms.find('AZIMUTH_ANGLE').text)

		for k in j.findall('Viewing_Incidence_Angles_Grids'):
		    for l in k.findall('Zenith'):
			for m in l.findall('Values_List'):
			    vza_sub = []
			    for x in m.findall('VALUES'):
				vza_sub.append(x.text.split())
			    bi, di, angles = k.attrib['bandId'], \
                                             k.attrib['detectorId'], np.array(vza_sub).astype(float)
			    vza[(int(bi),int(di))] = angles

		    for n in k.findall('Azimuth'):
			for o in n.findall('Values_List'):
			    vaa_sub = []
			    for p in o.findall('VALUES'):
				vaa_sub.append(p.text.split())
			    bi, di, angles = k.attrib['bandId'],\
                                             k.attrib['detectorId'], np.array(vaa_sub).astype(float)
			    vaa[(int(bi),int(di))] = angles

		for mvia in j.findall('Mean_Viewing_Incidence_Angle_List'):
		    for i in mvia.findall('Mean_Viewing_Incidence_Angle'):
			mvz[int(i.attrib['bandId'])] = float(i.find('ZENITH_ANGLE').text)
			mva[int(i.attrib['bandId'])] = float(i.find('AZIMUTH_ANGLE').text)
	self.saa, self.sza = np.array(saa).astype(float), np.array(sza).astype(float)
	dete_id = np.unique([i[1] for i in vaa.keys()])
	band_id = range(13)
	bands_vaa = []
	bands_vza = []
	for i in band_id:
	    band_vaa = np.zeros((23,23))
	    band_vza = np.zeros((23,23))
	    band_vaa[:] = np.nan
	    band_vza[:] = np.nan
	    for j in dete_id:
		good = ~np.isnan(vaa[(i,j)])
		band_vaa[good] = vaa[(i,j)][good]
		good = ~np.isnan(vza[(i,j)])
		band_vza[good] = vza[(i,j)][good]
	    bands_vaa.append(band_vaa)
	    bands_vza.append(band_vza)
	bands_vaa, bands_vza = np.array(bands_vaa), np.array(bands_vza)
	vaa  = {}; vza  = {}
	mva_ = {}; mvz_ = {}
	for i, band in enumerate(self.s2_bands):
	    vaa[band]  = bands_vaa[i]
	    vza[band]  = bands_vza[i]
	    mva_[band] = mva[i]
	    mvz_[band] = mvz[i]               

	if self.bands is None:
            bands = self.s2_bands
        else:
            bands = self.bands
	self.vza = {}; self.vaa = {}
	self.mvz = {}; self.mva = {}
	for band in bands:
	    self.vza[band] = vza[band]
	    self.vaa[band] = vaa[band]
	    self.mvz[band] = mvz_[band]
	    self.mva[band] = mva_[band]
	self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                       'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}

        if reconstruct:
            if len(glob(self.s2_file_dir + '/angles/VAA_VZA_*.img')) == 13:
                pass
            else:
                print 'Reconstructing Sentinel 2 angles...'
                subprocess.call(['python', './python/s2a_angle_bands_mod.py', \
                                  self.s2_file_dir+'/metadata.xml',  '1'])

	    if self.bands is None:
		bands = self.s2_bands
	    else:
		bands = self.bands

            self.vaa = {}; self.vza = {}
            fname = [self.s2_file_dir+'/angles/VAA_VZA_%s.img'%band for band in bands]
            pool = Pool(processes=len(fname))
            ret = pool.map(read_s2_band, fname)
            for i,angs in enumerate(ret):
                if slic is None:
                    self.vaa[bands[i]] = angs[0]
                    self.vza[bands[i]] = angs[1]
                else:
                    resolution_ratio = angs[0].shape[0]/10980
                    x_ind, y_ind = (np.array(slic)*resolution_ratio).astype(int)
                    self.vaa[bands[i]] = angs[0][x_ind, y_ind]
                    self.vza[bands[i]] = angs[1][x_ind, y_ind]
            '''
	    for band in bands:
		g = gdal.Open(self.s2_file_dir + '/angles/VAA_VZA_%s.img'%band)
                VAA, VZA = g.GetRasterBand(1).ReadAsArray(), g.GetRasterBand(2).ReadAsArray()
                if slic is None:
                    self.vaa[band] = VAA 
                    self.vza[band] = VZA
                else:
                    resolution_ratio = VAA.shape[0]/10980
                    x_ind, y_ind = (np.array(slic)*resolution_ratio).astype(int)
                    self.vaa[band] = VAA[x_ind, y_ind]
                    self.vza[band] = VZA[x_ind, y_ind]
            '''  
            self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                           'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}

    def get_wv(self,):
	fname   = [self.s2_file_dir+'/%s.jp2'%i for i in ['B8A', 'B09']]
	pool    = Pool(processes=len(fname))
	b8a, b9 = pool.map(read_s2_band, fname)

        b9  = np.repeat(np.repeat(b9, 3, axis=0), 3, axis=1) # repeat image to match the b8a
        SOLAR_SPECTRAL_IRRADIANCE_B_8A = 955.19
        SOLAR_SPECTRAL_IRRADIANCE_B_09 = 813.04	
        #the same sun earth distance correction factors cancled out
        CIBR = SOLAR_SPECTRAL_IRRADIANCE_B_8A/SOLAR_SPECTRAL_IRRADIANCE_B_09 * b8a/b9 
        return CIBR

if __name__ == '__main__':
    
    s2 = read_s2('/home/ucfafyi/DATA/S2_MODIS/s_data/', '29SQB', \
                  2017, 9, 4, bands = ['B02', 'B03', 'B04', 'B08', 'B11'] )
    '''
    s2.selected_img = s2.get_s2_toa() 
    s2.get_s2_cloud()
    '''
    s2.get_s2_angles()
    cibr = s2.get_wv()
