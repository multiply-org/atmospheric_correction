#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'python')
import gdal
import datetime
import logging
import numpy as np
from glob import glob
import cPickle as pkl
from osgeo import osr
from smoothn import smoothn
from multi_process import parmap
from reproject import reproject_data
from get_brdf import get_brdf_six
from atmo_paras_optimization import solving_atmo_paras
from modis_l1b_reader import MODIS_L1b_reader
from emulation_engine import AtmosphericEmulationEngine

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,h,v,
                 year, doy,
                 emus_dir    = '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 mod_l1b_dir = '/data/selene/ucfajlg/Ujia/MODIS_L1b/GRIDDED',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 qa_thresh   = 255,
                 mod_cloud   = None,
                 save_file   = False,
                 aero_res    = 30000, # resolution for aerosol retrival in meters should be larger than 500
                 ):

        self.year        = year 
        self.doy         = doy
        self.date        = datetime.datetime(self.year, 1, 1) \
                                             + datetime.timedelta(self.doy - 1)
        self.month       = self.date.month
        self.day         = self.date.day
        self.h           = h
        self.v           = v
        self.mcd43_dir   = mcd43_dir
        self.mod_l1b_dir = mod_l1b_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.mod_cloud   = mod_cloud 
        self.global_dem  = global_dem
        self.cams_dir    = cams_dir
        self.subsample   = 10
        self.aero_res    = aero_res
        mcd43_tmp        = '%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'
        self.mcd43_file  = glob(mcd43_tmp%(self.mcd43_dir,\
                                 self.year, self.doy, self.h, self.v))[0]
    def _load_emus(self, sensor):
        AEE = AtmosphericEmulationEngine(sensor, self.emus_dir)
        up_bounds   = AEE.emulators[0].inputs[:,4:7].max(axis=0)
        low_bounds  = AEE.emulators[0].inputs[:,4:7].min(axis=0)
        bounds = np.array([low_bounds, up_bounds]).T
        return AEE, bounds

    def getting_atmo_paras(self,):
        self.modis_logger = logging.getLogger('MODIS Atmospheric Correction')
        self.modis_logger.setLevel(logging.INFO)
        if not self.modis_logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.modis_logger.addHandler(ch)
        self.modis_logger.propagate = False
        
        self.modis_logger.info('Start to retrieve atmospheric parameters.')
        modis_l1b        =  MODIS_L1b_reader(self.mod_l1b_dir, "h%02dv%02d"%(self.h,self.v),self.year)
        self.modis_files = [(i,modis_l1b.granules[i]) for i in modis_l1b.granules.keys() if i.date() == self.date.date()]
        self.modis_logger.info('%d MODIS file(s) is(are) found for doy %04d-%03d.'%(len(self.modis_files), self.year, self.doy))
        for timestamp, modis_file in self.modis_files:
            self._doing_one_file(modis_file, timestamp)
            #break

    def _doing_one_file(self, modis_file, timestamp):
        self.modis_logger.info('Doing %s.'%modis_file.b1.split('/')[-1].split('_EV_')[0])
	band_files  = [getattr(modis_file, 'b%d'%band) for band in range(1,8)]
	angle_files = [getattr(modis_file, ang) for ang in ['vza', 'sza', 'vaa', 'saa']]
	modis_toa   = []
	modis_angle = []
        f           = lambda fname: gdal.Open(fname).ReadAsArray()

        self.modis_logger.info('Reading in MODIS TOA.')
        modis_toa   = parmap(f, band_files)

        self.modis_logger.info('Reading in angles.')
        modis_angle = parmap(f, angle_files)

	scale  = np.array(modis_file.scale)
	offset = np.array(modis_file.offset)

	self.modis_toa    = np.array(modis_toa)*np.array(scale)[:,None, None] + offset[:,None, None]
	self.modis_angle  = np.array(modis_angle)/100.
        self.example_file = band_files[0]
        self.sen_time     = timestamp
	self.solving_modis_aerosol()
     
    def _modis_aerosol(self,):
        self.modis_logger.info('Getting emualated surface reflectance.') 
        vza, sza, vaa, saa = self.modis_angle
	self.modis_boa, self.modis_boa_qa, self.brdf_stds = get_brdf_six(self.mcd43_file,angles = [vza, sza, vaa - saa],
							                 bands= (1,2,3,4,5,6,7), Linds= None)
	if self.mod_cloud is None:
	    self.modis_cloud = np.zeros_like(self.modis_toa[0]).astype(bool)
 
        self.modis_logger.info('Getting elevation.')
        ele             = reproject_data(self.global_dem, self.example_file)
        ele.get_it()
        mask            = ~np.isfinite(ele.data)
        ele.data        = np.ma.array(ele.data, mask = mask)
        self.elevation  = ele.data/1000.

        self.modis_logger.info('Getting pripors from ECMWF forcasts.')
        aod, tcwv, tco3 = self._read_cams(self.example_file)
        self.aod550     = aod  * (1-0.14) # validation of +14% biase
        self.tco3       = tco3 * 46.698 * (1 - 0.05)
        self.tcwv       = tcwv / 10.

        self.tco3_unc   = np.ones(self.tco3.shape)   * 0.2
        self.aod550_unc = np.ones(self.aod550.shape) * 0.5
        self.tcwv_unc   = np.ones(self.tcwv.shape)   * 0.2

	qua_mask = np.all(self.modis_boa_qa <= self.qa_thresh, axis =0) 
	boa_mask = np.all(~self.modis_boa.mask, axis = 0) &\
	                  np.all(self.modis_boa>0, axis=0) &\
			  np.all(self.modis_boa<1, axis=0)
	toa_mask = np.all(np.isfinite(self.modis_toa), axis=0) &\
                          np.all(self.modis_toa>0, axis=0) & \
                          np.all(self.modis_toa<1, axis=0)

	self.modis_mask = qua_mask & boa_mask & toa_mask & (~self.modis_cloud)
        self.modis_AEE, self.modis_bounds = self._load_emus(self.modis_sensor)
        self.modis_solved = []
     
    def _read_cams(self, example_file, parameters = ['aod550', 'tcwv', 'gtco3']):
        netcdf_file = datetime.datetime(self.sen_time.year, self.sen_time.month, \
                                        self.sen_time.day).strftime("%Y-%m-%d.nc")
        template    = 'NETCDF:"%s":%s'
        ind         = np.abs((self.sen_time.hour  + self.sen_time.minute/60. + \
                              self.sen_time.second/3600.) - np.arange(0,25,3)).argmin() 
        sr         = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        proj       = sr.ExportToWkt()
        results = []
        for para in parameters:
            fname   = template%(self.cams_dir + '/' + netcdf_file, para)
            g       = gdal.Open(fname)
            g.SetProjection(proj)
            sub     = g.GetRasterBand(ind+1)
            offset  = sub.GetOffset()
            scale   = sub.GetScale()
            bad_pix = int(sub.GetNoDataValue())
            rep     = reproject_data(g, example_file)
            rep.get_it()
            data    = rep.g.GetRasterBand(ind+1).ReadAsArray()
            data    = data*scale + offset
            mask    = (data == (bad_pix*scale + offset)) 
            if mask.sum()>=1:
                data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            results.append(data)
        return results
                   
    def solving_modis_aerosol(self,):

        self.modis_logger = logging.getLogger('MODIS Atmospheric Correction')
        self.modis_logger.setLevel(logging.INFO)
        if not self.modis_logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.modis_logger.addHandler(ch)
        self.modis_logger.propagate = False

        self.modis_sensor = 'TERRA'
        self._modis_aerosol()
        self.modis_solved = []
        if self.aero_res < 500:
            self.modis_logger.warning( 'The best resolution of aerosol should be larger \
                                     than 500 meters (inlcude), so it is set to 500 meters.')
            self.aero_res = 500
        self.block_size   = int(self.aero_res/500)
        num_blocks        = int(np.ceil(2400/self.block_size))
        self.modis_logger.info('Start solving......')
        blocks            = zip(np.repeat(range(num_blocks), num_blocks), np.tile(range(num_blocks), num_blocks))
        self.modis_solved = parmap(self._m_block_solver, blocks)
        #for i in range(num_blocks):
        #    for j in range(num_blocks):
        #        self.modis_logger.info('Doing block %03d-%03d.'%(i+1,j+1))
        #        self._m_block_solver([i,j])

        inds = np.array([[i[0], i[1]] for i in self.modis_solved])
        rets = np.array([i[2][0]      for i in self.modis_solved])

        aod_map    = np.zeros((num_blocks,num_blocks))
        aod_map[:] = np.nan
        tcwv_map   = aod_map.copy()
        tco3_map   = aod_map.copy()

        aod_map [inds[:,0], inds[:,1]] = rets[:,0]
        tcwv_map[inds[:,0], inds[:,1]] = rets[:,1]
        tco3_map[inds[:,0], inds[:,1]] = rets[:,2]
        para_names = 'aod550', 'tcwv', 'tco3'

        g = gdal.Open(self.example_file)
        xmin, ymax = g.GetGeoTransform()[0], g.GetGeoTransform()[3]
        projection = g.GetProjection()

        results = []
        self.modis_logger.info('Finished retrieval and saving them into local files.')
        for i,para_map in enumerate([aod_map, tcwv_map, tco3_map]):
            s     = smoothn(para_map.copy(), isrobust=True, verbose=False)[1]
            smed  = smoothn(para_map.copy(), isrobust=True, verbose=False, s=s)[0]
            xres, yres = self.block_size*500, self.block_size*500
            geotransform = (xmin, xres, 0, ymax, 0, -yres)
            nx, ny = smed.shape
            dst_ds = gdal.GetDriverByName('GTiff').Create(self.mod_l1b_dir + '/atmo_paras/' + \
                                                          self.example_file.split('/')[-1].split('_EV_')[0] \
                                                          + '_EV_%s.tif'%para_names[i], ny, nx, 1, gdal.GDT_Float32)

            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            dst_ds.GetRasterBand(1).WriteArray(smed)
            dst_ds.FlushCache()
            dst_ds = None
            results.append(smed)
        self.aod550_map, self.tcwv_map, self.tco3_map = results
        
    def _m_block_solver(self,block):
	i,j = block
        self.modis_logger.info('Doing block %03d-%03d.'%(i+1,j+1))
	block_mask= np.zeros_like(self.modis_mask).astype(bool)
	block_mask[i*self.block_size:(i+1)*self.block_size,j*self.block_size:(j+1)*self.block_size] = True        
	mask      = self.modis_mask[block_mask].reshape(self.block_size, self.block_size)
	prior     = self.aod550[block_mask].mean(), self.tcwv[block_mask].mean(), self.tco3[block_mask].mean()
	if mask.sum() <= 0:
            self.modis_logger.warning('No valid values in block %03d-%03d, and priors are used for this block.'%(i+1,j+1))
            return [i,j,[prior, 0], prior]
	else:
            
	    boa, toa  = self.modis_boa[:, block_mask].reshape(7,self.block_size, self.block_size),\
			self.modis_toa[:, block_mask].reshape(7,self.block_size, self.block_size)
	    vza, sza  = (self.modis_angle[:2, block_mask]*np.pi/180.).reshape(2,self.block_size, self.block_size)
	    vaa, saa  = self.modis_angle[2:, block_mask].reshape(2,self.block_size, self.block_size)
	    boa_qa    = self.modis_boa_qa[:, block_mask].reshape(7,self.block_size, self.block_size)
	    elevation = self.elevation[block_mask].reshape(self.block_size, self.block_size)
	    brdf_std  = self.brdf_stds[:,block_mask].reshape(7, self.block_size, self.block_size)
	    self.atmo = solving_atmo_paras(self.modis_sensor, self.emus_dir, boa, toa, sza, vza, saa, vaa, \
					   elevation, boa_qa, boa_bands=[645,869,469,555,1240,1640,2130], mask=mask, \
					   band_indexs=[0,1,2,3,4,5,6], prior=prior, subsample=self.subsample, brdf_std=brdf_std)
	    self.atmo._load_unc()
	    self.atmo.aod_unc   = self.aod550_unc[block_mask].max()
	    self.atmo.wv_unc    = self.tcwv_unc  [block_mask].max()  
	    self.atmo.ozone_unc = self.tco3_unc  [block_mask].max()
	    self.atmo.AEE       = self.modis_AEE
	    self.atmo.bounds    = self.modis_bounds
            return [i,j, self.atmo.optimization(), prior]

if __name__ == "__main__":
    aero = solve_aerosol(17, 5, 2017, 247,mcd43_dir = '/home/ucfafyi/DATA/Multiply/MCD43/' )
    aero.getting_atmo_paras()
    #solved  = aero.prepare_modis()
