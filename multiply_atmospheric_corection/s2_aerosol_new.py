#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'util')
import gdal
import json
import datetime
import logging
import numpy as np
from ddv import ddv
from glob import glob
from scipy import signal, ndimage
import cPickle as pkl
from osgeo import osr
from smoothn import smoothn
from grab_s2_toa import read_s2
from multi_process import parmap
from reproject import reproject_data
from get_brdf import get_brdf_six
from grab_uncertainty import grab_uncertainty
from atmo_paras_optimization_new import solving_atmo_paras
from spatial_mapping import Find_corresponding_pixels, cloud_dilation
from emulation_engine import AtmosphericEmulationEngine
from psf_optimize import psf_optimize

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 emus_dir    = '/home/ucfajlg/Data/util/S2S3Synergy/optical_emulators',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 wv_emus_dir = '/home/ucfafyi/DATA/Multiply/emus/wv_msi_retrieval.pkl',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 s2_tile     = '29SQB',
                 s2_psf      = None,
                 qa_thresh   = 255,
                 aero_res    = 3050, # resolution for aerosol retrival in meters should be larger than 500
                 reconstruct_s2_angle = True):

        self.year        = year 
        self.month       = month
        self.day         = day
        self.date        = datetime.datetime(self.year, self.month, self.day)
        self.doy         = self.date.timetuple().tm_yday
        self.mcd43_dir   = mcd43_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  = global_dem
        self.wv_emus_dir = wv_emus_dir
        self.cams_dir    = cams_dir
        self.s2_tile     = s2_tile
        self.s2_psf      = s2_psf 
        self.s2_u_bands  = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A', 'B09' #bands used for the atmo-cor
        self.band_indexs = [1, 2, 3, 7, 11, 12]
        self.boa_bands   = [469, 555, 645, 869, 1640, 2130]
        self.full_res    = (10980, 10980)
        self.aero_res    = aero_res
        self.mcd43_tmp   = '%s/MCD43A1.A%d%03d.%s.006.*.hdf'
        self.reconstruct_s2_angle  = reconstruct_s2_angle
        self.s2_spectral_transform = [[ 1.06946607,  1.03048916,  1.04039226,  1.00163932,  1.00010918, 0.95607606,  0.99951677],
                                      [ 0.0035921 , -0.00142761, -0.00383504, -0.00558762, -0.00570695, 0.00861192,  0.00188871]]       
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.s2_sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.s2_sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.s2_sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def repeat_extend(self,data, shape=(10980, 10980)):
        da_shape    = data.shape
        re_x, re_y  = int(1.*shape[0]/da_shape[0]), int(1.*shape[1]/da_shape[1])
        new_data    = np.zeros(shape)
        new_data[:] = -9999
        new_data[:re_x*da_shape[0], :re_y*da_shape[1]] = np.repeat(np.repeat(data, re_x,axis=0), re_y, axis=1)
        return new_data
        
    def gaussian(self, xstd, ystd, angle, norm = True):
        win = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        winx = int(round(win*(2**0.5)))
        winy = int(round(win*(2**0.5)))
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[center[0]-win/2: center[0]+win/2, center[1]-win/2:center[1]+win/2]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus 
    
    def _s2_aerosol(self,):
        
        self.s2_logger.propagate = False
        self.s2_logger.info('Start to retrieve atmospheric parameters.')
        self.s2 = read_s2(self.s2_toa_dir, self.s2_tile, self.year, self.month, self.day, self.s2_u_bands)
        self.s2_logger.info('Reading in TOA reflectance.')
	selected_img = self.s2.get_s2_toa() 
	self.s2.get_s2_cloud()
        self.s2_logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        #self.s2.cloud[:] = False # due to the bad cloud algrithm 
        self.s2_logger.info('Find corresponding pixels between S2 and MODIS tiles')
        tiles = Find_corresponding_pixels(self.s2.s2_file_dir+'/B04.jp2', destination_res=500) 
        if len(tiles.keys())>1:
            self.s2_logger.info('This sentinel 2 tile covers %d MODIS tile.'%len(tiles.keys()))
        self.mcd43_files = []
        boas, boa_qas, brdf_stds, Hxs, Hys    = [], [], [], [], []
        for key in tiles.keys():
            self.s2_logger.info('Getting BOA from MODIS tile: %s.'%key)
            mcd43_file  = glob(self.mcd43_tmp%(self.mcd43_dir, self.year, self.doy, key))[0]
            self.mcd43_files.append(mcd43_file)
            self.H_inds, self.L_inds = tiles[key]
	    Lx, Ly = self.L_inds
	    Hx, Hy = self.H_inds
            Hxs.append(Hx); Hys.append(Hy)
            self.s2_logger.info( 'Getting the angles and simulated surface reflectance.')
            self.s2.get_s2_angles(self.reconstruct_s2_angle)
 
	    if self.reconstruct_s2_angle:
		self.s2_angles = np.zeros((4, 6, len(Hx)))
		hx, hy = (Hx*23./self.full_res[0]).astype(int), \
                         (Hy*23./self.full_res[1]).astype(int) # index the 23*23 sun angles
		for j, band in enumerate (self.s2_u_bands[:-2]):
		    self.s2_angles[[0,2],j,:] = (self.s2.angles['vza'][band])[Hx, Hy], \
                                                (self.s2.angles['vaa'][band])[Hx, Hy]

		    self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy], \
                                                self.s2.angles['saa'][hx, hy]
	    else:
		self.s2_angles = np.zeros((4, 6, len(Hx)))
		hx, hy = (Hx*23./self.full_res[0]).astype(int), \
                         (Hy*23./self.full_res[0]).astype(int) # index the 23*23 sun angles
		for j, band in enumerate (self.s2_u_bands[:-2]):
		    self.s2_angles[[0,2],j,:] = self.s2.angles['vza'][band][hx, hy], \
                                                self.s2.angles['vaa'][band][hx, hy]
		    self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy], \
                                                self.s2.angles['saa'][hx, hy]

	    #use mean value to fill bad values
	    for i in range(4):
		mask = ~np.isfinite(self.s2_angles[i])
		if mask.sum()>0:
		    self.s2_angles[i][mask] = np.interp(np.flatnonzero(mask), \
							np.flatnonzero(~mask), \
                                                        self.s2_angles[i][~mask]) # simple interpolation
	    vza, sza = self.s2_angles[:2]
	    vaa, saa = self.s2_angles[2:]
            raa      = vaa - saa
	    # get the simulated surface reflectance
	    s2_boa, s2_boa_qa, brdf_std = get_brdf_six(mcd43_file, angles=[vza, sza, raa],\
                                                       bands=(3,4,1,2,6,7), Linds= [Lx, Ly])
            boas.append(s2_boa); boa_qas.append(s2_boa_qa); brdf_stds.append(brdf_std)
        
	self.s2_boa    = np.hstack(boas)
	self.s2_boa_qa = np.hstack(boa_qas)
	self.brdf_stds = np.hstack(brdf_stds)
        self.s2_logger.info('Applying spectral transform.')
        self.s2_boa = self.s2_boa*np.array(self.s2_spectral_transform)[0,:-1][...,None] + \
                                  np.array(self.s2_spectral_transform)[1,:-1][...,None]
	self.Hx  = np.hstack(Hxs)
        self.Hy  = np.hstack(Hys)
        x_resamp = (np.repeat(np.arange(self.num_blocks), self.num_blocks) / self.num_blocks * self.s2.angles['sza'].shape[0]).astype(int)
        y_resamp = (np.tile  (np.arange(self.num_blocks), self.num_blocks) / self.num_blocks * self.s2.angles['sza'].shape[1]).astype(int)
        self.sza = self.s2.angles['sza'][x_resamp, y_resamp].reshape(self.num_blocks, self.num_blocks)
        self.saa = self.s2.angles['saa'][x_resamp, y_resamp].reshape(self.num_blocks, self.num_blocks)
        self.vza = []
        self.vaa = []
        for band in self.s2_u_bands[:-2]:
            if self.reconstruct_s2_angle:
                shape = (self.num_blocks, self.s2.angles['vza'][band].shape[0] / self.num_blocks, \
                         self.num_blocks, self.s2.angles['vza'][band].shape[1] / self.num_blocks)
                self.vza.append(self.s2.angles['vza'][band].reshape(shape).mean(axis = (3, 1)))
                self.vaa.append(self.s2.angles['vaa'][band].reshape(shape).mean(axis = (3, 1)))
            else:
                self.vza.append(self.s2.angles['vza'][band][x_resamp, y_resamp].reshape(self.num_blocks, self.num_blocks)) 
                self.vaa.append(self.s2.angles['vaa'][band][x_resamp, y_resamp].reshape(self.num_blocks, self.num_blocks))
        self.vza = np.array(self.vza) 
        self.vaa = np.array(self.vaa)
        self.raa = self.saa[None, ...] - self.vaa
        self.s2_logger.info('Getting elevation.')
        example_file   = self.s2.s2_file_dir+'/B04.jp2'
        ele_data       = reproject_data(self.global_dem, example_file).data
        mask           = ~np.isfinite(ele_data)
        ele_data       = np.ma.array(ele_data, mask = mask)
        self.elevation = ele_data.reshape((self.num_blocks, ele_data.shape[0] / self.num_blocks, \
                                           self.num_blocks, ele_data.shape[1] / self.num_blocks)).mean(axis=(3,1))/1000.

        self.s2_logger.info('Getting pripors from ECMWF forcasts.')
	sen_time_str    = json.load(open(self.s2.s2_file_dir+'/tileInfo.json', 'r'))['timestamp']
      	self.sen_time   = datetime.datetime.strptime(sen_time_str, u'%Y-%m-%dT%H:%M:%S.%fZ') 
        aot, tcwv, tco3 = np.array(self._read_cams(example_file)).reshape((3, self.num_blocks, \
                                   self.block_size, self.num_blocks, self.block_size)).mean(axis=(4, 2))
        self.aot        = aot  #* (1-0.14) # validation of +14% biase
        self.tco3       = tco3 * 46.698 #* (1 - 0.05)
        tcwv            = tcwv / 10. 
        self.tco3_unc   = np.ones(self.tco3.shape)   * 0.2
        self.aot_unc    = np.ones(self.aot.shape) * 0.5
        
        self.s2_logger.info('Trying to get the tcwv from the emulation of sen2cor look up table.')
        try:
            b8a, b9  = np.repeat(np.repeat(selected_img['B8A']*0.0001, 2, axis=0), 2, axis=1)[self.Hx, self.Hy],\
                                 np.repeat(np.repeat(selected_img['B09']*0.0001, 6, axis=0), 6, axis=1)[self.Hx, self.Hy]
	    wv_emus   = pkl.load(open(self.wv_emus_dir, 'rb'))
            hx, hy    = (1. * self.Hx*self.vza.shape[0]/self.full_res[0]).astype(int), \
                        (1. * self.Hy*self.vza.shape[1]/self.full_res[0]).astype(int) 
            vza, vaa  = self.vza[-1, hx, hy], self.vaa[-1, hx, hy]

            hx, hy    = (1. * self.Hx*self.sza.shape[0]/self.full_res[0]).astype(int), \
                        (1. * self.Hy*self.sza.shape[1]/self.full_res[0]).astype(int) 
            sza, saa  = self.sza[hx, hy], self.saa[hx, hy]

            hx, hy    = (1. * self.Hx*self.elevation.shape[0]/self.full_res[0]).astype(int), \
                        (1. * self.Hy*self.elevation.shape[1]/self.full_res[0]).astype(int) 
            elevation = self.elevation[hx, hy]

	    inputs    = np.array([b9, b8a, vza, sza, abs(saa-vaa), elevation]).T
            tcwv_mask = b8a < 0.1 
            tcwv      = np.zeros(self.full_res)
            tcwv[:]   = np.nan
            tcwv_unc  = tcwv.copy()

	    s2_tcwv, s2_tcwv_unc, _ = wv_emus.predict(inputs, do_unc = True)
            if tcwv_mask.sum() >= 1:
                s2_tcwv[tcwv_mask]  = np.interp(np.flatnonzero( tcwv_mask), \
                                                np.flatnonzero(~tcwv_mask), s2_tcwv[~tcwv_mask]) # simple interpolation
            tcwv    [self.Hx, self.Hy] = s2_tcwv
            tcwv_unc[self.Hx, self.Hy] = s2_tcwv_unc
            self.tcwv                  = np.nanmean(tcwv    .reshape(self.num_blocks, self.block_size, \
                                                                     self.num_blocks, self.block_size), axis = (3,1))
            self.tcwv_unc              = np.nanmax (tcwv_unc.reshape(self.num_blocks, self.block_size, \
                                                                     self.num_blocks, self.block_size), axis = (3,1))
	except:
            self.s2_logger.warning('Getting tcwv from the emulation of sen2cor look up table failed, ECMWF data used.')
            self.tcwv     = tcwv
            self.tcwv_unc = np.ones(self.tcwv.shape) * 0.2

        self.s2_logger.info('Trying to get the aot from ddv method.')
        try:
            red_emus  = self.emus[0][3], self.emus[1][3], self.emus[2][3] 
            blue_emus = self.emus[0][1], self.emus[1][1], self.emus[2][1] 
            sza       = self.sza
            blue_vza  = self.vza[0]
            red_vza   = self.vza[2]
            blue_raa  = self.vaa[0] - self.saa
            red_raa   = self.vaa[2] - self.saa
            b2, b4,   = selected_img['B02']/10000., selected_img['B04']/10000.
            b8, b12   = selected_img['B08']/10000., selected_img['B12']/10000.
            b12       = np.repeat(np.repeat(b12, 2, axis = 1), 2, axis = 0)

            this_ddv  = ddv(b2, b4, b8, b12, 'msi', sza, 
                            np.array([blue_vza, red_vza]), 
			    np.array([blue_raa, red_raa]), 
                            self.elevation, self.tcwv, self.tco3, \
                            red_emus = red_emus, blue_emus = blue_emus)
            solved = this_ddv._ddv_prior()      

            if solved[0] < 0:
                self.s2_logger.warning('DDV failed and only cams data used for the prior.')
            else:
                self.s2_logger.info('DDV solved aot is %.02f, and it will used as the mean value of cams prediction.'%solved[0])
                self.aot += (solved[0]-self.aot.mean())
        except:
            self.s2_logger.warning('Getting aot from ddv failed.')

        self.s2_logger.info('Applying PSF model.')
        if self.s2_psf is None:
            self.s2_logger.info('No PSF parameters specified, start solving.')
            high_img    = np.repeat(np.repeat(selected_img['B11'], 2, axis=0), 2, axis=1)*0.0001
            high_indexs = self.Hx, self.Hy
            low_img     = self.s2_boa[4]
            qa, cloud   = self.s2_boa_qa[4], self.s2.cloud
            psf         = psf_optimize(high_img, high_indexs, low_img, qa, cloud, 2)
            xs, ys      = psf.fire_shift_optimize()
            xstd, ystd  = 29.75, 39
            ang         = 0
            self.s2_logger.info('Solved PSF parameters are: %.02f, %.02f, %d, %d, %d, and the correlation is: %f.' \
                                 %(xstd, ystd, 0, xs, ys, 1-psf.costs.min()))
        else:
            xstd, ystd, ang, xs, ys = self.s2_psf
        # apply psf shifts without going out of the image extend  
        shifted_mask = np.logical_and.reduce(((self.Hx+int(xs)>=0),
                                              (self.Hx+int(xs)<self.full_res[0]), 
                                              (self.Hy+int(ys)>=0),
                                              (self.Hy+int(ys)<self.full_res[0])))
        
        self.Hx, self.Hy = self.Hx[shifted_mask]+int(xs), self.Hy[shifted_mask]+int(ys)
        #self.Lx, self.Ly = self.Lx[shifted_mask], self.Ly[shifted_mask]
        self.s2_boa      = self.s2_boa   [:, shifted_mask]
        self.s2_boa_qa   = self.s2_boa_qa[:, shifted_mask]
        self.brdf_stds   = self.brdf_stds[:, shifted_mask]

        self.s2_logger.info('Getting the convolved TOA reflectance.')
        self.valid_pixs = sum(shifted_mask) # count how many pixels is still within the s2 tile 
        ker_size        = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        self.bad_pixs   = np.zeros(self.valid_pixs).astype(bool)
        imgs = []
        for i, band in enumerate(self.s2_u_bands[:-2]):
            if selected_img[band].shape != self.full_res:
                imgs.append( self.repeat_extend(selected_img[band], shape = self.full_res))
            else:
                imgs.append(selected_img[band])
        
        border_mask = np.zeros(self.full_res).astype(bool)
        border_mask[[0, -1], :] = True
        border_mask[:, [0, -1]] = True
        self.bad_pixs = cloud_dilation(self.s2.cloud | border_mask, iteration= ker_size/2)[self.Hx, self.Hy]
        del selected_img; del self.s2.selected_img;
        ker = self.gaussian(xstd, ystd, ang) 
        f   = lambda img: signal.fftconvolve(img, ker, mode='same')[self.Hx, self.Hy]*0.0001 
        self.s2_toa = np.array(parmap(f,imgs))
        #del imgs
        # get the valid value masks
        qua_mask = np.all(self.s2_boa_qa <= self.qa_thresh, axis = 0)
        boa_mask = np.all(~self.s2_boa.mask,axis = 0 ) &\
                          np.all(self.s2_boa > 0, axis = 0) &\
                          np.all(self.s2_boa < 1, axis = 0)
        toa_mask =       (~self.bad_pixs) &\
                          np.all(self.s2_toa > 0, axis = 0) &\
                          np.all(self.s2_toa < 1, axis = 0)
        self.s2_mask    = boa_mask & toa_mask & qua_mask 
        self.Hx         = self.Hx          [self.s2_mask]
        self.Hy         = self.Hy          [self.s2_mask]
        self.s2_toa     = self.s2_toa   [:, self.s2_mask]
        self.s2_boa     = self.s2_boa   [:, self.s2_mask]
        self.s2_boa_qa  = self.s2_boa_qa[:, self.s2_mask]
        self.brdf_stds  = self.brdf_stds[:, self.s2_mask]
        self.s2_boa_unc = grab_uncertainty(self.s2_boa, self.boa_bands, self.s2_boa_qa, self.brdf_stds).get_boa_unc()
        self.s2_logger.info('Solving...')
        self.aero = solving_atmo_paras(self.s2_boa, 
                                  self.s2_toa,
                                  self.sza, 
                                  self.vza,
                                  self.saa, 
                                  self.vaa,
                                  self.aot, 
                                  self.tcwv,
				  self.tco3, 
                                  self.elevation,
				  self.aot_unc,
				  self.tcwv_unc,
				  self.tco3_unc,
				  self.s2_boa_unc,
				  self.Hx, self.Hy,
				  self.full_res,
				  self.aero_res,
				  self.emus,
                                  self.band_indexs,
                                  self.boa_bands)
        solved = self.aero._optimization()
        return solved

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
	    rep_g   = reproject_data(g, example_file).g
	    data    = rep_g.GetRasterBand(ind+1).ReadAsArray()
	    data    = data*scale + offset
	    mask    = (data == (bad_pix*scale + offset))
	    if mask.sum()>=1:
		data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
	    results.append(data)
        return results

    def solving_s2_aerosol(self,):
        
        self.s2_logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
        self.s2_logger.setLevel(logging.INFO)
        if not self.s2_logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.s2_logger.addHandler(ch)
        self.s2_logger.propagate = False

        self.s2_sensor  = 'MSI'
        self.s2_logger.info('Doing Sentinel 2 tile: %s on %d-%02d-%02d.'%(self.s2_tile, self.year, self.month, self.day))
        self.block_size = int(self.aero_res/10)
        self.num_blocks = int(np.ceil(self.full_res[0]/self.block_size)) 
        self.solved     = self._s2_aerosol()[0].reshape(3, self.num_blocks, self.num_blocks)
        self.s2_logger.info('Finished retrieval and saving them into local files.')
        g = gdal.Open(self.s2.s2_file_dir+'/B04.jp2')
        xmin, ymax = g.GetGeoTransform()[0], g.GetGeoTransform()[3]
        projection = g.GetProjection()
        para_names = 'aot', 'tcwv', 'tco3'
        for i,para_map in enumerate(self.solved):
            xres, yres = self.block_size*10, self.block_size*10
            geotransform = (xmin, xres, 0, ymax, 0, -yres)
            nx, ny = self.num_blocks, self.num_blocks
            dst_ds = gdal.GetDriverByName('GTiff').Create(self.s2.s2_file_dir + \
                                          '/%s.tif'%para_names[i], ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform)   
            dst_ds.SetProjection(projection) 
            dst_ds.GetRasterBand(1).WriteArray(para_map)
            dst_ds.FlushCache()                     
            dst_ds = None
        self.aot_map, self.tcwv_map, self.tco3_map = self.solved

if __name__ == "__main__":
    aero = solve_aerosol( 2017, 9, 4, mcd43_dir = '/home/ucfafyi/DATA/Multiply/MCD43/', \
                                      emus_dir = '/home/ucfafyi/DATA/Multiply/emus/', s2_tile='29SQB', s2_psf=None)
    aero.solving_s2_aerosol()
    #solved  = aero.prepare_modis()
