#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'util')
import gdal
import json
import datetime
import logging
import numpy as np
from grab_l8_toa import read_l8
from glob import glob
from scipy import signal, ndimage
try:
    import cPickle as pkl
except:
    import pickle as pkl
from osgeo import osr
#from smoothn import smoothn
from multi_process import parmap
from scipy.ndimage import binary_dilation, binary_erosion
from reproject import reproject_data
from scipy.interpolate import griddata
from scipy.fftpack import dct, idct
from grab_brdf import MCD43_SurRef
from atmo_solver import solving_atmo_paras
from psf_optimize import psf_optimize
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import linregress

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,
                 year,
                 month,
                 day,
                 emus_dir    = './emus/',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 mod08_dir   = '/home/ucfafyi/DATA/Multiply/mod08/',                 
                 myd08_dir   = '/home/ucfafyi/DATA/Multiply/myd08/',                 
                 l8_tile     = (123, 34),
                 l8_psf      = None,
                 qa_thresh   = 0.099,
                 aero_res    = 1000, # resolution for aerosol retrival in meters should be larger than 500
                 ):

        self.year        = year
        self.month       = month
        self.day         = day
        self.date        = datetime.datetime(self.year, self.month, self.day)
        self.doy         = self.date.timetuple().tm_yday
        self.mcd43_dir   = mcd43_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.l8_toa_dir  = l8_toa_dir
        self.global_dem  = global_dem
        self.cams_dir    = cams_dir
        self.mod08_dir   = mod08_dir
        self.myd08_dir   = myd08_dir
        self.l8_tile     = l8_tile
        self.l8_psf      = l8_psf
        self.bands       = [2, 3, 4, 5, 6, 7]
        self.band_indexs = [1, 2, 3, 4, 5, 6]
        self.boa_bands   = [469, 555, 645, 869, 1640, 2130]
        self.aero_res    = aero_res
        self.mcd43_tmp   = '%s/MCD43A1.A%d%03d.%s.006.*.hdf'
        self.spectral_transform = [[1.0425211806,      1.03763437575,     1.02046102587,     0.999167480738,  1.00072211685,    0.955317665361  ], 
                                   [0.000960797104206, -0.00263498369438, -0.00179952807464, 0.0018999624331, -0.0072213121738, 0.00782954328347]] 
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_optimization_xap.pkl'%(self.sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_optimization_xbp.pkl'%(self.sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_optimization_xcp.pkl'%(self.sensor))[0]
        #f = lambda em: pkl.load(open(em, 'rb'))
        if sys.version_info >= (3,0):
            f = lambda em: pkl.load(open(em, 'rb'), encoding = 'latin1')
        else:
            f = lambda em: pkl.load(open(em, 'rb'))
        self.emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def gaussian(self, xstd, ystd, angle, norm = True):
        win = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        winx = int(round(win*(2**0.5)))
        winy = int(round(win*(2**0.5)))
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[int(center[0]-win/2): int(center[0]+win/2), int(center[1]-win/2) : int(center[1]+win/2)]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus
        
    def _save_img(self, fnames, refs, example_file):
        g            = gdal.Open(example_file)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(fnames, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)

    def _save_band(self, band_ref, projection, geotransform):
        fname, ref = band_ref
        nx, ny = ref.shape
        dst_ds = gdal.GetDriverByName('GTiff').Create(fname, ny, nx, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(ref)
        dst_ds.FlushCache()
        dst_ds = None

    def _mcd08_aot(self,):
        temp = 'HDF4_EOS:EOS_GRID:"%s":mod08:Aerosol_Optical_Depth_Land_Ocean_Mean'
        try:
            g = gdal.Open(temp%glob('%s/MOD08_D3.A2016%03d.006.*.hdf'%(self.mod08_dir, self.doy))[0])
            dat = reproject_data(g, self.example_file, outputType= gdal.GDT_Float32).data * g.GetRasterBand(1).GetScale() + g.GetRasterBand(1).GetOffset()
            dat[dat<=0]  = np.nan
            dat[dat>1.7] = np.nan 
            mod08_aot = np.nanmean(dat)
        except:
            mod08_aot = np.nan
        try:
            g1 = gdal.Open(temp%glob('%s/MYD08_D3.A2016%03d.006.*.hdf'%(self.myd08_dir, self.doy))[0])
            dat1 = reproject_data(g1, self.example_file, outputType= gdal.GDT_Float32).data * g1.GetRasterBand(1).GetScale() + g1.GetRasterBand(1).GetOffset()
            dat1[dat1<=0]  = np.nan
            dat1[dat1>1.7] = np.nan
            myd08_aot = np.nanmean(dat1)
        except:
            myd08_aot = np.nan
        return mod08_aot, myd08_aot

    def _get_psf(self,):
        if self.l8_psf is not None:
            xstd, ystd, ang, xs, ys = self.l8_psf
        else:
            self.logger.info('No PSF parameters specified, start solving.')
            xstd, ystd  = 12., 20.
            psf         = psf_optimize(self.toa[-1].data, [self.Hx, self.Hy], np.ma.array(self.boa[-1]), self.boa_qa[-1], self.bad_pix, 0.1, xstd=xstd, ystd= ystd)
            xs, ys      = psf.fire_shift_optimize()
            ang         = 0
            self.logger.info('Solved PSF parameters are: %.02f, %.02f, %d, %d, %d, and the correlation is: %.03f.' \
                              %(xstd, ystd, 0, xs, ys, 1-psf.costs.min()))

        shifted_mask = np.logical_and.reduce(((self.Hx+int(xs)>=0),
                                              (self.Hx+int(xs)<self.full_res[0]),
                                              (self.Hy+int(ys)>=0),
                                              (self.Hy+int(ys)<self.full_res[1])))
                               
        self.Hx, self.Hy = self.Hx[shifted_mask]+int(xs), self.Hy[shifted_mask]+int(ys)
        self.boa      = self.boa   [:, shifted_mask]
        self.boa_qa   = self.boa_qa[:, shifted_mask]

    def _extend_vals(self, val):
        if val.ndim == 2:
            temp            = np.zeros((self.efull_res, self.efull_res))
            temp[:]         = np.nan
            temp[:self.full_res[0], :self.full_res[1]] = val
        elif val.ndim == 3:
            temp            = np.zeros((val.shape[0], self.efull_res, self.efull_res))
            temp[:]         = np.nan
            temp[:, :self.full_res[0], :self.full_res[1]] = val 
        else:
            raise IOError('Only two and three dimensions array is supported.')
        return temp

    def _mask_bad_pix(self, l8):
        self.cloud    = l8._get_qa()
        ndvi          = (self.toa[3] - self.toa[2])/(1. * self.toa[3] + self.toa[2])
        water_mask    =  ((ndvi < 0.01) & (self.toa[3] < 0.11)) | ((ndvi < 0.1) & (self.toa[3] < 0.05)) | (self.toa[-1] < 0.0001)
        self.full_res = self.cloud.shape
        border_mask   = np.zeros(self.full_res).astype(bool)
        border_mask[[0, -1], :] = True
        border_mask[:, [0, -1]] = True
        xstd, ystd    = 12., 20.
        ker_size      = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        ewater_mask   = binary_erosion (water_mask, structure = np.ones((3,3)).astype(bool), iterations=5).astype(bool) 
        ecloud        = binary_erosion (self.cloud, structure = np.ones((3,3)).astype(bool), iterations=10).astype(bool)
        self.bad_pix  = binary_dilation(ecloud | border_mask | ewater_mask, structure = np.ones((3,3) ).astype(bool), \
                        iterations=int(ker_size/2+10)).astype(bool)
    
    def __sorting_data(self,):
        self.block_size = int(np.ceil(1. * self.aero_res / 30.))
        self.num_blocks = int(np.ceil(max(self.full_res) / (1. * self.block_size)))
        self.efull_res  = self.block_size * self.num_blocks
        shape1          =                    (self.num_blocks, self.block_size, self.num_blocks, self.block_size)
        shape2          = (self.vza.shape[0], self.num_blocks, self.block_size, self.num_blocks, self.block_size)
        self.ele        = np.nanmean(self._extend_vals(self.ele ).reshape(shape1), axis=(3,1))
        self.aot        = np.nanmean(self._extend_vals(self.aot ).reshape(shape1), axis=(3,1))
        self.tcwv       = np.nanmean(self._extend_vals(self.tcwv).reshape(shape1), axis=(3,1))
        self.tco3       = np.nanmean(self._extend_vals(self.tco3).reshape(shape1), axis=(3,1))
        self.saa        = np.nanmean(self._extend_vals(self.saa ).reshape(shape1), axis=(3,1))
        self.sza        = np.nanmean(self._extend_vals(self.sza ).reshape(shape1), axis=(3,1))
        self.vaa        = np.nanmean(self._extend_vals(self.vaa ).reshape(shape2), axis=(4,2))
        self.vza        = np.nanmean(self._extend_vals(self.vza ).reshape(shape2), axis=(4,2))
        self.aot_unc    = np.ones(self.aot.shape)  * 0.4
        self.tcwv_unc   = np.ones(self.tcwv.shape) * 0.1
        self.tco3_unc   = np.ones(self.tco3.shape) * 0.05
        self.aot[:]    = np.nanmean(self.aot)

    def _get_boa(self,l8):
        if len(glob(self.l8_toa_dir + '/MCD43_%s.npz'%(l8.header))) == 0:
            self.logger.warning('Since no Low resolution processing chain produced BRDF product is available at the moment, ' + \
                                'gap filled MCD43 is used and it may take a long time do the gap filling.')
            boa, unc, hx, hy, lx, ly, flist = MCD43_SurRef(self.mcd43_dir, \
                                                           self.example_file, \
                                                           self.year, \
                                                           self.doy, \
                                                           [l8.saa_sza, l8.vaa_vza], \
                                                           sun_view_ang_scale = [0.01, 0.01], \
                                                           bands = [3,4,1,2,6,7], tolz=0.003)
            np.savez(self.l8_toa_dir + '/MCD43_%s.npz'%l8.header, boa = boa, \
                     unc = unc, hx = hx, hy = hy, lx = lx, ly = ly, flist = flist) 
        else:           
            f = np.load(self.l8_toa_dir + '/MCD43_%s.npz'%l8.header, encoding='latin1')
            boa, unc, hx, hy, lx, ly, flist = f['boa'], f['unc'], f['hx'], f['hy'], f['lx'], f['ly'], f['flist']

        return boa, unc, hx, hy, lx, ly, flist

    def _resample_angles_and_elevation(self, l8):
        #self.block_size = int(np.ceil(1. * self.aero_res / 30.))
        self.num_blocks = int(np.ceil(max(self.full_res) * 30 / (1. * self.aero_res)))
        self.efull_res  = int(self.aero_res * self.num_blocks / 30.)
        f        = lambda fname: reproject_data(fname, \
                                                self.example_file, \
                                                xRes=self.aero_res, \
                                                yRes=self.aero_res, \
                                                xSize=self.num_blocks, \
                                                ySize=self.num_blocks, \
                                                srcNodata = [-32768, 0],\
                                                outputType= gdal.GDT_Float32, \
                                                resample = gdal.GRIORA_NearestNeighbour).data / 100.
        
        ret      = np.array(parmap(f, l8.vaa_vza))
        self.vaa, self.vza = ret[:, 0], ret[:, 1]
        self.vaa[self.vaa<-180] = np.nan
        self.vza[self.vza<=0]   = np.nan 
        
        #fnames   = l8.saa_sza + [self.global_dem, ]
        #ret      = parmap(f, fnames)
        self.saa, self.sza = f(l8.saa_sza)
        self.saa[self.saa<-180] = np.nan
        self.sza[self.sza<=0]   = np.nan 

        f        = lambda fname: reproject_data(fname, \
                                                self.example_file, \
                                                xRes=self.aero_res, \
                                                yRes=self.aero_res, \
                                                xSize=self.num_blocks, \
                                                ySize=self.num_blocks, \
                                                srcNodata = [-32768,0],\
                                                outputType= gdal.GDT_Float32,\
                                                resample = gdal.GRIORA_Bilinear).data
        self.ele = f(self.global_dem) / 1000.

        #self.ele = ret[1] / 1000.

    def _sorting_data(self,):
        self.vaa = np.array(parmap(self._fill_nan, list(self.vaa))) 
        self.vza = np.array(parmap(self._fill_nan, list(self.vza)))
        self.saa, self.sza, self.ele, self.aot, self.tcwv, self.tco3 = \
        parmap(self._fill_nan, [self.saa, self.sza, self.ele, self.aot, self.tcwv, self.tco3])
        self.aot_unc    = np.ones(self.aot.shape)  * 0.4
        self.tcwv_unc   = np.ones(self.tcwv.shape) * 0.1
        self.tco3_unc   = np.ones(self.tco3.shape) * 0.05
        self.aot[:]    = np.nanmean(self.aot)
       
    def _fill_nan(self, array):
        x_shp, y_shp = array.shape
        mask  = ~np.isnan(array)   
        valid = np.array(np.where(mask)).T
        value = array[mask]
        mesh  = np.repeat(range(x_shp), y_shp).reshape(x_shp, y_shp), \
                np.tile  (range(y_shp), x_shp).reshape(x_shp, y_shp)
        array = griddata(valid, value, mesh, method='nearest')
        return array

    def _get_convolved_toa(self,):
        self.bad_pixs = self.bad_pix[self.Hx, self.Hy]
        xstd, ystd = 12., 20.
        x_shp, y_shp = self.toa[0].shape
        xgaus  = np.exp(-2.*(np.pi**2)*(xstd**2)*((0.5 * np.arange(x_shp) / x_shp)**2))
        ygaus  = np.exp(-2.*(np.pi**2)*(ystd**2)*((0.5 * np.arange(y_shp) / y_shp)**2))
        gaus_2d = np.outer(xgaus, ygaus)
        f   = lambda img: idct(idct(dct(dct(img, axis=0, norm = 'ortho'), axis=1, norm='ortho') * gaus_2d, \
                                            axis=1, norm='ortho'), axis=0, norm='ortho')[self.Hx, self.Hy]
        self.toa  = np.array(parmap(f, list(self.toa))) 

    def _get_valid_pixels(self,):
        qua_mask = np.all(self.boa_qa <= self.qa_thresh, axis = 0)
        boa_mask = np.all(~self.boa.mask,axis = 0 ) &\
                          np.all(self.boa >= 0.001, axis = 0) &\
                          np.all(self.boa < 1, axis = 0)
        toa_mask = ~self.bad_pixs
        self.l8_mask = boa_mask & toa_mask & qua_mask
        self.Hx      = self.Hx       [self.l8_mask]
        self.Hy      = self.Hy       [self.l8_mask]
        self.toa     = self.toa   [:, self.l8_mask]
        self.boa     = self.boa   [:, self.l8_mask]
        self.boa_unc = self.boa_qa[:, self.l8_mask]
        tempm  = np.zeros((self.efull_res, self.efull_res))
        tempm[self.Hx, self.Hy] = 1
        #tempm = tempm.reshape(self.num_blocks, self.block_size, \
        #                      self.num_blocks, self.block_size).astype(int).sum(axis=(3,1))
        self.mask = tempm > 0 
       

    def _l8_aerosol(self,):
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        l8             = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        self.l8_header = l8.header
        self.example_file = glob(self.l8_toa_dir + '/%s_[b, B]%d.[t, T][i, I][f, F]'%(l8.header, 1))[0]

        self.logger.info('Reading in TOA reflectance.')
        self.sen_time = l8.sen_time
        self.toa      = l8._get_toa()

        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()

        self.logger.info('Getting simulated surface reflectance.')
        boa, unc, hx, hy, lx, ly, flist = self._get_boa(l8)
        self.Hx, self.Hy = hx, hy

        self.logger.info('Update cloud mask.')
        self._mask_bad_pix(l8)

        self.logger.info('Applying spectral transform.')
        self.boa_qa = np.ma.array(unc)
        self.boa    = np.ma.array(boa)*np.array(self.spectral_transform)[0][...,None] + \
                                       np.array(self.spectral_transform)[1][...,None]

        self.logger.info('Getting control variables.')
        self._resample_angles_and_elevation(l8)

        self.logger.info('Getting pripors from ECMWF forcasts.')
        self.aot, self.tcwv, self.tco3    = np.array(self._read_cams(self.example_file))
        self.logger.info('Mean values of priors are: %.03f, %.03f, %.03f'%\
                         (np.nanmean(self.aot), np.nanmean(self.tcwv), np.nanmean(self.tco3)))

        #self.logger.info('Getting DDV aot prior')
        #self._get_ddv_aot(self.toa, l8, self.tcwv, self.tco3, ele_data)
        del l8
        self.logger.info('Sorting data.')
        self._sorting_data()

        self.logger.info('Applying PSF model.')
        self._get_psf()

        self.logger.info('Getting the convolved TOA reflectance.')
        self._get_convolved_toa()

        self.logger.info('Getting valid pixels')
        self._get_valid_pixels()
        
        self.logger.info('Solving...')
        self.aero = solving_atmo_paras(self.boa,
                                       self.toa,
                                       self.sza,
                                       self.vza,
                                       self.saa,
                                       self.vaa,
                                       self.aot,
                                       self.tcwv,
                                       self.tco3,
                                       self.ele,
                                       self.aot_unc,
                                       self.tcwv_unc,
                                       self.tco3_unc,
                                       self.boa_unc,
                                       self.Hx, self.Hy,
                                       self.mask,
                                       (self.efull_res, self.efull_res),
                                       self.aero_res,
                                       self.emus,
                                       self.band_indexs,
                                       self.boa_bands,
                                       gamma = 10,
                                       alpha   = -1.6,
                                       pix_res = 30)
        solved    = self.aero._multi_grid_solver()
        return solved
 
    def _get_ddv_aot(self, toa, l8, tcwv, tco3, ele_data):

        ndvi_mask = (((toa[3] - 0.5*toa[5])/(toa[3] + 0.5*toa[5])) > 0.5) & (toa[5] > 0.01) & (toa[5] < 0.25) & (~self.bad_pix)
        if ndvi_mask.sum() < 100:
            self.logger.info('No enough DDV found in this sence for aot restieval, and only cams prediction used.') 
        else:
            Hx, Hy = np.where(ndvi_mask)
            if ndvi_mask.sum() > 1000: 
                random_choice     = np.random.choice(len(Hx), 1000, replace=False)
                random_choice.sort()
                Hx, Hy            = Hx[random_choice], Hy[random_choice]
                ndvi_mask[:]      = False
                ndvi_mask[Hx, Hy] = True
            Hx, Hy    = np.where(ndvi_mask)
            blue_vza  = np.cos(np.deg2rad(self.vza[0, Hx, Hy]))
            blue_sza  = np.cos(np.deg2rad(self.sza[0, Hx, Hy]))
            red_vza   = np.cos(np.deg2rad(self.vza[2, Hx, Hy])) 
            red_sza   = np.cos(np.deg2rad(self.sza[0, Hx, Hy]))
            blue_raa  = np.cos(np.deg2rad(self.vaa[0, Hx, Hy] - self.saa[0, Hx, Hy]))
            red_raa   = np.cos(np.deg2rad(self.vaa[2, Hx, Hy] - self.saa[0, Hx, Hy]))
            red, blue = toa[2, Hx, Hy], toa[0, Hx, Hy]
            swif      = toa[5, Hx, Hy]
            red_emus  = np.array(self.emus)[:, 3]
            blue_emus = np.array(self.emus)[:, 1]

            zero_aod    = np.zeros_like(red)
            red_inputs  = np.array([red_sza,  red_vza,  red_raa,  zero_aod, tcwv[Hx, Hy], tco3[Hx, Hy], ele_data[Hx, Hy]])
            blue_inputs = np.array([blue_sza, blue_vza, blue_raa, zero_aod, tcwv[Hx, Hy], tco3[Hx, Hy], ele_data[Hx, Hy]])
            
            p           = np.r_[np.arange(0.001, 1., 0.02), np.arange(1., 1.5, 0.05),  np.arange(1.5, 2., 0.1)]
            f           =  lambda aot: self._ddv_cost(aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus)
            costs       = parmap(f, p)
            min_ind     = np.argmin(costs)
            self.logger.info('DDV solved aod is %.03f, and it will used as the mean value for cams prediction.'% p[min_ind])
            self.aot[:] = p[min_ind]

    def _ddv_cost(self, aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus):
        blue_inputs[3, :] = aot
        red_inputs [3, :] = aot
        blue_xap_emu, blue_xbp_emu, blue_xcp_emu = blue_emus
        red_xap_emu,  red_xbp_emu,  red_xcp_emu  = red_emus
        blue_xap, blue_xbp, blue_xcp             = blue_xap_emu.predict(blue_inputs.T)[0], \
                                                   blue_xbp_emu.predict(blue_inputs.T)[0], \
                                                   blue_xcp_emu.predict(blue_inputs.T)[0]
        red_xap,  red_xbp,  red_xcp              = red_xap_emu.predict(red_inputs.T)  [0], \
                                                   red_xbp_emu.predict(red_inputs.T)  [0], \
                                                   red_xcp_emu.predict(red_inputs.T)  [0]
        y        = blue_xap * blue - blue_xbp
        blue_sur = y / (1 + blue_xcp * y)
        y        = red_xap * red - red_xbp
        red_sur  = y / (1 + red_xcp * y)
        blue_dif = 0 #(blue_sur - 0.25 * swif)**2
        red_dif  = 0 #(red_sur  - 0.5  * swif)**2
        rb_dif   = (blue_sur  - 0.5  * red_sur)**2
        cost     = 0.5 * (blue_dif + red_dif + rb_dif)
        return cost.sum()

    def _read_cams(self, example_file, parameters = ['aod550', 'tcwv', 'gtco3'], this_scale=[1., 0.1, 46.698]):
        netcdf_file = datetime.datetime(self.sen_time.year, self.sen_time.month, \
                                        self.sen_time.day).strftime("%Y-%m-%d.nc")
        template    = 'NETCDF:"%s":%s'
        ind         = np.abs((self.sen_time.hour  + self.sen_time.minute/60. + \
                              self.sen_time.second/3600.) - np.arange(0,25,3)).argmin()
        sr         = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        proj       = sr.ExportToWkt()
        results = []
        for i, para in enumerate(parameters):
            fname   = template%(self.cams_dir + '/' + netcdf_file, para)
            g       = gdal.Open(fname)
            g.SetProjection(proj)
            sub     = g.GetRasterBand(int(ind+1))
            offset  = sub.GetOffset()
            scale   = sub.GetScale()
            bad_pix = int(sub.GetNoDataValue())
            rep_g   = reproject_data(g, example_file,
                                     xRes=self.aero_res, \
                                     yRes=self.aero_res, \
                                     xSize=self.num_blocks, \
                                     ySize=self.num_blocks, \
                                     outputType= gdal.GDT_Float32,\
                                     resample = gdal.GRIORA_Bilinear).g

            data    = rep_g.GetRasterBand(int(ind+1)).ReadAsArray()
            data    = data*scale + offset
            #mask    = (data == (bad_pix*scale + offset)) | np.isnan(data)
            #if mask.sum()>=1:
            #    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            results.append(data*this_scale[i])
        return results

    def _save_posterior(self, name_array):
        name, array = name_array
        xmin, ymax  = self._example_g.GetGeoTransform()[0], \
                      self._example_g.GetGeoTransform()[3]
        projection  = self._example_g.GetProjection()
        xres, yres = self.aero_res, self.aero_res
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        nx, ny = self.num_blocks, self.num_blocks
        outputFileName = self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, name) 
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)   
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray(array)
        dst_ds.FlushCache()                     
        dst_ds = None
        return array
        
    def solving_l8_aerosol(self,):
        self.logger = logging.getLogger('Landsat 8 Atmospheric Correction')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.propagate = False

        self.sensor  = 'OLI'
        self.logger.info('Doing Landsat 8 tile: (%s, %s) on %d-%02d-%02d.' \
                          % (self.l8_tile[0], self.l8_tile[1], self.year, self.month, self.day))
        ret         = np.array(self._l8_aerosol())
        self.solved = ret[0].reshape(3, self.num_blocks, self.num_blocks)
        self.unc    = ret[1].reshape(3, self.num_blocks, self.num_blocks)
        self.logger.info('Finished retrieval and saving them into local files.')
        self._example_g = gdal.Open(self.example_file)
        para_names = 'aot', 'tcwv', 'tco3', 'aot_unc', 'tcwv_unc', 'tco3_unc'
        arrays          = list(self.solved ) + list(self.unc)
        name_arrays     = zip(para_names, arrays)
        ret = parmap(self._save_posterior, name_arrays)
        self.post_aot,     self.post_tcwv,     self.post_tco3, \
        self.post_aot_unc, self.post_tcwv_unc, self.post_tco3_unc = ret 

if __name__ == '__main__':
    aero = solve_aerosol(2017, 7, 10, l8_tile = (123, 34), mcd43_dir   = '/data/selene/ucfajlg/Hebei/MCD43/', l8_toa_dir = '/home/ucfafyi/DATA/S2_MODIS/l_data/LC08_L1TP_123034_20170710_20170725_01_T1')
    aero.solving_l8_aerosol()
