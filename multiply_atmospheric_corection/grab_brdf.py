import sys
sys.path.insert(0, 'util')
import gdal
import glob
import kernels
import numpy as np
from osgeo import osr
from smoothn import smoothn
from functools import partial
from multi_process import parmap
from reproject import reproject_data
from datetime import datetime, timedelta

x_step = -463.31271653
y_step = 463.31271653
m_y0, m_x0 = -20015109.354, 10007554.677

def r_modis(fname, xoff = None, yoff = None, xsize = None, ysize = None):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        if x_off is None:
            return g.ReadAsArray()
        elif g.RasterCount==1:
            return g.ReadAsArray(xoff, yoff, xsize, ysize)
        elif g.RasterCount>1:
            for band in range(g.RasterCount):
                band += 1
                rets.append(g.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize))
            return np.array(rets)
        else:
            raise IOError

def mtile_cal(lat, lon):
    # a function calculate the tile number for MODIS, based on the lat and lon
    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    modis_sinu = osr.SpatialReference() # define the SpatialReference object
    modis_sinu.ImportFromProj4 ( \
                    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    tx = osr.CoordinateTransformation( wgs84, modis_sinu)# from wgs84 to modis 
    ho,vo,z = tx.TransformPoint(lon, lat)# still use the function instead of using the equation....
    h = int((ho-m_y0)/(2400*y_step))
    v = int((vo-m_x0)/(2400*x_step))
    return h,v

def get_hv(example_file):
    g = gdal.Open(example_file)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterYSize, g.RasterXSize

    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    H_res_geo = osr.SpatialReference( )
    raster_wkt = g.GetProjection()
    H_res_geo.ImportFromWkt(raster_wkt)
    tx = osr.CoordinateTransformation(H_res_geo, wgs84)
    # so we need the four corners coordiates to check whether they are within the same modis tile
    (ul_lon, ul_lat, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])

    (lr_lon, lr_lat, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3] + geo_t[5]*y_size )

    (ll_lon, ll_lat, llz ) = tx.TransformPoint( geo_t[0] , \
                                          geo_t[3] + geo_t[5]*y_size )

    (ur_lon, ur_lat, urz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3]  )
    a0, b0 = None, None
    corners = [(ul_lon, ul_lat), (lr_lon, lr_lat), (ll_lon, ll_lat), (ur_lon, ur_lat)]
    tiles = []
    for i,j  in enumerate(corners):
        h, v = mtile_cal(j[1], j[0])
        tiles.append('h%02dv%02d'%(h,v))
    unique_tile = np.unique(np.array(tiles))
    return unique_tile

def array_to_raster(array, example_file):
    if array.ndim == 2:
        bands = 1
    elif array.ndim ==3:
        bands = array.shape[0]
    else:
        raise IOError('Only 2 or 3 D array is supported.')
    try:
        g = gdal.Open(example_file)
    except:
        g = example_file
    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', array.shape[-1], array.shape[-2], bands, gdal.GDT_Float64)
    ds.SetProjection(g.GetProjection())
    geotransform    = list(g.GetGeoTransform())  
    geotransform[1] = geotransform[1] * g.RasterXSize / (1. * array.shape[-1])
    geotransform[5] = geotransform[5] * g.RasterYSize / (1. * array.shape[-2])
    ds.SetGeoTransform(geotransform)
    if array.ndim == 3:
        for i in range(bands):
            ds.GetRasterBand(i+1).WriteArray(array[i])
    else:
         ds.GetRasterBand(1).WriteArray(array)
    return ds

def get_kk(angles):
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk

def MCD43_SurRef(MCD43_dir, example_file, year, doy, ang_files, sun_view_ang_scale=[1,1], bands = (7,), tolz = 0.001, reproject=False):
    f_temp = MCD43_dir + '/MCD43A1.A%s.%s.006*.hdf'
    temp1  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    
    unique_tile = get_hv(example_file)
    #print unique_tile
    date   = datetime.strptime('%d%03d'%(year, doy), '%Y%j')
    days   = [(date - timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    #data_f = [[temp1%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for band in bands for day in days]  
    #qa_f   = [[temp2%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for band in bands for day in days]
    try:
        fnames = np.array([[[temp1%(glob.glob(f_temp%(day, tile))[0], band), \
                             temp2%(glob.glob(f_temp%(day, tile))[0], band)] for \
                             tile in unique_tile] for band in bands for day in days]).transpose(0,2,1)
    except:
        print('Please download MCD43A1.006 files for tile(s): ', list(unique_tile), 'on dates', days)
        raise IOError
    g       = gdal.Open(example_file)
    mg      = reproject_data(example_file, gdal.BuildVRT('', list(fnames[0,0])), outputType = gdal.GDT_Float64).g
    temp_data = ~np.isnan(mg.ReadAsArray())
    geotransform = mg.GetGeoTransform() 
    xgeo = geotransform[0] + np.arange(0.5, mg.RasterXSize, 1) * geotransform[1]
    ygeo = geotransform[3] + np.arange(0.5, mg.RasterYSize, 1) * geotransform[5]
    xgeo = np.repeat(xgeo[None,...], mg.RasterYSize, axis=0)
    ygeo = np.repeat(ygeo[...,None], mg.RasterXSize, axis=1)
    m_proj = modis_sinu = osr.SpatialReference()
    m_proj.ImportFromProj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    h_proj = osr.SpatialReference() 
    h_proj.ImportFromWkt(gdal.Open(example_file).GetProjection())
    h_xgeo, h_ygeo, _ = np.array(osr.CoordinateTransformation(m_proj, h_proj).TransformPoints(list(zip(xgeo[temp_data], ygeo[temp_data])))).T
    geotransform = g.GetGeoTransform()
    hy = ((h_xgeo - geotransform[0])/geotransform[1]).astype(int)
    hx = ((h_ygeo - geotransform[3])/geotransform[5]).astype(int)
    hmask = (hx>=0) & (hx<g.RasterYSize) & (hy>=0) & (hy<g.RasterXSize)
    hy = hy[hmask]
    hx = hx[hmask]
    #print 'got vrt'
    max_x, max_y = np.array(np.where(temp_data)).max(axis=1)
    min_x, min_y = np.array(np.where(temp_data)).min(axis=1)
    xoff,  yoff  = int(min_y), int(min_x)
    xsize, ysize = int(max_y - min_y + 1), int(max_x - min_x + 1)
    #print 'read in data'
    f = lambda fname: [gdal.BuildVRT('', list(fname[0])).ReadAsArray(xoff, yoff, xsize, ysize), \
                       gdal.BuildVRT('', list(fname[1])).ReadAsArray(xoff, yoff, xsize, ysize)]
    #f      = lambda fname: gdal.BuildVRT('', fname).ReadAsArray(xoff, yoff, xsize, ysize)
    data, qa = np.array(parmap(f, fnames)).T
    data   = np.concatenate(data).reshape((len(bands), len(days), 3, ysize, xsize)).astype(float)
    data   = np.ma.array(data, mask = (data==32767)).astype(float)
    #test1, test2 = data.copy(), qa.copy()
    #global test1; global test2
    w      = 0.618034**np.concatenate(qa).reshape(len(bands), len(days), ysize, xsize).astype(float)
    #w      = 0.618034 ** qa.astype(float)
    f      = lambda band: np.array(smoothn(data[band[0],:,band[1],:,:], s=10., smoothOrder=1., \
                                   axis=0, TolZ=tolz, verbose=False, isrobust=True, W = w[band[0]]))[[0,3],]
    ba     = np.array([np.tile(range(len(bands)), 3), np.repeat(range(3), len(bands))]).T
    #print 'smoothing....'
    smed   = np.array(parmap(f, ba))
    #global smed
    dat    = np.concatenate(smed[:,0], axis=0).reshape(3, len(bands), len(days), ysize, \
                            xsize)[:,:,16, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    wei    = np.concatenate(smed[:,1], axis=0).reshape(3, len(bands), len(days), ysize, \
                            xsize)[:,:,16, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    #std    = data.std(axis = 1)[:, :, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    #print 'get angles...'
    sa_files, va_files = ang_files
    if isinstance(va_files[0], str):
        f   = lambda ang_file: reproject_data(ang_file, gdal.BuildVRT('', list(fnames[0,0])), outputType = gdal.GDT_Float64).data
        vas = np.array(parmap(f, va_files))
    elif isinstance(va_files[0], (np.ndarray, np.generic) ):
        f   = lambda array: reproject_data(array_to_raster(array, example_file), gdal.BuildVRT('', list(fnames[0,0])), outputType = gdal.GDT_Float64).data
        vas =  np.array(parmap(f, list(va_files)))
    vas = vas * sun_view_ang_scale[1]
    if isinstance(sa_files[0], str):
        f   = lambda ang_file: reproject_data(ang_file, gdal.BuildVRT('', list(fnames[0,0])), outputType = gdal.GDT_Float64).data
        sas = np.array(parmap(f, sa_files)) 
    elif isinstance(sa_files[0], (np.ndarray, np.generic) ):
        f   = lambda array: reproject_data(array_to_raster(array, example_file), gdal.BuildVRT('', list(fnames[0,0])), outputType = gdal.GDT_Float32).data
        sas =  np.array(parmap(f, list(sa_files)))
    if sas.shape[0] == 2:
        sas = np.repeat((sas * sun_view_ang_scale[0])[None, ...], len(bands), axis = 0)
    elif sas.shape[0] == len(bands):
        sas = sas * sun_view_ang_scale[0]
    else:
        raise IOError('Wrong shape of sun angles are given.')
    raa     = vas[:, 0, :, :] - sas[:, 0, :, :]
    angles  = vas[:, 1, temp_data], sas[:, 1, temp_data], raa[:, temp_data]
    kk      = get_kk(angles)
    k_vol   = kk.Ross
    k_geo   = kk.Li
    sur_ref = (dat[0] + dat[1]*k_vol + dat[2]*k_geo)*0.001
    wei     = 0.05 / wei
    #print wei
    unc     = np.sqrt(wei[0, :, :]**2 + (wei[1, :, :]**2)*k_vol**2 + (wei[2, :, :]**2)*k_geo**2)
    #unc     = np.sqrt((np.sqrt(std[:, 0, :]**2 + (std[:, 1, :]**2)*k_vol**2 + (std[:, 2, :]**2)*k_geo**2) * 0.001)**2 + \
    #                  (np.sqrt(wei[0, :, :]**2 + (wei[1, :, :]**2)*k_vol**2 + (wei[2, :, :]**2)*k_geo**2))**2) 
    unc     = np.minimum(unc, 0.1)
    #print unc
    if reproject:
        f_dat   = np.repeat(temp_data[None, ...], len(bands), axis=0).astype(float)
        f_dat[:]= np.nan 
        unc_dat = f_dat.copy() 
        f_dat  [:, temp_data] = sur_ref
        unc_dat[:, temp_data] = unc
        f       = lambda array: reproject_data(array_to_raster(array, gdal.BuildVRT('', list(fnames[0,0]))), example_file, outputType = gdal.GDT_Float32).data
        f_dat   = np.array(parmap(f, list(f_dat)))
        unc_dat = np.array(parmap(f, list(unc_dat))) 
        mask    = np.isnan(unc_dat) | (f_dat < 0.00001)
        f_dat[mask]   = np.nan
        unc_dat[mask] = np.nan
        unc_dat[unc_dat==0] = 0.1
        return f_dat, unc_dat
    else:
        lx, ly                = np.where(temp_data)
        sur_ref[sur_ref.mask] = np.nan
        unc[unc.mask]         = 0.1
        return sur_ref.data[:,hmask], unc.data[:,hmask], hx, hy, lx[hmask], ly[hmask], fnames[16,0]

if __name__ == '__main__':
    from datetime import datetime
    example_file = '/store/S2_data/29/S/QB/2017/1/12/0/B04.jp2'
    date = datetime.strptime('/'.join(example_file.split('/')[-5:-2]), '%Y/%m/%d')
    doy = date.timetuple().tm_yday  
    mcd43_dir = '/store/MCD43/'
    bands = [3,4,1, 2, 6,7]
    va_files  = ['/'.join(example_file.split('/')[:-1]) + '/angles/VAA_VZA_B%02d.img'%i for i in [2,3,4,8,11,12]]
    sza = np.zeros((10980, 10980))
    saa = sza.copy()
    from grab_s2_toa import read_s2
    tile =''.join(example_file.split('/')[-8:-5])
    s2 = read_s2('/store/S2_data/', tile, date.year, date.month, date.day, bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] )
    s2.get_s2_angles()
    sa_files = [s2.angles['saa'], s2.angles['sza']]
    ret = MCD43_SurRef(mcd43_dir, example_file, date.year, doy, [sa_files, va_files], sun_view_ang_scale=[1.,0.01], bands = bands, reproject=False)




