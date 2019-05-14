import os
import time
import gdal
import getpass
import logging
import requests
import numpy as np
from glob import glob
from functools import partial
from multiprocessing import Pool
from datetime import datetime, timedelta
from SIAC.modis_tile_cal import get_vector_hv, get_raster_hv
from os.path import expanduser
from SIAC.create_logger import create_logger
from six.moves import input

home = expanduser("~")
file_path = os.path.dirname(os.path.realpath(__file__))
logger = create_logger()

#print(file_path + '/data/.earthdata_auth')
#print(auth)


def find_local_files(aoi, obs_time, temporal_window = 16):
    days = [(obs_time - timedelta(days = int(i))).timetuple() for i in np.arange(temporal_window, 0, -1)] + \
            [(obs_time + timedelta(days = int(i))).timetuple() for i in np.arange(0, temporal_window+1,  1)]
    try:
        tiles = get_vector_hv(aoi)
    except:
        try:
            tiles = get_raster_hv(aoi)
        except:
            raise IOError('AOI has to be raster or vector object/files.')
    mcd43_file_pattern = 'MCD43A1.A{}{:03}.{}.006.*.hdf'
    expected_files = []
    for day in days:
        for tile in tiles:
            expected_files.append(mcd43_file_pattern.format(day.tm_year, day.tm_yday, tile))
    return expected_files



def find_files(aoi, obs_time, temporal_window = 16):
    days   = [(obs_time - timedelta(days = int(i))).strftime('%Y.%m.%d') for i in np.arange(temporal_window, 0, -1)] + \
             [(obs_time + timedelta(days = int(i))).strftime('%Y.%m.%d') for i in np.arange(0, temporal_window+1,  1)]
    try:
        tiles = get_vector_hv(aoi)
    except:
        try:
            tiles = get_raster_hv(aoi)
        except:
            raise IOError('AOI has to be raster or vector object/files.')
    fls = zip(np.repeat(tiles, len(days)), np.tile(days, len(tiles)))
    p = Pool(18)
    ret = p.map(get_one_tile, fls)
    p.close()
    p.join()
    return ret


def get_one_tile(tile_date):
    base = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD43A1.006/'
    tile, date = tile_date
    for j in range(100):
        r = requests.get(base + date)
        fname = [i.split('>')[-1] for i in r.content.decode().split('</') if (i[-3:]=='hdf') & (tile in i)]
        if len(fname) == 1:
            break
        else:
            #print(base + date)
            time.sleep(1)
    return base + date + '/' + fname[0]


def downloader(url_fname):
    if os.path.exists(file_path + '/data/.earthdata_auth'):
        try:
            username, password = np.loadtxt(file_path + '/data/.earthdata_auth', dtype=str)
            auth = tuple([username, password])
        except:
            logger.error('Please provide NASA Earthdata username and password for downloading MCD43 data, which can be applied here: https://urs.earthdata.nasa.gov.')
            username = input('Username for NASA Earthdata: ')
            password = getpass.getpass('Password for NASA Earthdata: ')
            up = username, password
            os.remove(file_path + '/data/.earthdata_auth')
            with open(file_path + '/data/.earthdata_auth', 'wb') as f:
                for i in up:
                    f.write((i + '\n').encode())
            auth = tuple([username, password])
    else:
        username = input('Username for NASA Earthdata: ')
        password = getpass.getpass('Password for NASA Earthdata: ')
        up = username, password
        with open(home + '/.earthdata_auth', 'wb') as f:
            for i in up:
                f.write((i + '\n').encode())
        auth = tuple([username, password])
    url, fname = url_fname
    with requests.Session() as s:
        s.max_redirects = 100000
        s.auth = auth
        r1     = s.get(url)
        r      = s.get(r1.url, stream=True, headers={'user-agent': 'My app'})
        if r.ok:
            remote_size = int(r.headers['Content-Length'])
            if os.path.exists(fname):
                local_size = os.path.getsize(fname)
                if local_size != remote_size:
                    os.remove(fname)
                    data = r.content
                    if len(data) == remote_size:
                        with open(fname, 'wb') as f:
                            f.write(data)
                    else:
                        raise IOError('Failed to download the whole file.')
            else:
                data = r.content
                if len(data) == remote_size:
                    with open(fname, 'wb') as f:
                        f.write(data)
                else:           
                    raise IOError('Failed to download the whole file.')
        else:
            print(r.content)


def daily_vrt(fnames_date, vrt_dir = None):
    temp1 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_%s'
    temp2 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_%s'

    fnames, date = fnames_date
    fnames = map(os.path.abspath, fnames)

    all_files = []
    for fname in fnames:
        all_files += glob(os.path.dirname(fname) + '/MCD43A1.A%s.h??v??.006.*.hdf'%date)
    if vrt_dir is None:
        vrt_dir = './MCD43_VRT/' 
    if not os.path.exists(vrt_dir):
        os.mkdir(vrt_dir)
    d = datetime.strptime(date, '%Y%j').strftime('%Y-%m-%d')
    date_dir = vrt_dir + '/' + '%s/'%d
    if not os.path.exists(date_dir):                               
        os.mkdir(date_dir) 
    for temp in [temp1, temp2]:                                                      
        for band in ['Band1','Band2','Band3','Band4','Band5','Band6','Band7', 'vis', 'nir', 'shortwave']:
            bs = []                                                                  
            for fname in all_files:                                                     
                bs.append(temp%(fname, band))
            vrt_data_set = gdal.BuildVRT(date_dir + '_'.join(['MCD43', date, bs[0].split(':')[-1]])+'.vrt', bs)
            if vrt_data_set is None:
                logger.info('Could not create vrt')
            else:
                vrt_data_set.FlushCache()


def get_local_MCD43(aoi, obs_time, mcd43_dir ='./MCD43/', vrt_dir ='./MCD43_VRT/'):
    logger.propagate = False
    expected_files = find_local_files(aoi, obs_time, 16)
    files = []
    for file in expected_files:
        globbed_files = glob(mcd43_dir + '/' + file)
        for globbed_file in globbed_files:
            if len(globbed_file) > 0:
                files.append(globbed_file)
    flist = np.array(files).flatten()
    all_dates = np.array([i.split('/')[-1] .split('.')[1][1:9] for i in flist])
    udates = np.unique(all_dates)
    fnames_dates =  [[flist[all_dates==date].tolist(),date] for date in udates]
    for fname_date in fnames_dates:
        daily_vrt(fname_date, vrt_dir)


def get_mcd43(aoi, obs_time, mcd43_dir = './MCD43/', vrt_dir = './MCD43_VRT/'):
    logger.propagate = False
    logger.info('Start downloading MCD43 for the AOI, this may take some time.')
    logger.info('Query files...')
    ret = find_files(aoi, obs_time, temporal_window = 16)
    url_fnames = [[i, mcd43_dir + '/' + i.split('/')[-1]] for i in ret]
    p = Pool(5)
    logger.info('Start downloading...')
    ret = p.map(downloader, url_fnames)
    p.close()
    p.join()
    flist = np.array(url_fnames)[:,1]
    all_dates = np.array([i.split('/')[-1] .split('.')[1][1:9] for i in flist])          
    udates = np.unique(all_dates)  
    fnames_dates =  [[flist[all_dates==date].tolist(),date] for date in udates]
    logger.info('Creating daily VRT...')
    par = partial(daily_vrt, vrt_dir = vrt_dir)
    p = Pool(len(fnames_dates))
    p.map(par, fnames_dates)
    p.close()
    p.join()


if __name__ == '__main__':
    aoi = '/home/ucfafyi/DATA/S2_MODIS/l_data/LC08_L1TP_014034_20170831_20170915_01_T1/AOI.json'
    aoi = '/home/ucfafyi/DATA/S2_MODIS/l_data/LC08_L1TP_014034_20170831_20170915_01_T1/aot.tif'
    obs_time = datetime(2017, 7, 8, 10, 8, 20)
    ret = get_mcd43(aoi, obs_time, mcd43_dir = '/home/ucfafyi/hep/MCD43/', vrt_dir = '/home/ucfafyi/DATA/Multiply/MCD43/')
