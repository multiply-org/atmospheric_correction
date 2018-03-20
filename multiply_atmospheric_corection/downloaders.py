#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'util')
import gdal
import requests
import numpy as np
from glob import glob
from get_modis import get_modisfiles
from grab_brdf import get_hv
from multi_process import parmap
from functools import partial
from datetime import datetime, timedelta
from get_tile_lat_lon import get_tile_lat_lon
from sentinel_downloader import download_sentinel_amazon

def down_s2(tile, s2_dir, year, month, day):
    download_sentinel_amazon(datetime(year, month, day), \
                             s2_dir, tile = tile, clouds=100,\
                             end_date=datetime(year, month, day) )

def down_l8_aws(header, l8_dir):
    temp = 'https://landsat-pds.s3.amazonaws.com/c1/L8/%s/%s/%s/'
    path, row = header.split('_')[2][:3], header.split('_')[2][3:]
    root = temp%(path, row, header)
    footer = ['B%d.TIF'%(band+1) for band in range(11)] + ['BQA.TIF', 'MTL.txt', 'ANG.txt']
    fnames = [header + '_' + foot for foot in footer]
    par = partial(downloader, url_root = root, file_dir = l8_dir)
    parmap(par, fnames)
    ndir = l8_dir + '/' + header
    if not os.path.exists(ndir):
        os.mkdir(ndir)
    for i in fnames:
        os.rename(l8_dir + '/' + i, ndir + '/' + i)

#https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/015/035/LC08_L1TP_015035_20160124_20170224_01_T1/LC08_L1TP_015035_20160124_20170224_01_T1_B1.TIF
def down_l8_google(header, l8_dir):
    temp = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/%s/%s/%s/'
    path, row = header.split('_')[2][:3], header.split('_')[2][3:]
    root = temp%(path, row, header)
    footer = ['B%d.TIF'%(band+1) for band in range(11)] + ['BQA.TIF', 'MTL.txt', 'ANG.txt']
    fnames = [header + '_' + foot for foot in footer]
    par = partial(downloader, url_root = root, file_dir = l8_dir)
    parmap(par, fnames)
    ndir = l8_dir + '/' + header
    if not os.path.exists(ndir):
        os.mkdir(ndir)
    for i in fnames:
        os.rename(l8_dir + '/' + i, ndir + '/' + i)


def downloader(fname, url_root, file_dir):
    new_url = url_root + fname
    new_req = requests.get(new_url, stream=True)
    if new_req.ok:
        print('downloading %s' % fname)
        with open(os.path.join(file_dir, fname), 'wb') as fp:
            for chunk in new_req.iter_content(chunk_size=1024):
                if chunk:
                    fp.write(chunk)
    else:
        print('Requests failed.')

def down_s2_emus(emus_dir, satellite):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
    req = requests.get(url)
    for line in req.text.split():
        if '.pkl' in line:
            fname   = line.split('"')[1].split('<')[0]
            if satellite in fname:
                downloader(fname, url, emus_dir)

def down_l8_emus(emus_dir):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
    req = requests.get(url)
    for line in req.text.split():
        if '.pkl' in line:
            fname   = line.split('"')[1].split('<')[0]
            if 'OLI' in fname:
                downloader(fname, url, emus_dir)

def down_cams(cams_dir, cams_file):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/cams/'
    downloader(cams_file, url, cams_dir)

def down_dem(eles_dir, example_file):
    lats, lons = get_tile_lat_lon(example_file)
    #url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/eles/'
    min_lon, max_lon = np.floor(min(lons)), np.ceil(max(lons))
    min_lat, max_lat = np.floor(min(lats)), np.ceil(max(lats))
    rebuilt_vrt = False
    for la in np.arange(min_lat, max_lat + 1):
        for lo in np.arange(min_lon, max_lon + 1):
            if la>=0:
                lat_str = 'N%02d'%(int(abs(la)))
            else:
                lat_str = 'S%02d'%(int(abs(la)))
            if lo>=0:
                lon_str = 'E%03d'%(int(abs(lo)))
            else:
                lon_str = 'W%03d'%(int(abs(lo)))
            fname = 'ASTGTM2_%s%s_dem.tif'%(lat_str, lon_str)
            if len(glob(os.path.join(eles_dir, fname)))==0:
                return '/vsicurl/http://www2.geog.ucl.ac.uk/~ucfafyi/eles/global_dem.vrt'
            else:
                return 0
                #downloader(fname, url, eles_dir)
                #rebuilt_vrt = True
    #if rebuilt_vrt:
    #    gdal.BuildVRT(eles_dir + '/global_dem.vrt', glob(eles_dir +'/*.tif'), outputBounds = (-180,-90,180,90)).FlushCache()
    
def down_s2_modis(modis_dir, s2_dir):
    date  = datetime.strptime('-'.join(s2_dir.split('/')[-5:-2]), '%Y-%m-%d')
    tiles = get_hv(s2_dir+'/B04.jp2')
    days   = [(date - timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    fls = zip(np.repeat(tiles, len(days)), np.tile(days, len(tiles)))
    f = lambda fl: helper(fl, modis_dir)
    parmap(f, fls, nprocs=5)

def down_l8_modis(modis_dir, l8_file):
    date = datetime.strptime(l8_file.split('/')[-1].split('_')[3], '%Y%m%d')
    tiles = get_hv(l8_file)
    days   = [(date - timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = int(i))).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    fls = zip(np.repeat(tiles, len(days)), np.tile(days, len(tiles)))
    f = lambda fl: helper(fl, modis_dir)
    parmap(f, fls, nprocs=5)

def helper(fl, modis_dir):
    f_temp = modis_dir + '/MCD43A1.A%s.%s.006*.hdf'
    tile, day = fl
    if len(glob(f_temp%(day, tile))) == 0:
        year, doy = int(day[:4]), int(day[4:])
        get_modisfiles( 'MOTA', 'MCD43A1.006', year, tile, None,
                         doy_start= doy, doy_end = doy + 1, out_dir = modis_dir, verbose=0)

