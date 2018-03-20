import re
import gdal
import numpy as np
def get_tile_lat_lon(fname):
    lats, lons = [], []
    for i in gdal.Info(fname).split('\n'):
        if ('Upper' in i) or ('Lower' in i):
            lon, lat = i.split('(')[-1].split(',')
            tlat = sum([j/60**i for i,j in enumerate(np.array(re.findall('(\d+)d(\d+)\\\'(\d+)\.(\d+)', lat.strip().replace(' ', ''))).astype(float).ravel())])
            tlon = sum([j/60**i for i,j in enumerate(np.array(re.findall('(\d+)d(\d+)\\\'(\d+)\.(\d+)', lon.strip().replace(' ', ''))).astype(float).ravel())])
            if 'S' in lat:
                tlat *= -1
            if 'W' in lon:
                tlon *=-1
            lats.append(tlat)
            lons.append(tlon)
    return lats, lons
