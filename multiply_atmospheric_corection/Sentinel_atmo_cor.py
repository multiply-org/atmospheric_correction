#/usr/bin/env python
import os
import sys
from glob import glob
from s2_aerosol import solve_aerosol
from s2_correction import atmospheric_correction
file_path = sys.argv[1]
s2_toa_dir = '/'.join(file_path.split('/')[:-8])
day        = int(file_path.split('/')[-3])
month      = int(file_path.split('/')[-4])
year       = int(file_path.split('/')[-5])
s2_tile = ''.join(file_path.split('/')[-8:-5])
if len(glob('emus/*')) < 5:
   print 'No emus, start downloading...'
   url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
   import requests
   req = requests.get(url)
   for line in req.text.split():
       if '.pkl' in line:
           fname   = line.split('"')[1].split('<')[0] 
           new_url = url + fname
           new_req = requests.get(new_url, stream=True) 
           print 'downloading %s' % fname
           with open(os.path.join('emus/', fname), 'wb') as fp:
                 for chunk in new_req.iter_content(chunk_size=1024):
                     if chunk:
                         fp.write(chunk)

aero = solve_aerosol(year, month, day, \
                     s2_toa_dir  = s2_toa_dir,
                     mcd43_dir   = './MCD43/', \
                     emus_dir    = './emus/', \
                     global_dem  = './eles/global_dem.vrt',\
                     wv_emus_dir = './emus/wv_msi_retrieval.pkl', \
                     cams_dir    = './cams/',\
                     s2_tile     = s2_tile, \
                     s2_psf      = None)
aero.solving_s2_aerosol()
atm = atmospheric_correction(year, \
                             month, \
                             day, s2_tile, \
                             s2_toa_dir  = s2_toa_dir,  \
                             global_dem  = './eles/global_dem.vrt', \
                             emus_dir    = './emus/')
atm.atmospheric_correction()  


