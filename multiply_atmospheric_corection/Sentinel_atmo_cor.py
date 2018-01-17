#/usr/bin/env python
#import os
import sys
import argparse
from glob import glob
from s2_aero import solve_aerosol
from s2_correction import atmospheric_correction

parser = argparse.ArgumentParser(description='Sentinel 2 Atmopsheric correction Excutable')
parser.add_argument('-f', "--file_path",      help='Sentinel 2 file path in the form of AWS: /directory/where/you/store/s2/data/29/S/QB/2017/9/4/0/',required=True)
parser.add_argument("-m", "--MCD43_file_dir", help="Directory where you store MCD43A1.006 data",        default='./MCD43/')
parser.add_argument("-e", "--emulator_dir",   help="Directory where you store emulators.",              default='./emus/')
parser.add_argument("-d", "--dem",            help="A global dem file, and a vrt file is recommonded.", default='./eles/global_dem.vrt')
parser.add_argument("-w", "--wv_emulator",    help="A water vapour restrieval emulator.",               default='./emus/wv_msi_retrieval.pkl')
parser.add_argument("-c", "--cams",           help="Directory where you store cams data.",              default='./cams/')
parser.add_argument("--version",              action="version",                                         version='%(prog)s - Version 2.0')

args = parser.parse_args()
file_path = args.file_path
s2_toa_dir = '/'.join(file_path.split('/')[:-8])
day        = int(file_path.split('/')[-3])
month      = int(file_path.split('/')[-4])
year       = int(file_path.split('/')[-5])
s2_tile = ''.join(file_path.split('/')[-8:-5])
#print glob(args.emulator_dir + '/*.pkl')
if len(glob(args.emulator_dir + '/*.pkl')) < 5:
   print('No emus, start downloading...')
   url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
   import requests
   req = requests.get(url)
   for line in req.text.split():
       if '.pkl' in line:
           fname   = line.split('"')[1].split('<')[0] 
           new_url = url + fname
           new_req = requests.get(new_url, stream=True) 
           print('downloading %s' % fname)
           with open(os.path.join(args.emulator_dir, fname), 'wb') as fp:
                 for chunk in new_req.iter_content(chunk_size=1024):
                     if chunk:
                         fp.write(chunk)

aero = solve_aerosol(year, month, day, \
                     s2_toa_dir  = s2_toa_dir,
                     mcd43_dir   = args.MCD43_file_dir, \
                     emus_dir    = args.emulator_dir, \
                     global_dem  = args.dem,\
                     wv_emus_dir = args.wv_emulator, \
                     cams_dir    = args.cams,\
                     s2_tile     = s2_tile, \
                     s2_psf      = None)

#aero = solve_aerosol(year, month, day, \
#                     s2_toa_dir = s2_toa_dir,
#                     mcd43_dir  = '/data/nemesis/MCD43/', \
#                     emus_dir   = '/home/ucfafyi/DATA/Multiply/emus/', s2_tile=s2_tile, s2_psf=None)
aero.solving_s2_aerosol()
atm = atmospheric_correction(year, \
                             month, \
                             day, \
                             s2_tile, \
                             s2_toa_dir  = s2_toa_dir,  \
                             global_dem  = args.dem, \
                             emus_dir    = args.emulator_dir)

#atm = atmospheric_correction(year, month, day, s2_tile, s2_toa_dir  = s2_toa_dir)
atm.atmospheric_correction()  
