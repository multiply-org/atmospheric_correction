#!/usr/bin/env python
import os
import sys
import argparse
from glob import glob
import numpy as np
from s2_aero import solve_aerosol
from s2_correction import atmospheric_correction
from downloaders import *
root = os.getcwd()
parser = argparse.ArgumentParser(description='Sentinel 2 Atmopsheric correction Excutable')
parser.add_argument('-f', "--file_path",      help='Sentinel 2 file path in the form of AWS: /directory/where/you/store/s2/data/29/S/QB/2017/9/4/0/',required=True)
parser.add_argument("-m", "--MCD43_file_dir", help="Directory where you store MCD43A1.006 data",        default=root + '/MCD43/')
parser.add_argument("-e", "--emulator_dir",   help="Directory where you store emulators.",              default=root + '/emus/')
parser.add_argument("-d", "--dem",            help="A global dem file, and a vrt file is recommonded.", default=root + '/eles/global_dem.vrt')
parser.add_argument("-w", "--wv_emulator",    help="A water vapour restrieval emulator.",               default=root + '/emus/wv_MSI_retrieval_S2A.pkl')
parser.add_argument("-c", "--cams",           help="Directory where you store cams data.",              default=root + '/cams/')
parser.add_argument("-s", "--satellite",      help="Data from which Satellite is used: S2A or S2B",     default='S2A')
parser.add_argument("--version",              action="version",                                         version='%(prog)s - Version 2.0')

args = parser.parse_args()
file_path = args.file_path
s2_toa_dir = '/'.join(file_path.split('/')[:-8])
day        = int(file_path.split('/')[-3])
month      = int(file_path.split('/')[-4])
year       = int(file_path.split('/')[-5])
s2_tile = ''.join(file_path.split('/')[-8:-5])
acquisition = file_path.split('/')[-2]
if not os.path.exists(file_path):
    down_s2(s2_tile, s2_toa_dir, year, month, day)

with open(file_path +'/tileInfo.json', 'rb') as f:
    for line in f.readlines():
        line = line.decode("utf-8")
        if 'productName' in line:
            args.satellite = line.split('"')[3].split('_')[0]

if len(glob(args.emulator_dir + '/*%s.pkl'%args.satellite)) < 3:
    down_s2_emus(args.emulator_dir, args.satellite)

cams_file = '%04d-%02d-%02d.nc'%(year, month, day)
if len(glob(os.path.join(args.cams, cams_file))) == 0:
    down_cams(args.cams, cams_file)

example_file = os.path.join(args.file_path, 'B04.jp2')
dem_dir = '/'.join(args.dem.split('/')[:-1])
url_dem = down_dem(dem_dir, example_file)
if url_dem:
    args.dem = url_dem    
down_s2_modis(args.MCD43_file_dir, args.file_path)
aero = solve_aerosol(year, month, day, \
                     s2_toa_dir  = s2_toa_dir,
                     mcd43_dir   = args.MCD43_file_dir, \
                     emus_dir    = args.emulator_dir, \
                     global_dem  = args.dem,\
                     wv_emus_dir = args.wv_emulator, \
                     cams_dir    = args.cams,\
                     s2_tile     = s2_tile, \
                     acquisition = acquisition,
                     satellite   = args.satellite, \
                     s2_psf      = None)
aero.solving_s2_aerosol()
atm = atmospheric_correction(year, \
                             month, \
                             day, \
                             s2_tile, \
                             acquisition = acquisition,
                             satellite   = args.satellite, \
                             s2_toa_dir  = s2_toa_dir,  \
                             global_dem  = args.dem, \
                             emus_dir    = args.emulator_dir)
atm.atmospheric_correction()  
