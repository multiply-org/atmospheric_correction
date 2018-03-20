#!/usr/bin/env python

import os
from glob import glob
import argparse
from l8_aerosol import solve_aerosol
from l8_correction import atmospheric_correction
from downloaders import *

root = os.getcwd()
parser = argparse.ArgumentParser(description='Landsat 8 Atmopsheric correction Excutable')
parser.add_argument('-f', "--l8_file",        help='A L8 file and the file can be not downloaded',      required = True)
parser.add_argument("-m", "--MCD43_file_dir", help="Directory where you store MCD43A1.006 data",        default  = root +'/MCD43/')
parser.add_argument("-e", "--emulator_dir",   help="Directory where you store emulators.",              default  = root + '/emus/')
parser.add_argument("-d", "--dem",            help="A global dem file, and a vrt file is recommonded.", default  = root + '/eles/global_dem.vrt')
parser.add_argument("-c", "--cams",           help="Directory where you store cams data.",              default  = root + '/cams/')
parser.add_argument("--version",              action="version",                                         version='%(prog)s - Version 2.0')
args = parser.parse_args()



l8_toa_dir       = '/'.join(args.l8_file.split('/')[:-1])
header           = '_'.join(args.l8_file.split('/')[-1].split('_')[:7])
if (len(glob(l8_toa_dir + '/' + header + '_[b, B]*.[T, t][I, i][F, f]')) >= 12):
    l8_toa_dir   =      l8_toa_dir + '/' + header + '/'
elif (len(glob(args.l8_file + '/' + header + '_[b, B]*.[T, t][I, i][F, f]')) >= 12):
    l8_toa_dir   = args.l8_file
else:
    down_l8_google(header, l8_toa_dir)
    #args.l8_file = glob(l8_toa_dir + '/' + header + '/' + header + '_[b, B]1.[t, T][i, I][f, F]')[0]
    l8_toa_dir   =      l8_toa_dir + '/' + header + '/'
args.l8_file     = glob(l8_toa_dir + '/' + header + '_[b, B]1.[t, T][i, I][f, F]')[0]
pr, date         = args.l8_file.split('/')[-1].split('_')[2:4]
path, row        = int(pr[:3]), int(pr[3:])
year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

if len(glob(args.emulator_dir + '/*OLI*.pkl')) < 6:
    down_l8_emus(args.emulator_dir)
cams_file = '%04d-%02d-%02d.nc'%(year, month, day)
if len(glob(os.path.join(args.cams, cams_file))) == 0:
    down_cams(args.cams, cams_file)

dem_dir = '/'.join(args.dem.split('/')[:-1])
url_dem = down_dem(dem_dir, args.l8_file)
if url_dem:
    args.dem = url_dem 
down_l8_modis(args.MCD43_file_dir, args.l8_file)
aero = solve_aerosol(year, month, day, l8_tile = (int(path), int(row)), emus_dir = args.emulator_dir, mcd43_dir   = args.MCD43_file_dir, l8_toa_dir = l8_toa_dir, global_dem=args.dem, cams_dir=args.cams)
aero.solving_l8_aerosol()
atmo_cor = atmospheric_correction(year, month, day, (int(path), int(row)), l8_toa_dir = l8_toa_dir, emus_dir = args.emulator_dir, global_dem=args.dem)
atmo_cor.atmospheric_correction()
