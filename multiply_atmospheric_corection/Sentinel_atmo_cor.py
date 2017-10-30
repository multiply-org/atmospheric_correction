#/usr/bin/env python
import sys
from s2_aerosol import solve_aerosol
from s2_correction import atmospheric_correction
file_path = sys.argv[1]
day     = int(file_path.split('/')[-3])
month   = int(file_path.split('/')[-4])
year    = int(file_path.split('/')[-5])
s2_tile = ''.join(file_path.split('/')[-8:-5])
aero = solve_aerosol(year, month, day, mcd43_dir = '/home/ucfafyi/DATA/S2_MODIS/m_data/', s2_tile=s2_tile, s2_psf=None)
aero.solving_s2_aerosol()
atm = atmospheric_correction(year, month, day, s2_tile)
atm.atmospheric_correction()  
