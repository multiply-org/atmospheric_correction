import os
import numpy as np
import gdal
import glob
from collections import namedtuple

'''
This grabs level 1b modis data from
directory_l1b assuming files are specified in a vrt


reflective_band_[1-7].vrt
MODTHERM_Band[vza,sza,vaa,saa].vrt

for specified doy and year

returns list of reflectance and angle datasets

'''

def resample( src_ds, target_ds, resolution=True, projection=False):

    src_ds = 'HDF4_EOS:EOS_GRID:%s:MOD_Grid_BRDF:BRDF_Albedo_Parameters_shortwave'%src_ds
    if type(src_ds) != gdal.Dataset:
        src_ds = gdal.Open(src_ds)
        if src_ds is None:
            raise ValueError
    src_proj = src_ds.GetProjection()
    src_geoT = src_ds.GetGeoTransform()

    if type(target_ds) != gdal.Dataset:
        target_ds = gdal.Open(target_ds)
        if target_ds is None:
            raise ValueError
        target_proj = target_ds.GetProjection()
        target_geoT = target_ds.GetGeoTransform()

    dst = gdal.GetDriverByName("MEM").Create("", src_ds.RasterXSize,
            src_ds.RasterYSize, target_ds.RasterCount, gdal.GDT_UInt16)
    dst.SetGeoTransform(src_geoT)
    dst.SetProjection(src_proj)

    gdal.ReprojectImage(target_ds, dst, src_proj,
                        src_proj, gdal.GRA_NearestNeighbour)
    data = dst.ReadAsArray()
    return data


def grab_dataset(number_in_list,directory_l1b,mcd43file):
    # TOA reflectances
    bandas = []
    for band in xrange(1,8):
        bandas.append ( gdal.Open(
            os.path.join(directory_l1b, "reflective_band_%d.vrt" % band)))

    # Get TOA reflectances
    refl_scales = np.array([ 5.342521763e-05, 3.336303416e-05,
            3.672689127e-05, 3.438158819e-05, 3.778625614e-05,
            3.471032323e-05, 2.820518966e-05])

    toa_refl = []
    # now read the data file
    for band in xrange(1,8):
        DN = bandas[band-1].GetFileList()[number_in_list]
        DN = resample(mcd43file,DN)
        refl = np.where (DN == 65535, np.nan,DN*refl_scales[band-1])
        toa_refl.append(refl)
    toa_refl = np.array(toa_refl)

    # Angles
    angles = []
    for a in "vza sza vaa saa".split():
        f = "MODTHERM_Band%s.vrt"%a.upper()
        # do this to avoid core dump reading the vrt file
        fname = gdal.Open(os.path.join(directory_l1b,f)).GetFileList()[number_in_list]
        angle = resample(mcd43file,fname)
        angles.append(np.where (angle == -32767, np.nan, angle/100.))
    angles = np.array(angles)
    return toa_refl,angles

def grab_modis_toa (year=2006,doy=200,verbose=True,
        mcd43file = '/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016128.h11v04.006.2016180234038.hdf',
        directory_l1b="/data/selene/ucfajlg/Bondville_MODIS/THERMAL"):

    TILE = mcd43file.split('.')[-4]
    directory_l1b = directory_l1b + '/%d'%year

    egfile = gdal.Open(os.path.join(directory_l1b, "reflective_band_%d.vrt" % 1))
    all_files = egfile.GetFileList()

    toa_refl = []
    angles   = [] 
    for i in xrange(len(all_files)):
        # look for doy
        try:
            code = all_files[i].split('MODREFL.A')[1].split('.')[0]
            this_year = int(code[:4])
            this_doy  = int(code[4:])
            if (this_year == year) and (this_doy == doy):
                r,a = grab_dataset(i,directory_l1b,mcd43file)
                toa_refl.append(r)
                angles.append(a)
        except:
            pass

    return toa_refl,angles

if __name__ == "__main__":

    r,a = grab_modis_toa()

  

