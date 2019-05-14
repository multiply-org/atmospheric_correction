#!/usr/bin/env python
import os
import gc
import sys
import gdal
import numpy as np
from glob import glob
import multiprocessing
from SIAC.reproject import reproject_data
from SIAC.s2_angle import resample_s2_angles
from skimage.morphology import disk, binary_dilation, binary_erosion

try:
    import cPickle as pkl
except:
    import pickle as pkl
file_path = os.path.dirname(os.path.realpath(__file__))
gc.disable()
cl = pkl.load(open(file_path + '/data/sen2cloud_detector.pkl', 'rb'))
gc.enable()
cl.n_jobs = multiprocessing.cpu_count()


def do_cloud(cloud_bands, cloud_name = None):
    toas = [reproject_data(str(band), cloud_bands[0], dstNodata=0, resample=5).data for band in cloud_bands]
    toas = np.array(toas)/10000.
    mask = np.all(toas >= 0.0001, axis=0)
    cloud_proba = cl.predict_proba(toas[:, mask].T)
    cloud_mask = np.zeros_like(toas[0])
    cloud_mask[mask] = cloud_proba[:,1]
    if cloud_name is None:
        return cloud_mask
    else:
        g =  gdal.Open(cloud_bands[0])
        dst = gdal.GetDriverByName('GTiff').Create(cloud_name, g.RasterXSize, g.RasterYSize, 1, gdal.GDT_Byte,  options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst.SetGeoTransform(g.GetGeoTransform())
        dst.SetProjection  (g.GetProjection())
        dst.GetRasterBand(1).WriteArray((cloud_mask * 100).astype(int))
        dst=None; g=None 
        return cloud_mask


def s2_pre_processing(s2_dir):
    s2_dir = os.path.abspath(s2_dir)
    scihub = []
    aws    = []
    for (dirpath, dirnames, filenames)  in os.walk(s2_dir):
        if len(filenames)>0:
            temp = [dirpath + '/' + i for i in filenames]
            for j in temp:
                if ('MTD' in j) & ('TL' in j) & ('xml' in j):
                    scihub.append(j)
                if 'metadata.xml' in j:
                    aws.append(j)
    s2_tiles = []
    for metafile in scihub + aws:
        sun_ang_name, view_ang_names, toa_refs, cloud_name = resample_s2_angles(metafile)
        cloud_bands = np.array(toa_refs)[[0,1,3,4,7,8,9,10,11,12]]
        cloud_mask = do_cloud(cloud_bands, cloud_name) > 0.9
        cloud_mask = binary_dilation(binary_erosion (cloud_mask, disk(2)), disk(3))
        s2_tiles.append([sun_ang_name, view_ang_names, toa_refs, cloud_name, cloud_mask, metafile])
    return s2_tiles
