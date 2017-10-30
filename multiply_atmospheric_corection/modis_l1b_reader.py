#!/usr/bin/env python
"""A class to operate with the MODIS L1b gridded product.
This class only takes into account my reprojected granules, and returns **ONLY** the usual 7 optical bands @500m and the relevant 
angles @1km. Extra bands (including thermal, for cloud screening)
aren't provided here.
"""
import datetime
import glob
import os
import re
from collections import namedtuple

import gdal

MODIS_granule = namedtuple("MODIS_granule", "sza saa vza vaa " +
                           "b1 b2 b3 b4 b5 b6 b7 scale offset")

class MODIS_L1b_reader(object):
    def __init__ (self, folder, tile, year):
        """Instantiate with the folder were all the granules are.
        You should also provide a year to filter out the granules.
        Once this is done, you get a class witha  dictionary called
        `granuels`, indexed by date and with the relevant filenames"""
        if os.path.exists(folder):
            self.folder = folder
        else:
            raise IOError("Folder %s doesn't exist" % folder)
        if year > 2000:
            self.year = year
        else:
            raise ValueError("MODIS starts from 2000")
        if re.match ("h\d\dv\d\d", tile):
            self.tile = tile
        else:
            raise ValueError("tile has to be hxxvxx (%s)" % tile)
        self._find_granules()

    def _find_granules(self):
        files = glob.glob(os.path.join(self.folder, 
            "MODIS_REFL.%s.A%d*_EV_250_Aggr500_RefSB_b0.tif" %
            (self.tile, self.year)))
        if len(files) == 0:
            raise IOError("No MODIS files in %s" % self.folder)
        self.granules = {}
        for fich in files:
            fname = os.path.basename(fich)
            date = ".".join(fname.split(".")[2:4])
            date = datetime.datetime.strptime(date, "A%Y%j.%H%M")
            gg = []
            scales = []
            offsets = []
            for angle in ["SolarZenith", "SolarAzimuth", "SensorZenith","SensorAzimuth"]:
                this_fname = fname.replace("MODIS_REFL", "MODTHERM")
                this_fname = this_fname.replace(
                        "EV_250_Aggr500_RefSB_b0.tif",
                        "%s.tif" % angle)
                gg.append(os.path.join(self.folder, this_fname))
            for band in xrange(2):
                this_fname = fname.replace(
                        "EV_250_Aggr500_RefSB_b0.tif",
                        "EV_250_Aggr500_RefSB_b%d.tif" % (band))
                g = gdal.Open(os.path.join(self.folder, this_fname))
                md = g.GetMetadata()
                scales.append(float(md['reflectance_scales'].split(",")[band]))
                offsets.append(float(md['reflectance_offsets'].split(",")[band]))
                gg.append(os.path.join(self.folder, this_fname))
            for band in xrange(5):
                this_fname = fname.replace(
                        "EV_250_Aggr500_RefSB_b0.tif",
                        "EV_500_RefSB_b%d.tif" % (band))
                g = gdal.Open(os.path.join(self.folder, this_fname))
                md = g.GetMetadata()
                scales.append(float(md['reflectance_scales'].split(",")[band]))
                offsets.append(float(md['reflectance_offsets'].split(",")[band]))
                gg.append(os.path.join(self.folder, this_fname))
            gg.append(scales)
            gg.append(offsets)
            self.granules[date] = MODIS_granule(*gg)
            

if __name__ == "__main__":
    modis_l1b = MODIS_L1b_reader( 
            "/data/selene/ucfajlg/Ujia/MODIS_L1b/GRIDDED/", "h17v05",
            2017)
        
    
