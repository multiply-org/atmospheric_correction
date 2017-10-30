#!/usr/bin/env python
"""A class to operate with the OLCI re-gridded product
"""
import datetime
import glob
import re
import os
from collections import namedtuple

import numpy as np
import gdal

OLCI_granule = namedtuple("OLCI_granule", "mask sza saa vza vaa " +
                        "b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 "+
                        "b13 b14 b15 b16 b17 b18 b19 b20 b21")

class OLCI_L1b_reader(object):

    def __init__ (self, tile, olci_path):
        "A class to read out OLCI data"""
        if os.path.exists (olci_path):
            self.olci_path = olci_path
        else:
            raise IOError("The given olci_path (%s)" + \
                            "does not exist!")
        if re.match ("h\d\dv\d\d", tile):
            self.tile = tile
        else:
            raise ValueError("tile has to be hxxvxx (%s)" % tile)

        self._find_olci_granules()
    
    def _process_fname(self, granule_path):
        granule = os.path.basename(granule_path)
        date = granule.split("____")[1].split("_")[0]
        date = datetime.datetime.strptime(date, 
                                          "%Y%m%dT%H%M%S")
        bands = ["Oa%02d_reflectance.%s.masked.tif" %(band, self.tile)
                 for band in xrange(1, 22)]
        metadata = ["%s.%s.tif" % ( meta, self.tile) 
                    for meta in ["quality_flags", "SZA", 
                                 "SAA", "OZA", "OAA"]]
        ll = []
        for layer in (metadata + bands):
            if not os.path.exists(os.path.join(granule_path,
                                               layer)):
                raise IOError
            ll.append(os.path.join(granule_path,
                                               layer))
        return date, OLCI_granule( *ll)                
                
            
            
            
    
    def _find_olci_granules(self):
        granules = glob.glob (os.path.join(self.olci_path, 
                                           "S3A_OL_1_EFR*.%s" % self.tile))
        if len(granules) == 0:
            raise ValueError("No OLCI granules found")
        
        self.granules = {}
        for granule in granules:
            this_date, this_data = self._process_fname(granule)
            self.granules[this_date] = this_data

            
if __name__ == "__main__":
    OLCI = OLCI_L1b_reader("h17v05", "/storage/ucfajlg/Ujia/S3_test/OLCI")
    
