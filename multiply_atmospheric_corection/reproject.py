import gdal
from osgeo import osr 
import numpy as np
import numpy.ma as ma
#gdalwarp -of VRT -t_srs "+proj=sinu" -te xmin ymin xmax ymax -ts 2400 2400 global_dem.vrt modis_cutoff.vrt

class reproject_data(object):
    '''
    A function uses a source and a target images and 
    and clip the source image to match the extend, 
    projection and resolution as the target image.

    '''
    def __init__(self, source_img,
                 target_img   = None,
                 dstSRS       = None,
                 verbose      = False,
                 (xmin, xmax) = (None, None),
                 (ymin, ymax) = (None, None),
                 (xRes, yRes) = (None, None)):

        self.source_img = source_img
        self.target_img = target_img
        self.verbose    = verbose
        self.dstSRS     = dstSRS
        self.xmin       = xmin
        self.xmax       = xmax
        self.ymin       = ymin
        self.ymax       = ymax
        self.xRes       = xRes
        self.yRes       = yRes
    #def get_it(self,):
        if (self.target_img is None) & (self.dstSRS is None):
            raise IOError, 'Projection should be specified ether from a file or a projection code.'
        elif self.target_img is not None:
            g     = gdal.Open(self.target_img)
            geo_t = g.GetGeoTransform()
            x_size, y_size = g.RasterXSize, g.RasterYSize     
            xmin, xmax = min(geo_t[0], geo_t[0] + x_size * geo_t[1]), \
                         max(geo_t[0], geo_t[0] + x_size * geo_t[1])  
            ymin, ymax = min(geo_t[3], geo_t[3] + y_size * geo_t[5]), \
                         max(geo_t[3], geo_t[3] + y_size * geo_t[5])
            xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
            dstSRS     = osr.SpatialReference( )
            raster_wkt = g.GetProjection()
            dstSRS.ImportFromWkt(raster_wkt)
            self.g = gdal.Warp('', self.source_img, format = 'MEM', outputBounds = \
                               [xmin, ymin, xmax, ymax], xRes = xRes, yRes = yRes, dstSRS = dstSRS)
            
        else:
            self.g = gdal.Warp('', self.source_img, format = 'MEM', outputBounds = \
                               [self.xmin, self.ymin, self.xmax, self.ymax], xRes = \
                                self.xRes, yRes = self.yRes, dstSRS = self.dstSRS, copyMetadata=True)
        if self.g.RasterCount <= 3:
            self.data = self.g.ReadAsArray()
            #return self.data
        elif self.verbose:
            print 'There are %d bands in this file, use g.GetRasterBand(<band>) to avoid reading the whole file.'%self.g.RasterCount

if __name__=='__main__':
    ele = reproject_data('/home/ucfafyi/DATA/S2_MODIS/SRTM/global_dem.vrt','/home/ucfafyi/DATA/S2_MODIS/s_data/29/S/QB/2016/12/23/0/B04.jp2') 
    #ele.get_it()
    mask = (ele.data == -32768) | (~np.isfinite(ele.data))
    ele.data = ma.array(ele.data, mask = mask)
