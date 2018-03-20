#!/usr/bin/env python 
import os
import sys
import ogr
import gdal
import psutil
import errno 
import numpy as np
from multiprocessing import Pool
import xml.etree.ElementTree as ET
from functools import partial
s2_file_dir = sys.argv[1]
#s2_file_dir = '/store/S2_data/50/S/LH/2017/2/27/0/'
def parse_xml(s2_file_dir):
    tree = ET.parse(s2_file_dir+'/metadata.xml')
    root = tree.getroot()
    #Sun_Angles_Grid
    saa =[]
    sza =[]
    msz = []
    msa = []
    #Viewing_Incidence_Angles_Grids
    vza = {}
    vaa = {}
    mvz = {}
    mva = {}
    for child in root:
        for j in child:
            for k in j.findall('Sun_Angles_Grid'):
                for l in k.findall('Zenith'):
                    for m in l.findall('Values_List'):
                        for x in m.findall('VALUES'):
                            sza.append(x.text.split())
     
                for n in k.findall('Azimuth'):
                    for o in n.findall('Values_List'):
                        for p in o.findall('VALUES'):
                            saa.append(p.text.split())
            for ms in j.findall('Mean_Sun_Angle'):
                msz = float(ms.find('ZENITH_ANGLE').text)
                msa = float(ms.find('AZIMUTH_ANGLE').text)
     
            for k in j.findall('Viewing_Incidence_Angles_Grids'):
                for l in k.findall('Zenith'):
                    for m in l.findall('Values_List'):
                        vza_sub = []
                        for x in m.findall('VALUES'):
                            vza_sub.append(x.text.split())
                        bi, di, angles = k.attrib['bandId'], \
                                         k.attrib['detectorId'], np.array(vza_sub).astype(float)
                        vza[(int(bi),int(di))] = angles
     
                for n in k.findall('Azimuth'):
                    for o in n.findall('Values_List'):
                        vaa_sub = []
                        for p in o.findall('VALUES'):
                            vaa_sub.append(p.text.split())
                        bi, di, angles = k.attrib['bandId'],\
                                         k.attrib['detectorId'], np.array(vaa_sub).astype(float)
                        vaa[(int(bi),int(di))] = angles
     
            for mvia in j.findall('Mean_Viewing_Incidence_Angle_List'):
                for i in mvia.findall('Mean_Viewing_Incidence_Angle'):
                    mvz[int(i.attrib['bandId'])] = float(i.find('ZENITH_ANGLE').text)
                    mva[int(i.attrib['bandId'])] = float(i.find('AZIMUTH_ANGLE').text)

    sza  = np.array(sza).astype(float)
    saa  = np.array(saa).astype(float)
    saa[saa>180] = saa[saa>180] - 360                 
    g                = gdal.Open(s2_file_dir + '/B04.jp2')
    geo              = g.GetGeoTransform()
    projection       = g.GetProjection()
    geotransform     = (geo[0], 5000, geo[2], geo[3], geo[4], -5000)
    directory = s2_file_dir  + '/angles/'
    try:        
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    outputFileName   = s2_file_dir + '/angles/SAA_SZA.tif'
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, 23, 23, 2, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
    dst_ds.SetGeoTransform(geotransform)   
    dst_ds.SetProjection(projection) 
    dst_ds.GetRasterBand(1).WriteArray((saa * 100).astype(int))
    dst_ds.GetRasterBand(2).WriteArray((sza * 100).astype(int))
    dst_ds, g = None, None
    return vaa, vza

def get_angle(band, vaa, vza, band_dict):
    vas = np.zeros((10980, 10980))
    vas[:] = -327.67
    vzs = np.zeros((10980, 10980))
    vzs[:] = -327.67
    gml = s2_file_dir+'/qi/MSK_DETFOO_%s.gml'%band
    g = ogr.Open(gml)
    xRes = 10; yRes=10
    g1     = gdal.Open(s2_file_dir+'/B04.jp2')
    geo_t = g1.GetGeoTransform()
    x_size, y_size = g1.RasterXSize, g1.RasterYSize
    x_min, x_max  = min(geo_t[0], geo_t[0] + x_size * geo_t[1]), \
      max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    y_min, y_max  = min(geo_t[3], geo_t[3] + y_size * geo_t[5]), \
      max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes  = abs(geo_t[1]), abs(geo_t[5])
    layer = g.GetLayer()
    foot1 = None                            
    foot2 = None                                   
    va1   = None
    vz1   = None         
    va2   = None                                                     
    vz2   = None     
    for i in range(layer.GetFeatureCount()):
        det = layer.GetFeature(i).items()['gml_id']
        foot1 = gdal.Rasterize("", gml, format="MEM", xRes=xRes, yRes=yRes, where="gml_id='%s'"%det, outputBounds=[ x_min, y_min, x_max, y_max], noData=np.nan, burnValues=1).ReadAsArray()
        key =  band_dict[det.split('-')[-3]], int(det.split('-')[-2])
        va1 = vaa[key]       
        vz1 = vza[key]                                                
        if i>0:
            overlap = foot1 * foot2
            x,y = np.where(overlap)
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(),y.max()
            ll = x[x==xmax][-1], y[x==xmax][-1]
            lr = x[y==ymax][-1], y[y==ymax][-1]
            ul = x[y==ymin][0], y[y==ymin][0]
            ur = x[x==xmin][0], y[x==xmin][0]
            p1 = np.mean([lr, ur], axis=0)
            p2 = np.mean([ll, ul], axis=0)
            x1,y1 = np.where(foot2)
            vamax, vamin  = np.nanmax(va2), np.nanmin(va2)
            vzmax, vzmin  = np.nanmax(vz2), np.nanmin(vz2)
            if not (p1==p2).all():
                p = np.poly1d(np.polyfit([p1[0], p2[0]],[p1[1], p2[1]],1))
                foot2[x[y > p(x)], y[y > p(x)]] = False
                if np.where(va2 == vamin)[1][0] <= np.where(va2 == vamax)[1][0]:
                    tmp1 = vamin.copy()
                    vamin = vamax 
                    vamax = tmp1
                if np.where(vz2 == vzmin)[1][0] <= np.where(vz2 == vzmax)[1][0]:
                    tmp2 = vzmin.copy()
                    vzmin = vzmax
                    vzmax = tmp2 
                dist = abs(p(x1)-y1)/(np.sqrt(1+p.c[0]**2))
                vas[x1,y1] = vamin + dist/(dist.max()-dist.min()) * (vamax-vamin)
                vzs[x1,y1] = vzmin + dist/(dist.max()-dist.min()) * (vzmax-vzmin)
            else:
                vas[x1,y1] = vamin
                vzs[x1,y1] = vzmin
            x1,y1 = np.where(foot1)
            if i == layer.GetFeatureCount()-1:
                vamax, vamin  = np.nanmax(va1), np.nanmin(va1)
                vzmax, vzmin  = np.nanmax(vz1), np.nanmin(vz1) 
                if not (p1==p2).all():
                    foot1[x[y <= p(x)], y[y <= p(x)]] = False
                    if np.where(va1 == vamin)[1][0] >= np.where(va1 == vamax)[1][0]:
                        tmp1 = vamin.copy() 
                        vamin = vamax
                        vamax = tmp1
                    if np.where(vz1 == vzmin)[1][0] >= np.where(vz1 == vzmax)[1][0]:
                        tmp2 = vzmin.copy()  
                        vzmin = vzmax 
                        vzmax = tmp2
                    dist = abs(p(x1)-y1)/(np.sqrt(1+p.c[0]**2))
                    vas[x1,y1] = vamin + dist/(dist.max()-dist.min()) * (vamax-vamin)
                    vzs[x1,y1] = vzmin + dist/(dist.max()-dist.min()) * (vzmax-vzmin)
                else:
                    vas[x1,y1] = vamin 
                    vas[x1,y1] = vamin 
        foot2 = foot1
        va2   = va1
        vz2   = vz1
    outputFileName   = s2_file_dir + '/angles/VAA_VZA_%s.tif'%band         
    if os.path.exists(outputFileName):                   
        os.remove(outputFileName)                        
    dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, 10980, 10980, 2, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
    dst_ds.SetGeoTransform(g1.GetGeoTransform())         
    dst_ds.SetProjection(g1.GetProjection())             
    mask      = vas < -180                      
    vas[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), vas[~mask]).astype(float)
    mask      = vzs < 0                              
    vzs[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), vzs[~mask]).astype(float)
    vas[vas>180] = vas[vas>180] - 360                    
    dst_ds.GetRasterBand(1).WriteArray((vas * 100).astype(int))            
    dst_ds.GetRasterBand(2).WriteArray((vzs * 100).astype(int))            
    dst_ds.GetRasterBand(1).SetNoDataValue(-32767)       
    dst_ds.GetRasterBand(2).SetNoDataValue(-32767)       
    dst_ds.FlushCache()                                  
    dst_ds = None  
    g1 = None     

def resample_s2_angles(s2_file_dir):
    #check the available rams and decide cores can be used
    band_ram = 75000000 * 1024. / 13.
    av_ram = psutil.virtual_memory().available 
    procs = np.minimum(int(av_ram / band_ram), psutil.cpu_count())
    #start multiprocessing
    vaa, vza = parse_xml(s2_file_dir)
    bands    = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12' #all bands
    band_dict = dict(zip(bands, range(13)))
    par = partial(get_angle, vaa=vaa, vza=vza, band_dict=band_dict)
    p = Pool(procs)
    ret = p.map(par, bands)
    p.close()
    p.join()
resample_s2_angles(s2_file_dir)
