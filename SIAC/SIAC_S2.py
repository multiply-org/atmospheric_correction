import os
import argparse
import requests
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from glob import glob
from SIAC.get_MCD43 import get_mcd43, get_local_MCD43
from datetime import datetime
from SIAC.the_aerosol import solve_aerosol
from SIAC.create_logger import create_logger, create_component_progress_logger
from SIAC.the_correction import atmospheric_correction
from SIAC.s2_preprocessing import s2_pre_processing
from SIAC.downloaders import downloader
from SIAC.multi_process import parmap
from os.path import expanduser
home = expanduser("~")
file_path = os.path.dirname(os.path.realpath(__file__))

logger = create_logger()


def _ensure_dir_format(dir: str):
    if not dir.endswith('/'):
        dir += '/'
    return dir


def SIAC_S2(s2_t, dem_vrt, cams_dir, send_back = False, mcd43 = home + '/MCD43/', emu_dir = file_path + '/emus/',
            vrt_dir = home + '/MCD43_VRT/', download_mcd43 = 'True', aoi = None):
    logger.info('Starting SIAC')
    cams_dir = _ensure_dir_format(cams_dir)
    mcd43 = _ensure_dir_format(mcd43)
    emu_dir = _ensure_dir_format(emu_dir)
    vrt_dir = _ensure_dir_format(vrt_dir)
    if not os.path.exists(emu_dir):
        emu_dir = file_path + '/emus/'
        os.mkdir(emu_dir)
    if len(glob(emu_dir + '/' + 'isotropic_MSI_emulators_*_x?p_S2?.pkl')) < 12:
        url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
        req = requests.get(url)
        to_down = []
        for line in req.text.split():
            if 'MSI' in line:
                fname   = line.split('"')[1].split('<')[0]
                if 'MSI' in fname:
                    to_down.append([fname, url])
        f = lambda fname_url: downloader(fname_url[0], fname_url[1], emu_dir)
        parmap(f, to_down)
    download_mcd43 = download_mcd43 == 'True'
    rets = s2_pre_processing(s2_t)
    aero_atmos = []
    component_progress_logger = create_component_progress_logger()
    for i, ret in enumerate(rets):
        lower_bound = 20 + int((i / len(rets)) * 80)
        upper_bound = 20 + int(((i + 1)/ len(rets)) * 80)
        component_progress_logger.info(f'{20 + int((i / len(rets)) * 80)}')
        ret += (dem_vrt, cams_dir, emu_dir, mcd43, vrt_dir, download_mcd43, aoi)
        #sun_ang_name, view_ang_names, toa_refs, cloud_name, cloud_mask, metafile = ret
        aero_atmo = do_correction(*ret, lower_bound, upper_bound)
        if send_back:
            aero_atmos.append(aero_atmo)
    if send_back:
        return aero_atmos

def do_correction(sun_ang_name, view_ang_names, toa_refs, cloud_name, cloud_mask, metafile, dem_vrt, cams_dir,
                  emus_dir, mcd43 = home + '/MCD43/', vrt_dir = home + '/MCD43_VRT/', download_mcd_43 = True, aoi=None,
                  lower_bound=0, upper_bound=100):

    if os.path.realpath(mcd43) in os.path.realpath(home + '/MCD43/'):
        if not os.path.exists(home + '/MCD43/'):
            os.mkdir(home + '/MCD43/')

    if os.path.realpath(vrt_dir) in os.path.realpath(home + '/MCD43_VRT/'):
        if not os.path.exists(home + '/MCD43_VRT/'):
            os.mkdir(home + '/MCD43_VRT/')

    base = toa_refs[0].replace('B01.jp2', '')
    with open(metafile) as f:
        for i in f.readlines():
            if 'SENSING_TIME' in i:
                sensing_time = i.split('</')[0].split('>')[-1]
                obs_time = datetime.strptime(sensing_time, u'%Y-%m-%dT%H:%M:%S.%fZ')
            if 'TILE_ID' in i:
                sat = i.split('</')[0].split('>')[-1].split('_')[0]
                tile = i.split('</')[0].split('>')[-1]
    log_file = os.path.dirname(metafile) + '/SIAC_S2.log'
    logger = create_logger(log_file)
    logger.info('Starting atmospheric corretion for %s' % tile)
    if not np.all(cloud_mask):
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        if download_mcd_43:
            get_mcd43(toa_refs[0], obs_time, mcd43_dir=mcd43, vrt_dir=vrt_dir, log_file=log_file)
        else:
            get_local_MCD43(toa_refs[0], obs_time, mcd43_dir=mcd43, vrt_dir=vrt_dir)
                    # logger = create_logger(log_file)
    else:
        logger.info('No clean pixel in this scene and no MCD43 is downloaded.')
    sensor_sat = 'MSI', sat
    band_index  = [1,2,3,7,11,12]
    band_wv    = [469, 555, 645, 859, 1640, 2130]
    toa_bands   = (np.array(toa_refs)[band_index,]).tolist()
    view_angles = (np.array(view_ang_names)[band_index,]).tolist()
    sun_angles  = sun_ang_name
    aero = solve_aerosol(sensor_sat,toa_bands,band_wv, band_index,view_angles,\
                         sun_angles,obs_time,cloud_mask, gamma=10., spec_m_dir= \
                         file_path+'/spectral_mapping/', emus_dir=emus_dir, mcd43_dir=vrt_dir, aoi=aoi,
                         global_dem=dem_vrt, cams_dir=cams_dir, log_file = log_file)
    aero._solving(lower_bound, upper_bound)
    toa_bands  = toa_refs
    view_angles = view_ang_names
    aot = base + 'aot.tif'
    tcwv = base + 'tcwv.tif'
    tco3 = base + 'tco3.tif'
    aot_unc = base + 'aot_unc.tif'
    tcwv_unc = base + 'tcwv_unc.tif'
    tco3_unc = base + 'tco3_unc.tif'
    rgb = [toa_bands[3], toa_bands[2], toa_bands[1]]
    band_index = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    atmo = atmospheric_correction(sensor_sat,toa_bands, band_index,view_angles,\
                                  sun_angles, aot = aot, cloud_mask = cloud_mask,\
                                  tcwv = tcwv, tco3 = tco3, aot_unc = aot_unc, \
                                  tcwv_unc = tcwv_unc, tco3_unc = tco3_unc, rgb = \
                                  rgb, emus_dir=emus_dir, global_dem=dem_vrt, cams_dir=cams_dir, log_file = log_file)
    atmo._doing_correction()
    return aero, atmo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentinel 2 Atmospheric Correction Excutable')
    parser.add_argument('-f', "--file_path",      help='Sentinel 2 file path in the form of AWS', required=True)
    parser.add_argument("-m", "--MCD43_file_dir", help="Directory where you store MCD43A1.006 data")
    parser.add_argument("-e", "--emulator_dir",   help="Directory where you store emulators.")
    parser.add_argument("-d", "--dem",            help="A global dem file, and a vrt file is recommended.")
    parser.add_argument("-o", "--download",       help="Whether to download MCD 43 Data.")
    parser.add_argument("-c", "--cams",           help="Directory where you store cams data.")
    parser.add_argument("-a", "--aoi",            help="Area of Interest.")
    args = parser.parse_args()
    SIAC_S2(s2_t=args.file_path, dem_vrt=args.dem, cams_dir=args.cams, mcd43=args.MCD43_file_dir,
        vrt_dir=args.MCD43_file_dir, download_mcd43=args.download, emu_dir=args.emulator_dir, aoi=args.aoi)
