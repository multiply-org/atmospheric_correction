#/usr/bin/env python
import gdal
import numpy as np
import numpy.ma as ma
import kernels
from multiprocessing import Pool
from functools import partial
from datetime import datetime, timedelta
from glob import glob

def r_modis(fname, slic=None):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        if slic==None:
            return g.ReadAsArray()
        elif g.RasterCount==1: 
            Lx,Ly = slic
            return g.ReadAsArray()[Lx,Ly]
        elif g.RasterCount>1:
            Lx,Ly = slic
            return g.ReadAsArray()[:, Lx, Ly]
        else:
            raise IOError

#bands = [2,3,4,8,13,11,12]


def get_kk(angles):
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk


def qa_to_ReW(modisQAs, bands):
    magic = 0.618034
    modis = r_modis(modisQAs[3][0])
    QA = np.array([np.right_shift(np.bitwise_and(modis, np.left_shift(15,i*4)), i*4) for i in np.arange(0,7)])[bands,]
    relative_W = magic**QA
    relative_W[relative_W<magic**4]=0
    return relative_W

def get_rs(modisQAs, modis_filenames, angles, bands = range(7)):
    
    kk = get_kk(angles)
    k_vol = kk.Ross
    k_geo = kk.Li

    br = np.array([r_modis(modis_filenames[i][0]) for i in bands])
    mask = (br[:,0,:,:] > 32766) | (br[:,1,:,:] > 32766) |(br[:,2,:,:] > 32766)
    rw = qa_to_ReW(modisQAs, bands) # correpsonding relative weights
    brdf = br[:,0,:,:] + (br[:,1,:,:].T*k_vol).T + (br[:,2,:,:].T*k_geo).T
    brdf = ma.array(brdf, mask = mask)

    return [brdf,rw]


def get_brdf_six(fname, angles, bands = (7,), Linds = None, do_unc = True):    
    temp1 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    p  =Pool(len(bands)*2)
    fnames = [temp1%(fname, band) for band in bands] + [temp2%(fname, band) for band in bands]   
    kk = get_kk(angles)
    k_vol = kk.Ross
    k_geo = kk.Li
    par = partial(r_modis, slic=Linds)
    ret = p.map(par, fnames)
    br, qa = np.array(ret[:len(bands)]), np.array(ret[len(bands):])
    if Linds is None:
        brdf = br[:,0,:,:] + br[:,1,:,:]*k_vol + br[:,2,:,:]*k_geo
    else:
        brdf = br[:,0] + br[:,1]*k_vol + br[:,2]*k_geo
    if do_unc:
        doy   = fname.split('.')[-5]
	date  = datetime.strptime(doy, 'A%Y%j')
	day_before = [(date - timedelta(days = i)).strftime('A%Y%j') for i in range(1,4)]
	day_after  = [(date + timedelta(days = i)).strftime('A%Y%j') for i in range(1,4)]
	finder = fname.split('MCD43A1')[0] + 'MCD43A1.%s.' + fname.split('.')[-4] +'.006.*hdf'
	before_f = sorted([glob(finder%i)[0] for i in day_before])
	after_f =  sorted([glob(finder%i)[0] for i in day_after])
        fnames = [temp1%(beforef, band) for beforef in before_f for band in bands] + \
                 [temp1%(afterf, band) for afterf in after_f for band in bands]

        p   = Pool(len(bands)*2)
        par = partial(r_modis, slic=Linds)
        ret = p.map(par, fnames)
        all_br = np.r_[np.array(ret).reshape((6,len(bands),3) + \
                       ret[0].shape[1:]), br.reshape((1,) + br.shape)]
        std = np.std(all_br, axis=0)
        if Linds is None:
            unc = np.sqrt(std[:,0,:,:]**2 + (std[:,1,:,:]**2)*k_vol**2 + (std[:,2,:,:]**2)*k_geo**2)
            
        else:
            unc = np.sqrt(std[:,0]**2 + (std[:,1]**2)*k_vol**2 + (std[:,2]**2)*k_geo**2)
        return [brdf*0.001, qa, unc*0.001]
    else:
        return [brdf*0.001, qa]
        
	'''
	if Linds==None:
	    
	    br = np.array([r_modis(temp1%(fname, band)) for band in bands])
	    qa = np.array([r_modis(temp2%(fname, band)) for band in bands])
	    #mask = (br[:,0,:,:] > 32766) | (br[:,1,:,:] > 32766) |(br[:,2,:,:] > 32766)
	    brdf = br[:,0,:,:] + br[:,1,:,:]*k_vol + br[:,2,:,:]*k_geo
	    #brdf = ma.array(brdf, mask = mask)
	    return [brdf*0.001, qa]
	else:
	    Lx, Ly = Linds
	    br = np.array([r_modis(temp1%(fname, band), slic=[Lx, Ly]) for band in bands])
	    qa = np.array([r_modis(temp2%(fname, band), slic=[Lx, Ly]) for band in bands])
	    brdf = br[:,0] + br[:,1]*k_vol + br[:,2]*k_geo
	    if flag==None:
		return [brdf*0.001, qa]
	    else:
		mask = (qa<=flag)
		#val_brdf = brdf[:,mask]
		#val_ins = np.array(Linds)[:,mask]
		return [brdf*0.001, mask]
	 '''




if __name__=='__main__':
    from multiprocessing import Pool
    from functools import partial
    from glob import glob
    p  =Pool(14)
    fnames = ['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'%('/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016311.h19v10.006.2016320215901.hdf', i) for i in range(1,8)]+\
	      ['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'%('/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016311.h19v10.006.2016320215901.hdf', i) for i in range(1,8)]
    par = partial(r_modis, slic=None)
    #ret = p.map(par, fnames)
    #val1 = get_brdf_six('/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016311.h19v10.006.2016320215901.hdf',angles = [np.array(10),np.array(20),np.array(30)], bands=(1,2,3,4,5,6,7))
    #val2 = get_brdf_six('/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016311.h19v10.006.2016320215901.hdf',angles = [np.array(10),np.array(20),np.array(30)], bands=(1,2,3,4,5,6,7), Linds = [np.arange(2400), np.arange(2400)])

    fname = '/data/selene/ucfajlg/Ujia/MCD43/MCD43A1.A2017247.h17v05.006.2017256031007.hdf'
    doy   = fname.split('.')[-5]
    date  = datetime.strptime(doy, 'A%Y%j')
    day_before = (date - timedelta(days = 1)).strftime('A%Y%j')
    day_after  = (date + timedelta(days = 1)).strftime('A%Y%j')
    finder = fname.split('MCD43A1')[0] + 'MCD43A1.%s.' + fname.split('.')[-4] +'.006.*hdf'
    before_f = glob(finder%day_before)[0]
    after_f = glob(finder%day_after)[0]
    fnames = ['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'%(fname, i) for i in range(1,8)]+ ['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'%(fname, i) for i in range(1,8)]+ ['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'%(before_f,i) for i in range(1,8)] +['HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'%(after_f,i) for i in range(1,8)]
    ret = p.map(par, fnames)




