#!/usr/bin/env python
"""
An emulation engine for KaFKA. This emulation engine is designed to be useful 
for the atmospheric correction part.
"""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.


import os
import glob
import sys
import cPickle
import logging as log
import numpy as np

import gp_emulator # unnecessary?


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (13.07.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


def create_inverse_emulators ( original_emulator, band_pass, sel_par ):
    """
    This function takes a multivariable output trained emulator
    and "retrains it" to take input reflectances and report a
    prediction of single input parameters (i.e., regression from
    reflectance/radiance to state). This is a useful starting 
    point for spatial problems. Maybe.
    
    Parameters
    ------------
    original_emulator: emulator
        An emulator (type gp_emulator.GaussianProcess)
    band_pass: array
        A 2d bandpass array (nbands, nfreq). Logical type
    sel_par: list
        A list indicating the positions of the parameters that will 
        be used in the inverse emulator.
    """
    
    # For simplicity, let's get the training data out of the emulator
    X = original_emulator.X_train*1.
    y = original_emulator.y_train*1.
    # Apply band pass functions here...
    n_bands = band_pass.shape[0]
    xx = np.array( [ X[:, band_pass[i,:]].sum(axis=1)/ \
        (1.*band_pass[i,:].sum()) \
        for i in xrange( n_bands ) ] )

    # A container to store the emulators
    gps = []
    for  param in sel_par:
        gp = GaussianProcess ( xx.T, y[:, i] )
        gp.learn_hyperparameters( n_tries = 3 )
        gps.append(gp)
    return gps
    
def perband_emulators ( emulators, band_pass ):
    """This function creates per band emulators from the full-spectrum
    emulator. Should be faster in many cases"""
    
    n_bands = band_pass.shape[0]
    x_train_pband = [ emulators.X_train[:,band_pass[i,:]].mean(axis=1) \
        for i in xrange( n_bands ) ]
    x_train_pband = np.array ( x_train_pband )
    emus = []
    for i in xrange( n_bands ):
        gp = GaussianProcess ( emulators.y_train[:]*1, \
                x_train_pband[i,:] )
        gp.learn_hyperparameters ( n_tries=5 )
        emus.append ( gp )
    return emus


def extract_angles(angles):
    """A method that copes with different ways the user has to
    provide angles. This method should be able to cope with either
    (i) a single scalar angle, (ii) a 2D array of angles and (iii)
    either a scalar or 2D array of angles per band."""
    
    if type(angles) == list:
        temp = [np.asarray(x).reshape(1, -1)[0,:] for x in angles]
        angles = temp
    else:
        angles = np.asarray(angles).reshape(1, -1)[0,:]
        
    return angles


class AtmosphericEmulationEngine(object):
    """An emulation engine for single band atmospheric RT models.
    For reference, the ordering of the emulator parameters is
    1. cos(sza)
    2. cos(vza)
    3. saa (in degrees)
    4. vaa (in degrees)
    5. AOT@550
    6. Water Vapour (in 6s units, cm)
    7. Ozone concentration (in 6S units, cm-atm)
    8. Altitude (in km)
    """
    def __init__ ( self, sensor, emulator_folder):
        self.sensor = sensor
        self._locate_emulators(sensor, emulator_folder)

    def _locate_emulators(self, sensor, emulator_folder):
        self.emulators = []
        self.emulator_names = []
        files = glob.glob(os.path.join(emulator_folder, 
                "*%s*.pkl" % sensor))
        files.sort()
        try:
            for fich in files:
                emulator_file = os.path.basename(fich)
                # Create an emulator label (e.g. band name)
                self.emulator_names = emulator_file
                log.info("Found file %s, storing as %s" %
                            (fich, emulator_file))
            from multiprocessing import Pool
            p = Pool(len(files))
            f = lambda fich: cPickle.load(open(fich, 'r'))
            self.emulators = p.map(f, files)
        except:
	    for fich in files:
		emulator_file = os.path.basename(fich)
		# Create an emulator label (e.g. band name)
		self.emulator_names = emulator_file
		self.emulators.append ( cPickle.load(open(fich, 'r')))
		log.info("Found file %s, storing as %s" %
	   		    (fich, emulator_file))
        self.emulators = np.array(self.emulators).ravel()
        self.n_bands = len(self.emulators)


    def emulator_kernel_atmosphere(self, kernel_weights, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_kernels=True, bands=None):
        """This method puts together a 2D array with the parameters
        for the emulator. This method takes kernel weights for
        different bands (iso, vol, geo for self.n_bands),
        atmospheric parameters (AOT, TCWV and O3), as well as some
        "control" variables (view/illumination angles and elevation).
        The method returns the forward modelled TOA reflectances and
        the associated Jacobian. If the option `gradient_kernels` is 
        set to `True`, the Jacobian will also be calculated for the kernels 
        (e.g. in the case of minimising a combined cost function of
        atmosphere and land surface).
        
        We expect `kernel_weights` to be a `3 x self.n_bands x n_pixels`
        array, `atmosphere` to be a `3 x n_pixels` array, and vza, sza,
        vaa, saa and elevation to be `n_pixels` arrays, or if they are
        assumed constaint, they can be scalars.
        
        The `bands` option is there to select individual bands, and it
        should either be a scalar (with the band position in the 
        emulator array), or a list (again with band positions). In this
        case, the kernels can be passed only for the band(s) that are
        requested, but in the same order as the bands. E.g. if 
        `bands=[3,4,5]`, then `kernel_weights` bands should also be
        ordered as band positions 3, 4 and 5 along the second axis.
        """
        
        # the controls can be scalars or arrays
        # We convert them to arrays if needed
        sza = extract_angles(sza)
        saa = extract_angles(saa)
        vza = extract_angles(vza)
        vaa = extract_angles(vaa)
        elevation = np.asarray(elevation).reshape(1, -1)[0,:]
        # the mother of all arrays will be 3*nbands+3+4
        n_pix1 = kernel_weights.shape[2]
        n_pix2 = atmosphere.shape[1]
        assert n_pix1 == n_pix2  # In reality could check angles and stuff
        n_pix = n_pix1 
        x = np.zeros((8 + 3, n_pix)) # 11 parameters innit?
        # Only populate the atmospheric parameters
        # Angles and kernels are defined per band below
        x[7:, :] = np.c_[atmosphere[0,:], atmosphere[1,:], 
                         atmosphere[2,:], elevation*np.ones(n_pix)].T        
        H0 = []
        dH = []
        if bands is None: # Do all bands
            for j, band in enumerate(range(self.n_bands)):
                emu = self.emulators[band]
                if type(sza) == list:
                    x[3] = np.cos(sza[j])*np.ones(n_pix)
                else:
                    x[3] = np.cos(sza)*np.ones(n_pix)
                if type(vza) == list:
                    x[4] = np.cos(vza[j])*np.ones(n_pix)
                else:
                    x[4] = np.cos(vza)*np.ones(n_pix)
                if type(saa) == list:
                    x[5] = saa[j]*np.ones(n_pix)
                else:
                    x[5] = saa*np.ones(n_pix)
                if type(vaa) == list:
                    x[6] = vaa[j]*np.ones(n_pix)
                else:
                    x[6] = vaa*np.ones(n_pix)

                x[0, :] = kernel_weights[0, band, :] # Iso
                x[1, :] = kernel_weights[1, band, :] # Vol
                x[2, :] = kernel_weights[2, band, :] # Geo
                H0_, dH_ = emu.predict(x, do_unc=False)
                if not gradient_kernels:
                    dH_ = dH_[3:, :] # Ignore the kernels in the gradient 
                H0.append(H0_)
                dH.append(dH_)
                
        else:
            # This is needed in case we get a single band
            the_bands = (bands,) if not isinstance(bands, 
                                                   (tuple, list)) else bands
            if max(the_bands) > (self.n_bands-1):
                raise ValueError("There are only " + 
                    "%d bands, and you asked for %d position" 
                    % (self.n_bands, max(the_bands)))
            sel_bands = len(the_bands)
            if kernel_weights.shape[1] == sel_bands:
                # We only got passed a subset of the bands
                is_subset = True
            else:
                is_subset = False
            for j, band in enumerate(the_bands):
                emu = self.emulators[band]
                if is_subset:
                    if type(sza) == list:
                        x[3] = np.cos(sza[j])*np.ones(n_pix)
                    else:
                        x[3] = np.cos(sza)*np.ones(n_pix)
                    if type(vza) == list:
                        x[4] = np.cos(vza[j])*np.ones(n_pix)
                    else:
                        x[4] = np.cos(vza)*np.ones(n_pix)
                    if type(saa) == list:
                        x[5] = saa[j]*np.ones(n_pix)
                    else:
                        x[5] = saa*np.ones(n_pix)
                    if type(vaa) == list:
                        x[6] = vaa[j]*np.ones(n_pix)
                    else:
                        x[6] = vaa*np.ones(n_pix)

                    x[0, :] = kernel_weights[0, j, :] # Iso
                    x[1, :] = kernel_weights[1, j, :] # Vol
                    x[2, :] = kernel_weights[2, j, :] # Geo                    
                else:
                    if type(sza) == list:
                        x[3] = np.cos(sza[j])*np.ones(n_pix)
                    else:
                        x[3] = np.cos(sza)*np.ones(n_pix)
                    if type(vza) == list:
                        x[4] = np.cos(vza[j])*np.ones(n_pix)
                    else:
                        x[4] = np.cos(vza)*np.ones(n_pix)
                    if type(saa) == list:
                        x[5] = saa[j]*np.ones(n_pix)
                    else:
                        x[5] = saa*np.ones(n_pix)
                    if type(vaa) == list:
                        x[6] = vaa[j]*np.ones(n_pix)
                    else:
                        x[6] = vaa*np.ones(n_pix)

                    x[0, :] = kernel_weights[0, band, :] # Iso
                    x[1, :] = kernel_weights[1, band, :] # Vol
                    x[2, :] = kernel_weights[2, band, :] # Geo
                H0_, dH_ = emu.predict(x.T, do_unc=False)
                if not gradient_kernels:
                    dH_ = dH_[3:, :] # Ignore the kernels in the gradient 
                H0.append(H0_)
                dH.append(dH_)
        return H0, dH
            
            
    def emulator_reflectance_atmosphere(self, reflectance, atmosphere, 
                sza, vza, saa, vaa, elevation, 
                gradient_refl=True, bands=None):
        """This method puts together a 2D array with the parameters
        for the emulator. This method takes SDR for
        different bands, atmospheric parameters (AOT, TCWV and O3), 
        as well as some "control" variables (view/illumination angles 
        and elevation).
        
        The method returns the forward modelled TOA reflectances and
        the associated Jacobian. If the option `gradient_refl` is 
        set to `True`, the Jacobian will also be calculated for the kernels 
        (e.g. in the case of minimising a combined cost function of
        atmosphere and land surface).
        
        We expect `reflectance` to be a `self.n_bands x n_pixels`
        array, `atmosphere` to be a `3 x n_pixels` array, and vza, sza,
        vaa, saa and elevation to be `n_pixels` arrays, or if they are
        assumed constaint, they can be scalars.
        
        The `bands` option is there to select individual bands, and it
        should either be a scalar (with the band position in the 
        emulator array), or a list (again with band positions). In this
        case, the reflectance be passed only for the band(s) that are
        requested, but in the same order as the bands. E.g. if 
        `bands=[3,4,5]`, then `reflectance` bands should also be
        ordered as band positions 3, 4 and 5 along the second axis.
        """
        # the controls can be scalars or arrays
        # We convert them to arrays if needed

        sza = extract_angles(sza)
        saa = extract_angles(saa)
        vza = extract_angles(vza)
        vaa = extract_angles(vaa)

        elevation = np.asarray(elevation).reshape(1, -1)[0,:]
        # the mother of all arrays will be 3*nbands+3+4
        n_pix1 = reflectance.shape[1]
        n_pix2 = atmosphere.shape[1]
        assert n_pix1 == n_pix2  # In reality could check angles and stuff
        n_pix = n_pix1 
        x = np.zeros((9, n_pix)) # 10 parameters innit?
        # Set up atmospheric parameters
        # Angles and isotropic reflectance are/can be defined **per band**
        x[4:-1, :] = np.c_[atmosphere[0,:], atmosphere[1,:], 
                         atmosphere[2,:], elevation*np.ones(n_pix)].T
        H0 = []
        dH = []
        if bands is None: # Do all bands
            for j, band in enumerate(range(self.n_bands)):
                emu = self.emulators[band]
                if type(sza) == list:
                    x[0] = np.cos(sza[j])*np.ones(n_pix)
                else:
                    x[0] = np.cos(sza)*np.ones(n_pix)
                if type(vza) == list:
                    x[1] = np.cos(vza[j])*np.ones(n_pix)
                else:
                    x[1] = np.cos(vza)*np.ones(n_pix)
                if type(saa) == list:
                    x[2] = saa[j]*np.ones(n_pix)
                else:
                    x[2] = saa*np.ones(n_pix)
                if type(vaa) == list:
                    x[3] = vaa[j]*np.ones(n_pix)
                else:
                    x[3] = vaa*np.ones(n_pix)

                x[-1, ] = reflectance[band, :]
                H0_, dH_ = emu.predict(x.T, do_unc=False)
                if not gradient_refl:
                    dH_ = dH_[:-1, :] # Ignore the SDR in the gradient 
                H0.append(H0_)
                dH.append(dH_)
                
        else:
            # This is needed in case we get a single band
            the_bands = (bands,) if not isinstance(bands, 
                                                   (tuple, list)) else bands
            if max(the_bands) > (self.n_bands-1):
                raise ValueError("There are only " + 
                    "%d bands, and you asked for %d position" 
                    % (self.n_bands, max(the_bands)))

            sel_bands = len(the_bands)
            if reflectance.shape[0] == sel_bands:
                # We only got passed a subset of the bands
                is_subset = True
            else:
                is_subset = False
            for j, band in enumerate(the_bands):
                emu = self.emulators[band]
                if is_subset:
                    if type(sza) == list:
                        x[0] = np.cos(sza[j])*np.ones(n_pix)
                    else:
                        x[0] = np.cos(sza)*np.ones(n_pix)
                    if type(vza) == list:
                        x[1] = np.cos(vza[j])*np.ones(n_pix)
                    else:
                        x[1] = np.cos(vza)*np.ones(n_pix)
                    if type(saa) == list:
                        x[2] = saa[j]*np.ones(n_pix)
                    else:
                        x[2] = saa*np.ones(n_pix)
                    if type(vaa) == list:
                        x[3] = vaa[j]*np.ones(n_pix)
                    else:
                        x[3] = vaa*np.ones(n_pix)
                    x[-1, :] = reflectance[j, :] 
                else:
                    if type(sza) == list:
                        x[0] = np.cos(sza[j])*np.ones(n_pix)
                    else:
                        x[0] = np.cos(sza)*np.ones(n_pix)
                    if type(vza) == list:
                        x[1] = np.cos(vza[j])*np.ones(n_pix)
                    else:
                        x[1] = np.cos(vza)*np.ones(n_pix)
                    if type(saa) == list:
                        x[2] = saa[j]*np.ones(n_pix)
                    else:
                        x[2] = saa*np.ones(n_pix)
                    if type(vaa) == list:
                        x[3] = vaa[j]*np.ones(n_pix)
                    else:
                        x[3] = vaa*np.ones(n_pix)
                    x[-1, :] = reflectance[band, :]
                    
                H0_, dH_ = emu.predict(x.T, do_unc=False)
                if not gradient_refl:
                    dH_ = dH_[:-1, :] # Ignore the SDR in the gradient 
                H0.append(H0_)
                dH.append(dH_)
        return H0, dH            
            
            
            
class RTEmulationEngine(object):
    """A class to load up single band emulators and use them.
    NOTE that in some cases you might want to override the __init__
    method and just pass the emulators themselves. The `predict`
    method only requires `self.n_bands` and `self.emulators` (a
    list with length `self.n_bands`) to work."""
    def __init__ (self, sensor, model, emu_folder):
        """Locates emulators for a particular model and emulator.
        Currently based on filename patterns, in the future a better
        approach will be to fish the data from a database."""
        
        self._locate_emulators( model, sensor, emulator_folder)

    def _locate_emulators(self, model, sensor, emulator_folder):
        self.emulators = []
        self.emulator_names = []
        files = glob.glob(os.path.join(emulator_folder, 
                "%s*%s*.pkl" % (model, sensor)))
        files.sort()
        if len(files)>1:
            for fich in files:
                emulator_file = os.path.basename(fich)
                # Create an emulator label (e.g. band name)
                self.emulator_names = emulator_file
                log.info("Found file %s, storing as %s" %
                            (fich, emulator_file))
            from multiprocessing import Pool
            p = Pool(len(files))
            f = lambda fich: cPickle.load(open(fich, 'r'))
            self.emulators = p.map(f, files)
        else:
	    for fich in files:
		emulator_file = os.path.basename(fich)
		# Create an emulator label (e.g. band name)
		self.emulator_names = emulator_file
		self.emulators.append ( cPickle.load(open(fich, 'r')))
		log.info("Found file %s, storing as %s" %
			    (fich, emulator_file))
        
        self.n_bands = len(self.emulators)
        
    def predict(self, x):
        """A simple multiband predict method. Returns prediction of
        model values and approximate gradient. Note that `x` input
        has to be in the same order as expected by emulator.
        """
        H0 = []
        dH = []
        # In here, x is the state for all samples
        for band in xrange(self.n_bands):
            H0_, dH_ = self.emulators[band].predict (x, do_unc=False)
            H0.append(H0_)
            dH.append(dH)
        return np.array(H0).squeeze(), np.array(dH).squeeze()
        
        
        
        
class RTEmulationEngineSpectral(object):
    """A standard emulation engine for a spectral RT model."""
    def __init__ (self, emulator_file, srf):
        # Use case #1: an emulator file is given and loaded.
        # Options: 1.a -> Spectral emulator
        #          1.b -> Not spectral (e.g. 2stream)
        if not os.path.exists(emulator_file):
            
            raise IOError("Emulator file %s doesn't exist" %
                              emulator_file)
        self.emulator = self._load_emulator(emulator_file)
        self.set_srf(srf)
        
    def _load_emulator(emulator_file):
        self.emulator = gp_emulator.MultivariateEmulator(dump=emulator_file)
        

    
    def set_srf(self, srf):
        self.n_bands = len(srf)
        self.srf = srf
    
    def predict(self, x):
        
        f, g = self.emulator.predict ( np.atleast_2d( x ) )
        fwd_model_obs = []
        gradient = []
        for i in xrange( self.n_bands ):
            d = f*self.srf[i]/(self.srf[i].sum())
            grad = g*self.srf[i][None, :]/(self.srf[i].sum())
        fwd_model_obs.append ( d)
        gradient.append ( grad)
        return np.array(fwd_model_obs).squeeze(), np.array(gradient).squeeze()
