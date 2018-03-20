# Sentinel 2 and Landsat 8 Atmospheric correction 
### Feng Yin
### Department of Geography, UCL
### ucfafyi@ucl.ac.uk

This atmospheric correction method uses MODIS MCD43 BRDF product to get a coarse resolution simulation of earth surface. A model based on MODIS PSF is built to deal with the scale differences between MODIS and Sentinel 2 / Landsat 8. We uses the ECMWF CAMS prediction as a prior for the atmospheric states, coupling with 6S model to solve for the atmospheric parameters. We do not have a proper cloud mask at the moment and no topography correction as well. Homogeneouse surface is used without considering the BRDF effects.

## Data needed:
* MCD43 (`multiply_atmospheric_corection/MCD43`): 16 days before and 16 days after the Sentinel 2 / Landsat 8 sensing date
* ECMWF CAMS Near Real Time prediction (`multiply_atmospheric_corection/cams`): a time step of 3 hours with the start time of 00:00:00 over the date
* global dem (`multiply_atmospheric_corection/eles`): Global DEM VRT file built from ASTGTM2 DEM, and a bash script under eles/ can be used to generate with the individual files.
* emus (`multiply_atmospheric_corection/emus`): emulators for the 6S and wv restrival, can be found at: http://www2.geog.ucl.ac.uk/~ucfafyi/emus/

## Usage:
* A typical usage for Sentinel 2 is:
`./Sentinel2_AtmoCor.py -f /directory/where/you/store/s2/data/29/S/QB/2017/1/12/0/ [-m MCD43_dir -e emus_dir -d global_DEM -c cams_dir]`
* A typical usage for Landsat 8 is:
`./Landsat8_AtmoCor.py -f /directory/where/you/store/l8/data/LC08_L1TP_029029_20160720_20170222_01_T1 [-m MCD43_dir -e emus_dir -d global_DEM -c cams_dir]`
* Arguments inside [ ] means optional, in the case you store the data in the framework specified above.

## gp_emulator:
If you are using python 3.6, please reinstall gp_emulator on branch `python_3_6_compatible`.

## Output:
### Sentinel 2:
The outputs are the corrected TOA images saved as `B0*_sur.tif` for each band and uncertainty `B0*_sur_unc.tif`. TOA_RGB.tif and BOA_RGB.tif are generated for a fast visual check of correction results. They are all under `/directory/where/you/store/s2/data/29/S/QB/2017/9/4/0/` as the example usage.

### Landsat 8:
The outputs are the corrected TOA images saved as `LC08_L1TP_029029_20160720_20170222_01_T1_sur_b*.tif` for each band and  uncertainty `LC08_L1TP_029029_20160720_20170222_01_T1_sur_b*_unc.tif`, and `LC08_L1TP_029029_20160720_20170222_01_T1_TOA_RGB.tif` and `LC08_L1TP_029029_20160720_20170222_01_T1_BOA_RGB.tif` are also generated for fast visual check. They are all under `/directory/where/you/store/l8/data/LC08_L1TP_029029_20160720_20170222_01_T1`.
