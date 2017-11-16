from math import sqrt, cos, sin, tan, pi, asin, acos, atan, atan2
import math
import sys
import re
import os
import struct
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, '/opt/anaconda/lib/python2.7/site-packages/')
from osgeo import gdal
import errno

if len(sys.argv) < 2:
	print 'usage:  python ', sys.argv[0], ' <XML Tile Metadata File> [Subsample Factor]'
	sys.exit()


############################################################################
# Sudipta's addition to enable spatial subset
############################################################################
K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = math.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

def from_latlon(latitude, longitude, force_zone_number=None):
    if not -80.0 <= latitude <= 84.0:
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not -180.0 <= longitude <= 180.0:
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')

    lat_rad = math.radians(latitude)
    lat_sin = math.sin(lat_rad)
    lat_cos = math.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(latitude)

    lon_rad = math.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = math.radians(central_lon)

    n = R / math.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * math.sin(2 * lat_rad) +
             M3 * math.sin(4 * lat_rad) -
             M4 * math.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if latitude < 0:
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(latitude):
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3

############################################################################
# End Sudipta's addition
############################################################################

# Define constants
a = 6378137.0                   # WGS 84 semi-major axis in meters
b = 6356752.314                 # WGS 84 semi-minor axis in meters
ecc = 1.0 - b / a * b / a       # WGS 84 ellipsoid eccentricity
todeg = 180.0 / pi		# Converts radians to degrees

# Define functions used to construct image observations
def LOSVec( Lat, Lon, Zen, Az ):
	LSRx = ( -sin(Lon), cos(Lon), 0.0 )
	LSRy = ( -sin(Lat)*cos(Lon), -sin(Lat)*sin(Lon), cos(Lat) )
	LSRz = ( cos(Lat)*cos(Lon), cos(Lat)*sin(Lon), sin(Lat) )
	LOS = ( sin(Zen)*sin(Az), sin(Zen)*cos(Az), cos(Zen) )
	Sat = ( LOS[0]*LSRx[0] + LOS[1]*LSRy[0] + LOS[2]*LSRz[0],
	        LOS[0]*LSRx[1] + LOS[1]*LSRy[1] + LOS[2]*LSRz[1],
	        LOS[0]*LSRx[2] + LOS[1]*LSRy[2] + LOS[2]*LSRz[2] )
        Rn = a / sqrt( 1.0 - ecc *sin(Lat)*sin(Lat))
        Gx = ( Rn*cos(Lat)*cos(Lon),
               Rn*cos(Lat)*sin(Lon),
               Rn*(1-ecc)*sin(Lat) )
	return ( Sat, Gx )

def GrndVec( Lat, Lon ):
        Rn = a / sqrt( 1.0 - ecc *sin(Lat)*sin(Lat))
        Gx = ( Rn*cos(Lat)*cos(Lon),
               Rn*cos(Lat)*sin(Lon),
               Rn*(1-ecc)*sin(Lat) )
	return ( Gx )

# Inverse (X/Y to lat/long) UTM projection
def utm_inv( Zone, X, Y, a=6378137.0, b=6356752.31414 ):
        if Zone < 0 :
                FNorth = 10000000.0     # Southern hemisphere False Northing
        else:
                FNorth = 0.0	        # Northern hemisphere False Northing
        FEast = 500000.0                # UTM False Easting
        Scale = 0.9996                  # Scale at CM (UTM parameter)
        LatOrigin = 0.0                 # Latitude origin (UTM parameter)
        CMDeg = -177 + (abs(int(Zone))-1)*6
        CM = float(CMDeg)*pi/180.0      # Central meridian (based on zone)
        ecc = 1.0 - b/a*b/a
        ep = ecc/(1.0-ecc)
        M0 = a*((1.0-ecc*(0.25+ecc*(3.0/64.0+ecc*5.0/256.0)))*LatOrigin
               -ecc*(0.375+ecc*(3.0/32.0+ecc*45.0/1024.0))*sin(2.0*LatOrigin)
               +ecc*ecc*(15.0/256.0+ecc*45.0/1024.0)*sin(4.0*LatOrigin)
               -ecc*ecc*ecc*35.0/3072.0*sin(6.0*LatOrigin))
        M = M0+(Y-FNorth)/Scale
        Mu = M/(a*(1.0-ecc*(0.25+ecc*(3.0/64.0+ecc*5.0/256.0))))
        e1 = (1.0-sqrt(1-ecc))/(1.0+sqrt(1.0-ecc))
        Phi1 = Mu+(e1*(1.5-27.0/32.0*e1*e1)*sin(2.0*Mu)
                   +e1*e1*(21.0/16.0-55.0/32.0*e1*e1)*sin(4.0*Mu)
                   +151.0/96.0*e1*e1*e1*sin(6.0*Mu)
                   +1097.0/512.0*e1*e1*e1*e1*sin(8.0*Mu))  
        slat = sin(Phi1)
        clat = cos(Phi1)
        Rn1 = a/sqrt(1.0-ecc*slat*slat)
        T1 = slat*slat/clat/clat
        C1 = ep*clat*clat
        R1 = Rn1*(1.0-ecc)/(1.0-ecc*slat*slat)
        D = (X-FEast)/Rn1/Scale
        # Calculate Lat/Lon
        Lat = Phi1 - (Rn1*slat/clat/R1*(D*D/2.0
                        -(5.0+3.0*T1+10.0*C1-4.0*C1*C1-9.0*ep)*D*D*D*D/24.0
                        +(61.0+90.0*T1+298.0*C1+45.0*T1*T1-252.0*ep-3.0*C1*C1)*D*D*D*D*D*D/720.0))
        Lon = CM + (D-(1.0+2.0*T1+C1)*D*D*D/6.0+(5.0-2.0*C1+28.0*T1-3.0*C1*C1+8.0*ep+24.0*T1*T1)
                    *D*D*D*D*D/120.0)/clat
        
        return (Lat, Lon)

def get_angleobs( XML_File ):

	# Parse the XML file 
	tree = ET.parse(XML_File)
	root = tree.getroot()

	# Find the angles
	for child in root:
		if child.tag[-12:] == 'General_Info':
			geninfo = child
		if child.tag[-14:] == 'Geometric_Info':
			geoinfo = child

	for segment in geninfo:
		if segment.tag == 'TILE_ID':
			tile_id = segment.text.strip()

	for segment in geoinfo:
        	if segment.tag == 'Tile_Geocoding':
			frame = segment
		if segment.tag == 'Tile_Angles':
			angles = segment

	for box in frame:
		if box.tag == 'HORIZONTAL_CS_NAME':
			czone = box.text.strip()[-3:]
			hemis = czone[-1:]
                	zone = int(czone[:-1])
		if box.tag == 'Size' and box.attrib['resolution'] == '60':
			for field in box:
				if field.tag == 'NROWS':
					nrows = int(field.text)
				if field.tag == 'NCOLS':
					ncols = int(field.text)
		if box.tag == 'Geoposition' and box.attrib['resolution'] == '60':
			for field in box:
				if field.tag == 'ULX':
					ulx = float(field.text)
				if field.tag == 'ULY':
					uly = float(field.text)
	if hemis == 'S':
		lzone = -zone
	else:
		lzone = zone
	AngleObs = { 'zone' : zone, 'hemis' : hemis, 'nrows' : nrows, 'ncols' : ncols, 'ul_x' : ulx, 'ul_y' : uly, 'obs' : [] }

	for angle in angles:
		if angle.tag == 'Viewing_Incidence_Angles_Grids':
			bandId = int(angle.attrib['bandId'])
			detectorId = int(angle.attrib['detectorId'])
			for bset in angle:
				if bset.tag == 'Zenith':
					zenith = bset
				if bset.tag == 'Azimuth':
					azimuth = bset
			for field in zenith:
				if field.tag == 'COL_STEP':
					col_step = int(field.text)
				if field.tag == 'ROW_STEP':
					row_step = int(field.text)
				if field.tag == 'Values_List':
					zvallist = field
			for field in azimuth:
				if field.tag == 'Values_List':
					avallist = field
			for rindex in range(len(zvallist)):
				zvalrow = zvallist[rindex]
				avalrow = avallist[rindex]
				zvalues = zvalrow.text.split(' ')
				avalues = avalrow.text.split(' ')
				values = zip( zvalues, avalues )
				ycoord = uly - rindex * row_step
				for cindex in range(len(values)):
					xcoord = ulx + cindex * col_step
					(lat, lon) = utm_inv( lzone, xcoord, ycoord )
					if ( values[cindex][0] != 'NaN' and values[cindex][1] != 'NaN' ):
						zen = float( values[cindex][0] ) / todeg
						az = float( values[cindex][1] ) / todeg
						( Sat, Gx ) = LOSVec( lat, lon, zen, az )
						observe = [ bandId, detectorId, xcoord, ycoord, Sat, Gx ]
						AngleObs['obs'].append( observe )

	return (tile_id, AngleObs)

def get_detfootprint( XML_File ):

	# Extract the directory
	Foot_Dir = os.path.dirname( XML_File )
	#Foot_Dir += '/QI_DATA/'
        Foot_Dir += '/qi/'
        bands = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12'
	# Parse the XML file 
	tree = ET.parse(XML_File)
	root = tree.getroot()

	# Find the detector footprint files
	footprints = []
	for child in root:
		if child.tag[-23:] == 'Quality_Indicators_Info':
			qualinfo = child

	for segment in qualinfo:
        	if segment.tag == 'Pixel_Level_QI':
			pixlevel = segment
        '''
	for qifile in pixlevel:
		if qifile.tag == 'MASK_FILENAME':
			if qifile.attrib['type'] == 'MSK_DETFOO':
				bandId = int( qifile.attrib['bandId'] )
				qifname = Foot_Dir + os.path.basename( qifile.text.strip() )
				footprints.append( (bandId, qifname) )
        ''' 
        footprints = [(i, Foot_Dir+'MSK_DETFOO_%s.gml'%bands[i]) for i in range(13)]
	bandfoot = []
	for foot in footprints:
		bandId = int( foot[0] )
		tree2 = ET.parse(foot[1])
	       	root2 = tree2.getroot()
		for child in root2:
			if child.tag[-11:] == 'maskMembers':
				thismember = child
				for feature in thismember:
					if feature.tag[-11:] == 'MaskFeature':
						for thisattribute in feature.attrib:
							if thisattribute[-2:] == 'id':
								detId = int (feature.attrib[thisattribute].split('-')[2] )
								bandName = feature.attrib[thisattribute].split('-')[1]
								thisband = { 'detId' : detId , 'bandId' : bandId, 'bandName' : bandName, 'coords' : [] }
						thisfeature = feature
						for extent in thisfeature:
							if extent.tag[-8:] == 'extentOf':
								thisextent = extent
								for polygon in thisextent:
									if polygon.tag[-7:] == 'Polygon':
										thispolygon = polygon
										for exterior in thispolygon:
											if exterior.tag[-8:] == 'exterior':
												thisexterior = exterior
												for ring in thisexterior:
													if ring.tag[-10:] == 'LinearRing':
														thisring = ring
														for poslist in thisring:
															if poslist.tag[-7:] == 'posList':
																ncoord = int( poslist.attrib['srsDimension'] )
																fields = poslist.text.split(' ')
																index = 0
																for field in fields:
																	if index == 0:
																		x = float( field )
																	elif index == 1:
																		y = float( field )
																		thisband['coords'].append( (x,y) )
																	index = (index + 1) % ncoord
																bandfoot.append(thisband)

	return bandfoot

# Define functions used to construct image observations
def Magnitude( A ):
        return sqrt( A[0]*A[0] + A[1]*A[1] + A[2]*A[2] )

def Dot( A, B ):
        return A[0]*B[0] + A[1]*B[1] + A[2]*B[2]

def CalcObs( obs, Orbit, Omega0, Lon0 ):
	Vx = [ 0.0, 0.0, 0.0 ]
	ltime = obs[6]
	Sat = obs[4]
	Gx = obs[5]
	cta = Omega0 - 2*pi*ltime/Orbit[4]
	gclat = asin( sin( cta ) * sin( Orbit[3] ) )
	gclon = Lon0 + asin( tan( gclat ) / -tan( Orbit[3] ) ) - 2*pi*ltime/86400
	Vx[0] = Orbit[2] * cos( gclat ) * cos( gclon ) - Gx[0]
	Vx[1] = Orbit[2] * cos( gclat ) * sin( gclon ) - Gx[1]
	Vx[2] = Orbit[2] * sin( gclat ) - Gx[2]
	Vdist = Magnitude( Vx )
	Vx[0] = Vx[0] / Vdist - Sat[0]
	Vx[1] = Vx[1] / Vdist - Sat[1]
	Vx[2] = Vx[2] / Vdist - Sat[2]
        return Vx

def Partial_O( obs, Orbit ):
	P0 = np.zeros( (3,4) ) 
	Omega0 = asin( sin( Orbit[0] ) / sin( Orbit[3] ) )
	Lon0 = Orbit[1] - asin( tan( Orbit[0] ) / -tan( Orbit[3] ) )
	Dx = CalcObs( obs, Orbit, Omega0, Lon0 )
	POrb = Orbit
	Pert = [ 0.00001, 0.00001, 10.0, 0.0001 ]	# Perturbations to Lat, Lon, Radius, and Inclination
	for index in range(len(Pert)):
		POrb[index] += Pert[index]
		Omega0 = asin( sin( POrb[0] ) / sin( POrb[3] ) )
		Lon0 = POrb[1] - asin( tan( POrb[0] ) / -tan( POrb[3] ) )
		Dp = CalcObs( obs, POrb, Omega0, Lon0 )
		P0[ 0, index ] = (Dp[0] - Dx[0])/Pert[index]
		P0[ 1, index ] = (Dp[1] - Dx[1])/Pert[index]
		P0[ 2, index ] = (Dp[2] - Dx[2])/Pert[index]
		POrb[index] -= Pert[index]
	
        return P0

def Partial_T( obs, Orbit ):
	P1 = [ 0.0, 0.0, 0.0 ]
	Omega0 = asin( sin( Orbit[0] ) / sin( Orbit[3] ) )
	Lon0 = Orbit[1] - asin( tan( Orbit[0] ) / -tan( Orbit[3] ) )
	Dx = CalcObs( obs, Orbit, Omega0, Lon0 )
	Pobs = obs
	Pert = 0.1		# Time perturbation
	Pobs[6] += Pert
	Dp = CalcObs( Pobs, Orbit, Omega0, Lon0 )
	P1[0] = (Dp[0] - Dx[0])/Pert
	P1[1] = (Dp[1] - Dx[1])/Pert
	P1[2] = (Dp[2] - Dx[2])/Pert
	Pobs[6] -= Pert
	
        return P1

def Fit_Time( ul_x, ul_y, Obs ):
	Time_Parms = []
	for band in range(13):
		TParm_List = []
		for sca in range(13):
			A0 = np.matrix( np.zeros( (4, 4) ) )
			L0 = np.matrix( np.zeros( (4, 1) ) )
			X0 = np.matrix( np.zeros( (4, 1) ) )
			for los in Obs:
				if los[0] == band and (sca == 0 or los[1] == sca): 
					dx = los[2] - ul_x
					dy = ul_y - los[3]
					A0[0,0] += 1.0
					A0[0,1] += dx
					A0[0,2] += dy
					A0[0,3] += dx*dy
					L0[0,0] += los[6]
					A0[1,0] += dx
					A0[1,1] += dx*dx
					A0[1,2] += dx*dy
					A0[1,3] += dx*dx*dy
					L0[1,0] += dx*los[6]
					A0[2,0] += dy
					A0[2,1] += dy*dx
					A0[2,2] += dy*dy
					A0[2,3] += dy*dx*dy
					L0[2,0] += dy*los[6]
					A0[3,0] += dx*dy
					A0[3,1] += dx*dy*dx
					A0[3,2] += dx*dy*dy
					A0[3,3] += dx*dy*dx*dy
					L0[3,0] += dx*dy*los[6]
			# Detector 0 is the band average solution which is used to strengthen detectors with few points
			if sca == 0:
				A0all = A0
				L0all = L0
			# Make sure we have a valid solution for this detector
			try:
				A0inv = A0**-1
			# Bring in the band average data for the rate terms
			except:
				if A0[0,0] < 1.0:
					A0[0,0] = 1.0
				A0[1,1] = A0all[1,1]
				A0[1,2] = A0all[1,2]
				A0[1,3] = A0all[1,3]
				A0[2,1] = A0all[2,1]
				A0[2,2] = A0all[2,2]
				A0[2,3] = A0all[2,3]
				A0[3,1] = A0all[3,1]
				A0[3,2] = A0all[3,2]
				A0[3,3] = A0all[3,3]
				L0[1,0] = L0all[1,0]
				L0[2,0] = L0all[2,0]
				L0[3,0] = L0all[3,0]
				A0inv = A0**-1
			X0 = A0inv * L0
			TParm_List.append( list( X0 ) )
		this_time = { 'band' : band, 'tmodel' : TParm_List }
		Time_Parms.append( this_time )

		# Calculate fit statistic
		rmsfit = 0
		numobs = 0
		coeffs = TParm_List
		for los in Obs:
			if los[0] == band:
				det = los[1]
				dx = los[2] - ul_x
				dy = ul_y - los[3]
                                dt = coeffs[det][0] + coeffs[det][1]*dx + coeffs[det][2]*dy + coeffs[det][3]*dx*dy - los[6]
				numobs += 1
				rmsfit += dt*dt
		if numobs > 0:
			rmsfit = sqrt(rmsfit / numobs)
			print 'Time fit for band ',band,' RMS = ',rmsfit,' seconds'

	return Time_Parms

def Fit_Orbit( AngleObs ):

	# Initialize the orbit parameters
	Orbit = [ 0.0, 0.0, 7169868.175, 98.62/todeg, 6041.958 ]	# Reference Lat, Reference Lon, Radius, Inclination, Period
	Orbit0 = [ 0.0, 0.0, 7169868.175, 98.62/todeg, 6041.958 ]	# Reference Lat, Reference Lon, Radius, Inclination, Period

	# Load the angle records
	ul_x = AngleObs['ul_x']
	ul_y = AngleObs['ul_y']
	Obs = []
	numobs = 0
	for viewrec in AngleObs['obs']:
		numobs += 1
		# Construct observation record
       		Sat = [ viewrec[4][0],  viewrec[4][1], viewrec[4][2] ]
		Gx =  viewrec[5]
		Obs.append( [ viewrec[0], viewrec[1], viewrec[2], viewrec[3], viewrec[4], viewrec[5], 0.0 ] )
		# Project the view vector out to the satellite orbital radius
		Gmag = Magnitude( Gx )
		Vdot = Dot( Sat, Gx )
		Vdist = sqrt( Orbit[2]*Orbit[2] + Vdot*Vdot - Gmag*Gmag )
		Px = [ Gx[0]+Sat[0]*Vdist, Gx[1]+Sat[1]*Vdist, Gx[2]+Sat[2]*Vdist ]
		Orbit[1] += atan2( Px[1], Px[0] )
		Orbit[0] += atan( Px[2] / sqrt( Px[0]*Px[0] + Px[1]*Px[1] ) )

	Orbit[1] /= numobs
	Orbit[0] /= numobs
	Orbit0[0] = Orbit[0]
	Orbit0[1] = Orbit[1]

	#Iterate solution for orbital parameters and observation times
	convtol = 0.001		# 1 millisecond RMS time correction
	rmstime = 15.0
	orbtol = 1.0
	orbrss = 1000.0
	first_iter = 0
	print 'Reconstructing Orbit from View Angles'
	while rmstime > convtol or orbrss > orbtol:
		AngResid = 0.0
		Omega0 = asin( sin( Orbit[0] ) / sin( Orbit[3] ) )
		Lon0 = Orbit[1] - asin( tan( Orbit[0] ) / -tan( Orbit[3] ) )
		A0 = np.matrix( np.zeros( (4, 4) ) )
		L0 = np.matrix( np.zeros( (4, 1) ) )
		X0 = np.matrix( np.zeros( (4, 1) ) )
		M1 = np.matrix( np.zeros( (4, 1) ) )
		BackSub = []
		for los in Obs:
			Vx = CalcObs( los, Orbit, Omega0, Lon0 )
			AngResid += Dot( Vx, Vx )
			V0 = np.matrix( np.array(Vx) ).reshape(3,1)
			# Calculate the partial derivatives w.r.t. the orbit parameters
			P0 = Partial_O( los, Orbit )
			P0t = np.matrix( np.transpose( P0 ) )
			P0 = np.matrix( P0 )
			A0 = A0 + P0t * P0
			L0 = L0 + P0t * V0
			P1 = Partial_T( los, Orbit )
			M1 = P0t * np.matrix(np.array(P1)).reshape(3,1)
			A1 = 1.0 / Dot( P1, P1 )
			L1 = Dot( P1, Vx ) * A1
			M1t = A1 * M1.reshape(1,4)
			A0 = A0 + M1 * M1t
			L0 = L0 + M1 * L1
			BackSub.append( [ L1, M1 * A1 ] )
		# Solve for Orbital parameter corrections
		if first_iter == 0:
			X0 = np.matrix( np.zeros( (4, 1) ) )
			first_iter = 1
		else:
			X0 = (A0**-1) * L0
		# Back Substitute for Time Corrections
		rmstime = 0.0
		for index in range(len(Obs)):
			dtime = BackSub[index][0] - Dot( BackSub[index][1], X0 )
			rmstime += dtime * dtime
			Obs[index][6] -= dtime
		# Update Orbit Parameters
		Orbit[0] -= X0[0,0]
		Orbit[1] -= X0[1,0]
		Orbit[2] -= X0[2,0]
		Orbit[3] -= X0[3,0]
		# Evaluate Observation Residual RMS
		AngResid = sqrt( AngResid / numobs )
		# Evaluate Convergence
		rmstime = sqrt( rmstime / numobs )
		# Orbit Convergence
		X0[0,0] *= 6378137.0
		X0[1,0] *= 6378137.0
		X0[3,0] *= Orbit[2]
		orbrss = sqrt( X0[0,0]*X0[0,0] + X0[1,0]*X0[1,0] + X0[2,0]*X0[2,0] + X0[3,0]*X0[3,0] )

	print 'Lat    = ', Orbit[0]*todeg
	print 'Lon    = ', Orbit[1]*todeg
	print 'Radius = ', Orbit[2]
	print 'Incl   = ', Orbit[3]*todeg
	print 'RMS Orbit Fit (meters): ', orbrss
	print 'RMS Time Fit (seconds): ', rmstime
	print 'RMS LOS Residual: ', AngResid

	print 'Fitting Tile Observation Times'

	Time_Parms = Fit_Time( ul_x, ul_y, Obs )

	return (Orbit, Time_Parms)

def CalcOrbit( ltime, Orbit ):
	cta = Orbit[5] - 2*pi*ltime/Orbit[4]
	gclat = asin( sin( cta ) * sin( Orbit[3] ) )
	gclon = Orbit[6] + asin( tan( gclat ) / -tan( Orbit[3] ) ) - 2*pi*ltime/86400
	Px = [ Orbit[2]*cos(gclat)*cos(gclon), Orbit[2]*cos(gclat)*sin(gclon), Orbit[2]*sin(gclat) ]
        return Px

#def CalcGroundVectors( AngleObs, gsd, subsamp, nrows, ncols ):
# sudipta changed above to support spatial subset
def CalcGroundVectors( AngleObs, gsd, subsamp, start_row, end_row, start_col, end_col, out_rows, out_cols ):
	GVecs = np.zeros( (out_rows, out_cols, 3) )
	ul_x = AngleObs['ul_x']
	ul_y = AngleObs['ul_y']
	zone = AngleObs['zone']
	if AngleObs['hemis'] == 'S':
		zone *= -1
	#for row in range( nrows ):
	# sudipta changed above to support spatial subset
	for row in range( start_row, end_row ):
		y = ul_y - float( row * gsd * subsamp ) - gsd/2.0
		#for col in range( ncols ):
		# sudipta changed above to support spatial subset
		for col in range( start_col, end_col ):
			x = ul_x + float( col * gsd * subsamp ) + gsd/2.0
			(lat, lon) = utm_inv( zone, x, y )
			Gx = GrndVec( lat, lon )
			GVecs[row,col,0] = Gx[0]
			GVecs[row,col,1] = Gx[1]
			GVecs[row,col,2] = Gx[2]
	return GVecs

def WriteHeader( Out_File, out_rows, out_cols, ul_x, ul_y, gsd, zone, n_or_s):
	Hdr_File = Out_File + '.hdr'
	if n_or_s == 'S':
		hemis = 'South'
	else:
		hemis = 'North'
	ofile = open( Hdr_File, 'w' )
	ofile.write('ENVI\n')
	ofile.write('description = { S2 View Angle Band File }\n')
	ofile.write('lines = %d\n' % out_rows )
	ofile.write('samples = %d\n' % out_cols )
	ofile.write('bands = 2\n')
	ofile.write('header offset = 0\n')
	ofile.write('file type = ENVI Standard\n')
	ofile.write('data type = 2\n')
	ofile.write('interleave = bsq\n')
	ofile.write('byte order = 0\n')
	ofile.write('x start = 0\n')
	ofile.write('y start = 0\n')
	ofile.write('map info = {UTM, 1.0, 1.0, %6.3lf, %6.3lf, %6.3lf, %6.3lf, %d, %s, WGS-84, units=Meters}\n' % (ul_x, ul_y, gsd, gsd, zone, hemis) )
	ofile.write('band names = {Azimuth, Zenith}\n')
	ofile.close()
	return Hdr_File

# Capture the input and output file names
XML_File = sys.argv[1]

# Set the output angle file GSD
gsd = [ 60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20 ]
subsamp = 10
if len(sys.argv) > 2:
	subsamp = int( sys.argv[2] );
print 'Using subsampling factor of %d.' % subsamp


# Sudipta spatial subset setting
sul_lat = sul_lon = slr_lat = slr_lon = None

if len(sys.argv) > 3: # expect spatial subset bbox coords as <ullat,ullon,lrlat,lrlon>
	sul_lat,sul_lon,slr_lat,slr_lon = [float(x) for x in sys.argv[3].split(',')]
	print "sul_lat,sul_lon,slr_lat,slr_lon = {},{},{},{}".format(sul_lat,sul_lon,slr_lat,slr_lon)

#sul_lat = 45.50
#slr_lat = 45.10
#sul_lon = 12.75
#slr_lon = 13.14


	

# Load the angle observations from the metadata
(Tile_ID, AngleObs) = get_angleobs( XML_File )
Tile_Base = Tile_ID.split('.')
print 'Loaded view angles from metadata for tile: ',Tile_ID.split('_')[-2][1:] + '_T_' + (Tile_ID.split('_')[-4]).split('T')[0] 

# Reconstruct the Orbit from the Angles
(Orbit, TimeParms) = Fit_Orbit( AngleObs )
Omega0 = asin( sin( Orbit[0] ) / sin( Orbit[3] ) )
Orbit.append( Omega0 )
Lon0 = Orbit[1] - asin( tan( Orbit[0] ) / -tan( Orbit[3] ) )
Orbit.append( Lon0 )

print 'Orbit processing complete'

# Load the detector footprints
BandFoot = get_detfootprint( XML_File )
print 'Loaded detector footprints from QI files'

# Loop through the bands using TimeParms which are in band order
def loop(tparms, AngleObs, gsd, subsamp, BandFoot, Orbit,  XML_File, sul_lat,sul_lon,slr_lat,slr_lon):
    band = tparms['band']
    coeffs = tparms['tmodel']
    # Set up the output array
    out_rows = AngleObs['nrows'] * 60 / gsd[band] / subsamp
    out_cols = AngleObs['ncols'] * 60 / gsd[band] / subsamp
    if subsamp > 1:
	    out_rows += 1
	    out_cols += 1
    #######################################################
    ## Sudipta addition to support spatial subset
    ######################################################
    if (sul_lat is not None):
	    # Convert the spatial subset bbox lat, lon to UTM coords.
	    ul_sx,ul_sy,_,_ = from_latlon(sul_lat, sul_lon)
	    lr_sx,lr_sy,_,_ = from_latlon(slr_lat, slr_lon)
	    
	    # now calculate the bbox row, col pairs
	    ulx = AngleObs['ul_x']
	    uly = AngleObs['ul_y']
	    ul_s_c = max(0,int((ul_sx - ulx)/gsd[band]/subsamp))
	    ul_s_r = max(0,int((uly - ul_sy)/gsd[band]/subsamp))
	    lr_s_c = min(out_cols,int((lr_sx - ulx)/gsd[band]/subsamp))
	    lr_s_r = min(out_rows,int((uly - lr_sy)/gsd[band]/subsamp))
    else:
	    ul_s_r = 0
	    ul_s_c = 0
	    lr_s_r = out_rows
	    lr_s_c = out_cols
    
    print "ul_s_r = {}, ul_s_c = {}, lr_s_r = {}, lr_s_c = {}".format(ul_s_r, ul_s_c, lr_s_r, lr_s_c)
    #sys.exit(0)
    #######################################################
    ## Sudipta addition to support spatial subset
    ######################################################
    
    #GVecs = CalcGroundVectors( AngleObs, gsd[band], subsamp, out_rows, out_cols)
    # sudipta changed above to support spatial subset
    GVecs = CalcGroundVectors( AngleObs, gsd[band], subsamp, ul_s_r, lr_s_r, ul_s_c, lr_s_c, out_rows, out_cols)
    zenith = np.zeros( (out_rows, out_cols) )
    azimuth = np.zeros( (out_rows, out_cols) )
    detcount = np.matrix( np.zeros( (out_rows, out_cols) ) )
    # Find the detector footprints for this band
    for foot in BandFoot:
	    if foot['bandId'] == band:
		    detId = foot['detId']
		    bandName = foot['bandName']
		    print 'Scanning band ', band, ' detector ', detId
		    minloc = [ foot['coords'][0][0], foot['coords'][0][1] ]
		    maxloc = [ foot['coords'][0][0], foot['coords'][0][1] ]
		    for pointloc in foot['coords']:
			    if pointloc[0] < minloc[0]:
				    minloc[0] = pointloc[0]
			    if pointloc[0] > maxloc[0]:
				    maxloc[0] = pointloc[0]
			    if pointloc[1] < minloc[1]:
				    minloc[1] = pointloc[1]
			    if pointloc[1] > maxloc[1]:
				    maxloc[1] = pointloc[1]
		    segs = []
		    for index in range(len(foot['coords'])-1):
			    point0 = foot['coords'][index]
			    point1 = foot['coords'][index+1]
			    if point1[1] == point0[1]:
				    slope = 0.0
				    intercept = point0[0]
			    else:
				    slope = (point1[0] -  point0[0]) / (point1[1] - point0[1])
				    intercept = point0[0] - slope * point0[1]
			    if point1[1] < point0[1]:
				    ymin = point1[1]
				    ymax = point0[1]
			    else:
				    ymin = point0[1]
				    ymax = point1[1]
			    segs.append( { 'y0' : point0[1], 'ymin' : ymin, 'ymax' : ymax, 'slope' : slope, 'intercept' : intercept } )
		    # Scan the array
		    #for row in range( out_rows ):
		    # sudipta changed above to support spatial subset
		    for row in range( ul_s_r, lr_s_r):
			    dy = float(row*gsd[band]*subsamp)
			    y = AngleObs['ul_y'] - dy - gsd[band]/2.0
			    if y < minloc[1] or y > maxloc[1]:
				    continue
			    xlist = []
			    for seg in segs:
				    if y == seg['y0'] or (y > seg['ymin'] and y < seg['ymax']):
					    x = seg['intercept'] + y * seg['slope']
					    xlist.append( x )
			    xlist.sort()
			    if len(xlist)%2 > 0:
				    print 'Invalid footprint intersection'
				    break
			    #for col in range( out_cols ):
			    # sudipta changed above to support spatial subset
			    for col in range( ul_s_c, lr_s_c):
				    dx = float(col*gsd[band]*subsamp)
				    x = AngleObs['ul_x'] + dx + gsd[band]/2.0
				    if x < minloc[0] or x > maxloc[0]:
					    continue
				    # See if this point is inside the footprint
				    index = 0
				    while index < len(xlist):
					    if x >= xlist[index] and x < xlist[index+1]:
						    # It is
						    calctime = coeffs[detId][0] + coeffs[detId][1]*dx + coeffs[detId][2]*dy + coeffs[detId][3]*dx*dy
						    detcount[row,col] += 1
						    Px = CalcOrbit( calctime, Orbit )
						    Gx = [ GVecs[row,col,0], GVecs[row,col,1], GVecs[row,col,2] ]
						    Vx = [ Px[0]-Gx[0], Px[1]-Gx[1], Px[2]-Gx[2] ]
						    Vlen = Magnitude( Vx )
						    Vx = [ Vx[0]/Vlen, Vx[1]/Vlen, Vx[2]/Vlen ]
						    LSRz = [ Gx[0]/a, Gx[1]/a, Gx[2]/b ]
						    Vlen = sqrt( LSRz[0]*LSRz[0] + LSRz[1]*LSRz[1] )
						    LSRx = [ -LSRz[1]/Vlen, LSRz[0]/Vlen, 0.0 ]
						    LSRy = [ LSRz[1]*LSRx[2]-LSRz[2]*LSRx[1], LSRz[2]*LSRx[0]-LSRz[0]*LSRx[2], LSRz[0]*LSRx[1]-LSRz[1]*LSRx[0] ]
						    LSRVec = [ Dot( Vx, LSRx ), Dot( Vx, LSRy ), Dot( Vx, LSRz ) ]
						    zenith[row,col] += round( acos( LSRVec[2] ) * todeg * 100.0 )
						    azimuth[row,col] +=  round( atan2( LSRVec[0], LSRVec[1] ) * todeg * 100.0 )
						    if detcount[row,col] > 1:
							    zenith[row,col] /= detcount[row,col]
							    azimuth[row,col] /= detcount[row,col]
						    index = len(xlist)
					    else:
						    index += 2
					    
    #print "row = {}, col = {}, zenith = {}".format(1000,450,zenith[1000,450])
    # Write out the angles
    
    Dir = os.path.dirname( XML_File )
    directory = Dir + '/angles/'
    try:
	os.makedirs(directory)
    except OSError as e:
	if e.errno != errno.EEXIST:
	    raise
    Out_File = directory + 'VAA_VZA_'+ bandName + '.img'
    #Out_File = Tile_Base[0][:-4] + '_Sat_' + bandName + '.img'
    #hfile = open( Out_File, 'wb' )
    #for ang in np.nditer( azimuth, order='C'):
    #	hfile.write( struct.pack('h',ang) )
    #for ang in np.nditer( zenith, order='C'):
    #	hfile.write( struct.pack('h',ang) )
    #hfile.close()
    # Sudiptas addition
    driver = gdal.GetDriverByName("ENVI")
    hfile = driver.Create(Out_File, out_rows, out_cols, 2, gdal.GDT_Int16)
    hfile.GetRasterBand(1).WriteArray(azimuth, 0, 0)
    hfile.GetRasterBand(2).WriteArray(zenith, 0, 0)
    hfile = None
    # remove the default envi header file that gdal creates to replace
    tmphdr = Dir + '/angles/VAA_VZA_'+ bandName + '.hdr' 
    #tmphdr = Tile_Base[0][:-4] + '_Sat_' + bandName + '.hdr'
    os.remove(tmphdr)
    Hdr_File = WriteHeader( Out_File, out_rows, out_cols, AngleObs['ul_x'], AngleObs['ul_y'], gsd[band]*subsamp, AngleObs['zone'], AngleObs['hemis'] )
    #print 'Created image file %s and header file %s.' % (Out_File, Hdr_File)
    #sys.exit(0)
from functools import partial 
par = partial(loop, AngleObs=AngleObs, gsd=gsd, subsamp = subsamp, BandFoot=BandFoot, Orbit = Orbit,\
              XML_File = XML_File, sul_lat=sul_lat, sul_lon = sul_lon, slr_lat = slr_lat,slr_lon = slr_lon)

from multiprocessing import Pool
p = Pool(len(TimeParms))
p.map(par, TimeParms)
#par(TimeParms[0])
#loop(TimeParms, AngleObs, gsd, subsamp, BandFoot, Orbit,  XML_File, sul_lat,sul_lon,slr_lat,slr_lon)
