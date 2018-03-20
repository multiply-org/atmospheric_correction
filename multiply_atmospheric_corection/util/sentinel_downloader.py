#!/usr/bin/env python
"""
A simple interface to download Sentinel-1 and Sentinel-2 datasets from
the COPERNICUS Sentinel Hub.
"""
from functools import partial
import hashlib
import os
import datetime
import sys
import xml.etree.cElementTree as ET
import re

import requests
from concurrent import futures

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# hub_url = "https://scihub.copernicus.eu/dhus/search?q="
hub_url = "https://scihub.copernicus.eu/apihub/search?q="
MGRS_CONVERT = "http://legallandconverter.com/cgi-bin/shopmgrs3.cgi"
aws_url = 'http://sentinel-s2-l1c.s3.amazonaws.com/?delimiter=/&prefix=tiles/'
aws_url_dload = 'http://sentinel-s2-l1c.s3.amazonaws.com/'
requests.packages.urllib3.disable_warnings()


def get_mgrs(longitude, latitude):
    """A method that uses a website to infer the Military Grid Reference System
    tile that is used by the Amazon data buckets from the latitude/longitude

    Parameters
    -------------
    longitude: float
        The longitude in decimal degrees
    latitude: float
        The latitude in decimal degrees
    Returns
    --------
    The MGRS tile (e.g. 29TNJ)
    """
    r = requests.post(MGRS_CONVERT,
                      data=dict(latitude=latitude,
                                longitude=longitude, xcmd="Calc", cmd="gps"))
    for liner in r.text.split("\n"):
        if liner.find("<title>") >= 0:
            mgrs_tile = liner.replace("<title>", "").replace("</title>", "")
            mgrs_tile = mgrs_tile.replace(" ", "")
    try:
        return mgrs_tile[:5]  # This should be enough
    except NameError:
        return None


def calculate_md5(fname):
    hasher = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def do_query(query, user="guest", passwd="guest"):
    """
    A simple function to pass a query to the Sentinel scihub website. If
    successful this function will return the XML file back for further
    processing.

    query: str
        A query string, such as "https://scihub.copernicus.eu/dhus/odata/v1/"
        "Products?$orderby=IngestionDate%20desc&$top=100&$skip=100"
    Returns:
        The relevant XML file, or raises error
    """
    r = requests.get(query, auth=(user, passwd), verify=False)
    if r.status_code == 200:
        return r.text
    else:
        raise IOError("Something went wrong! Error code %d" % r.status_code)


def download_product(source, target, user="guest", passwd="guest"):
    """
    Download a product from the SentinelScihub site, and save it to a named
    local disk location given by ``target``.

    source: str
        A product fully qualified URL
    target: str
        A filename where to download the URL specified
    """
    md5_source = source.replace("$value", "/Checksum/Value/$value")
    r = requests.get(md5_source, auth=(user, passwd), verify=False)
    md5 = r.text
    if os.path.exists(target):
        md5_file = calculate_md5(target)
        if md5 == md5_file:
            return
    chunks = 1048576 # 1MiB...
    while True:
        LOG.debug("Getting %s" % source)
        r = requests.get(source, auth=(user, passwd), stream=True,
                         verify=False)
        if not r.ok:
            raise IOError("Can't start download... [%s]" % source)
        file_size = int(r.headers['content-length'])
        LOG.info("Downloading to -> %s" % target)
        LOG.info("%d bytes..." % file_size)
        with open(target, 'wb') as fp:
            cntr = 0
            dload = 0
            for chunk in r.iter_content(chunk_size=chunks):
                if chunk:
                    cntr += 1
                    if cntr > 100:
                        dload += cntr * chunks
                        LOG.info("\tWriting %d/%d [%5.2f %%]" % (dload, file_size,
                                                              100. * float(dload) / 
                                                              float(file_size)))
                        sys.stdout.flush()
                        cntr = 0

                    fp.write(chunk)
                    fp.flush()
                    os.fsync(fp)

        md5_file = calculate_md5(target)
        if md5_file == md5:
            break
        return


def parse_xml(xml):
    """
    Parse an OData XML file to havest some relevant information re products
    available and so on. It will return a list of dictionaries, with one
    dictionary per product returned from the query. Each dicionary will have a
    number of keys (see ``fields_of_interest``), as well as ``link`` and
    ``qui
    """
    fields_of_interest = ["filename", "identifier", "instrumentshortname",
                          "orbitnumber", "orbitdirection", "producttype",
                          "beginposition", "endposition"]
    tree = ET.ElementTree(ET.fromstring(xml))
    # Search for all the acquired images...
    granules = []
    for elem in tree.iter(tag="{http://www.w3.org/2005/Atom}entry"):
        granule = {}
        for img in elem.getchildren():
            if img.tag.find("id") >= 0:
                granule['id'] = img.text
            if img.tag.find("link") and img.attrib.has_key("href"):

                if img.attrib['href'].find("Quicklook") >= 0:
                    granule['quicklook'] = img.attrib['href']
                elif img.attrib['href'].find("$value") >= 0:
                    granule['link'] = img.attrib['href'].replace("$value", "")

            if img.attrib.has_key("name"):
                if img.attrib['name'] in fields_of_interest:
                    granule[img.attrib['name']] = img.text

        granules.append(granule)

    return granules
    # print img.tag, img.attrib, img.text
    # for x in img.getchildren():


def download_sentinel(location, input_start_date, input_sensor, output_dir,
                      input_end_date=None, username="guest", password="guest"):
    input_sensor = input_sensor.upper()
    sensor_list = ["S1", "S2"]
    if not input_sensor in sensor_list:
        raise ValueError("Sensor can only be S1 or S2. You provided %s"
                         % input_sensor)
    else:
        if input_sensor.upper() == "S1":
            sensor = "Sentinel-1"
        elif input_sensor.upper() == "S2":
            sensor = "Sentinel-2"
        sensor_str = 'platformname:%s' % sensor
        #sensor_str = 'filename:%s' % input_sensor.upper()
    try:
        start_date = datetime.datetime.strptime(input_start_date,
                                                "%Y.%m.%d").isoformat()
    except ValueError:
        try:
            start_date = datetime.datetime.strptime(input_start_date,
                                                    "%Y-%m-%d").isoformat()
        except ValueError:
            start_date = datetime.datetime.strptime(input_start_date,
                                                    "%Y/%j").isoformat()
    start_date = start_date + "Z"

    if input_end_date is None:
        end_date = "NOW"
    else:
        try:
            end_date = datetime.datetime.strptime(input_end_date,
                                                  "%Y.%m.%d").isoformat()
        except ValueError:
            try:
                end_date = datetime.datetime.strptime(input_end_date,
                                                      "%Y-%m-%d").isoformat()
            except ValueError:
                end_date = datetime.datetime.strptime(input_end_date,
                                                      "%Y/%j").isoformat()

    if len(location) == 2:
        location_str = 'footprint:"Intersects(%f, %f)"' % (location[0], location[1])
    elif len(location) == 4:
        location_str = 'footprint:"Intersects( POLYGON(( " + \
            "%f %f, %f %f, %f %f, %f %f, %f %f) ))"' % (
            location[0], location[0],
            location[0], location[1],
            location[1], location[1],
            location[1], location[0],
            location[0], location[0])

    time_str = 'beginposition:[%s TO %s]' % (start_date, end_date)

    query = "%s AND %s AND %s" % (location_str, time_str, sensor_str)
    query = "%s%s" % (hub_url, query)
    # query = "%s%s" % ( hub_url, urllib2.quote(query ) )
    LOG.debug(query)
    result = do_query(query, user=username, passwd=password)
    granules = parse_xml(result)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    ret_files = []
    for granule in granules:
        download_product(granule['link'] + "$value", os.path.join(output_dir,
                        granule['filename'].replace("SAFE", "zip")),
                        user=username, passwd=password)
        ret_files.append(os.path.join(output_dir,
                                      granule['filename'].replace("SAFE", "zip")))

    return granules, ret_files


def parse_aws_xml(xml_text, clouds=None):
    
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()
    files_to_get = []
    for elem in tree.iter():
        for k in elem.getchildren():
            if k.tag.find ("Key") >= 0:
                if k.text.find ("tiles") >= 0:
                    files_to_get.append( k.text )
                    
    if len(files_to_get) > 0 and clouds is not None:
        
        for fich in files_to_get:
            if fich.find ("metadata.xml") >= 0:
                metadata_file = aws_url_dload + fich
                r = requests.get(metadata_file)
                tree = ET.ElementTree(ET.fromstring(r.text))
                root = tree.getroot()
                for cl in root.iter("CLOUDY_PIXEL_PERCENTAGE"):
                    if float(cl.text) > clouds:
                        return []
                    else:
                        return files_to_get
    return files_to_get
    
def aws_grabber(url, output_dir):
    output_fname = os.path.join(output_dir, url.split("tiles/")[-1])
    if not os.path.exists(os.path.dirname (output_fname)):
        # We should never get here, as the directory should always exist
        # Note that in parallel, this can sometimes create a race condition
        # Groan
        os.makedirs (os.path.dirname(output_fname))
    with open(output_fname, 'wb') as fp:
        while True:
            try:
                r = requests.get(url, stream=True)
                break
            except requests.execeptions.ConnectionError:
                time.sleep ( 240 )
        for block in r.iter_content(8192):
            fp.write(block)
    LOG.debug("Done with %s" % output_fname)
    return output_fname


def download_sentinel_amazon(start_date, output_dir,
                             tile=None,
                             longitude=None, latitude=None,
                             end_date=None, n_threads=15, just_previews=False,
                             verbose=False, clouds=None):
    """A method to download data from the Amazon cloud """
    # First, we get hold of the MGRS reference...
    if tile is None:
        mgrs_reference = get_mgrs(longitude, latitude)
    else:
        mgrs_reference = tile
    #if verbose:
    #    print("We need MGRS reference %s" % mgrs_reference)
    utm_code = mgrs_reference[:2]
    lat_band = mgrs_reference[2]
    square = mgrs_reference[3:]
    #LOG.info("Location coordinates: %s" % mgrs_reference )

    front_url = aws_url + "%s/%s/%s" % (utm_code, lat_band, square)
    this_date = start_date
    one_day = datetime.timedelta(days=1)
    files_to_download = []
    if end_date is None:
        end_date = datetime.datetime.today()
    #LOG.info("Scanning archive...")
    acqs_to_dload = 0
    this_dates = []
    
    while this_date <= end_date:
        up_url = front_url + '/%d/%d/%d/'%(this_date.year, this_date.month, this_date.day)
        r = requests.get(up_url)
        views = [i.split('</Prefix>')[0] for i in r.text.split('<Prefix>') if ('</Prefix>' in i) & (len(i.split('</Prefix>')[0].split('/'))==9)]
        for view in range(len(views)):
            the_url = "{0}{1}".format(front_url, "/{0:d}/{1:d}/{2:d}/{3:d}/".format(
                this_date.year, this_date.month, this_date.day, view))
            r = requests.get(the_url)
            
            more_files = parse_aws_xml (r.text, clouds=clouds)
            
            if len(more_files) > 0:
                acqs_to_dload += 1
                rqi = requests.get (the_url + "qi/")
                raux = requests.get (the_url + "aux/")
                qi = parse_aws_xml (rqi.text)
                aux = parse_aws_xml (raux.text)
                more_files.extend (qi)
                more_files.extend (aux)
                files_to_download.extend (more_files)
                #LOG.info("Will download data for %s..." % 
                #         this_date.strftime("%Y/%m/%d"))
            this_dates.append(this_date)
        this_date += one_day
    #LOG.info("Will download %d acquisitions" % acqs_to_dload)
    the_urls = []
    if just_previews:
        the_files = []
        for fich in files_to_download:
            if fich.find ("preview") >= 0:
                the_files.append ( fich )
        files_to_download = the_files
        
    for fich in files_to_download:
        the_urls.append(aws_url_dload + fich)
        ootput_dir = os.path.dirname ( os.path.join(output_dir,
                                                    fich.split("tiles/")[-1]))
        if not os.path.exists ( ootput_dir ):
            
            #LOG.info("Creating output directory (%s)" % ootput_dir)
            os.makedirs ( ootput_dir )
    ok_files = []
    #LOG.info( "Downloading a grand total of %d files" % 
    #        len ( files_to_download ))
    download_granule_patch = partial(aws_grabber, output_dir=output_dir)
    with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for fich in executor.map(download_granule_patch, the_urls):
            ok_files.append(fich)
    return this_dates
if __name__ == "__main__":    # location = (43.3650, -8.4100)
    # input_start_date = "2015.01.01"
    # input_end_date = None

    # username = "guest"
    # password = "guest"

    # input_sensor = "S2"


    # output_dir = "/data/selene/ucfajlg/tmp/"
    # granules, retfiles = download_sentinel ( location, input_start_date,
    # input_sensor, output_dir )
    lng = -8.4100
    lat = 43.3650
    #lat = 39.0985 # Barrax
    #lng = -2.1082 
    #lat = 28.55 # Libya 4
    #lng = 23.39
    print("Testing S2 on AWS...")
    download_sentinel_amazon(lat, lng, datetime.datetime(2016, 1, 11),
                             "/tmp/", end_date=datetime.datetime(2016, 12, 25),
                             clouds=10)
    #print "Testing S2 on COPERNICUS scientific hub"
    #location=(lat,lng)
    #input_start_date="2017.1.11"
    #input_sensor="S2"
    #output_dir="/tmp/"
    #print "Set username and password variables for Sentinel hub!!!"
    #download_sentinel(location, input_start_date, input_sensor, output_dir,
                      #input_end_date=None, username, password)
