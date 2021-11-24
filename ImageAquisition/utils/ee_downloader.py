'''
More complex interface than Google's to download images from Planet. Unlike Google, Planet allows us to query images at a certain time. However, Planet's API is not great and there are a variety of issues ranging from a little annoying to fairly serious. This interface simplifies the use from a user's perspective.
'''
import math
import os
import json
import re
import ee
ee.Authenticate()
ee.Initialize()


class EEDownloader:
    def __init__(self, item_type='LANDSAT/LE07/C01/T1_ANNUAL_RAW', item_select=['B3', 'B2', 'B1'], drive_folder="coordinates_poverty_tcc_test", dimensions='3000x3000'):
        self.item_type = item_type
        self.item_select = item_select
        self.drive_folder = drive_folder
        self.dimensions = dimensions
    
    def create_cords(self, lat, lon, zoom):
        xtile, ytile = deg_to_tile(lat, lon, zoom)

        coords = [tilexy_to_deg(xtile, ytile, zoom, a, b) for a,b in [(0,0), (0,255), (255,255), (255,0)]]
        return [[b,a] for a,b in coords]
    
    def download_image(self, lat, lon, min_year, min_month, max_year, max_month, zoom=14, image_name=None):
        '''
        Use this method to download an image at a lat, lon in some time range
        If multiple images are available, the latest is downloaded
        
        I would not increase zoom
        cloud_max is the maximum cloud filter, defaulting to 5%
        '''
        # Load a landsat image and select three bands.
        dataset = ee.ImageCollection(self.item_type) \
            .filterDate(str(min_year)+'-'+str(min_month)+'-01', str(max_year)+'-'+str(max_month)+'-01') \
            .select(self.item_select)
                          
        landsat = ee.Image(dataset.first())

        # Create a geometry representing an export region.
        coord = self.create_cords(lat, lon, zoom)
        geometry = ee.Geometry.Polygon(coord)

        # Export the image, specifying scale and region.
        task_config = {
            #'scale': 100,
            'region': geometry,
            'dimensions': self.dimensions,
            'driveFolder': self.drive_folder,
            'formatOptions': {
                'cloudOptimized': True
            }
        }
        
        if(image_name is None):
            image_name = re.sub(r'\.', r'_', str(lat))+"_"+re.sub(r'\.', r'_', str(lon))

        task = ee.batch.Export.image(landsat, image_name, task_config)
        task.start() 
        
        return ("", task_config)

    
"""
Important geoconversion functions
"""

def tilexy_to_deg(xtile, ytile, zoom, x, y):
    """Converts a specific location on a tile (x,y) to geocoordinates."""
    decimal_x = xtile + x / 256
    decimal_y = ytile + y / 256
    n = 2.0 ** zoom
    lon_deg = decimal_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * decimal_y / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def deg_to_tilexy(lat_deg, lon_deg, zoom):
    """Converts geocoordinates to an x,y position on a tile."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = ((lon_deg + 180.0) / 360.0 * n)
    y = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad)))
        / math.pi) / 2.0 * n)
    return (int((x % 1) * 256), int((y % 1) * 256))

def tile_to_deg(xtile, ytile, zoom):
    """Returns the coordinates of the northwest corner of a Slippy Map
    x,y tile"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def deg_to_tile(lat_deg, lon_deg, zoom):
    """Converts coordinates into the nearest x,y Slippy Map tile"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad)))
                 / math.pi) / 2.0 * n)
    return (xtile, ytile)

       
