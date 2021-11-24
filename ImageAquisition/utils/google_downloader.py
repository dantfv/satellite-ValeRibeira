'''
Very simple download interface to download images from Google's Static Maps API
'''

import requests
import matplotlib.pyplot as plt
from io import BytesIO

class GoogleDownloader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.url = 'https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size=400x400&maptype=satellite&key={}'
    
    def download_image(self, lat, long, zoom):
        res = requests.get(self.url.format(lat, long, zoom, self.access_token))
        # server needs to make image available, takes a few seconds
        if res.status_code == 403:
            return 'RETRY', None
        assert res.status_code < 400, print(f'Error - failed to download {lat}, {long}, {zoom}')

        image = plt.imread(BytesIO(res.content))
        features = {
                "type": "center_point",
                "coordinates": [lat, long],
                "zoom": zoom
            }
        
        return (image, features)
    
