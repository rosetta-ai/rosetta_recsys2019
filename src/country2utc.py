import pandas as pd
import numpy as np
import pickle
from utils import *
import pycountry
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import datetime
import pytz

tzf = TimezoneFinder()

with open('../input/train_v2.p', 'rb') as f:
    train = pickle.load(f)
    

with open('../input/test_v2.p', 'rb') as f:
    test = pickle.load(f)


def location2utc_offset(location):
    '''
    return the utc offset given the location
    '''
    location = geolocator.geocode(location)
    
    if location == None:
        return np.nan

    lat = location.latitude
    lon = location.longitude
    offset_sec = datetime.datetime.now(pytz.timezone(tzf.timezone_at(lng=lon, lat=lat)))
    return offset_sec.utcoffset().total_seconds()/60/60


all_countries = [platform2country(s) for s in set(train.platform.tolist() + test.platform.tolist())]

offsets= [location2utc_offset(c)  for c in all_countries ]

# map country to offsets
country2offsets_dict = dict(set(zip(all_countries, offsets)))
with open('../input/country2offsets_dict.p','wb') as f:
    pickle.dump(country2offsets_dict, f)