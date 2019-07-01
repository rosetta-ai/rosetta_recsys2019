import pickle
from constant import *
import torch
import os
import random
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from scipy.special import erfinv
from ordered_set import OrderedSet
import scipy
from collections import Counter
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import pycountry
import datetime
import pytz

tzf = TimezoneFinder()


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


def platform2country(platform):
    '''
    return country name given platform
    '''
    
    if pycountry.countries.get(alpha_2=platform) != None:
        try:
            return pycountry.countries.get(alpha_2=platform).common_name
        except:
            return pycountry.countries.get(alpha_2=platform).name 
        
            
    else:
        return np.nan


def location2utc_offset(location):
    '''
    return the utc offset given the location
    '''
    geolocator = Nominatim(user_agent=str(location))
    # print(location)
    location = geolocator.geocode(location)
    
    if location == None:
        return np.nan
    try:
        lat = location.latitude
        lon = location.longitude
        offset_sec = datetime.datetime.now(pytz.timezone(tzf.timezone_at(lng=lon, lat=lat)))
        return offset_sec.utcoffset().total_seconds()/60/60
    except:
        return np.nan

def find_longest_repetitive_sequences(sequence):
    '''
    returns a dict that maps each element with the length of its longest repetitive sequneces in the list
    args:
        sequence: list
    
    '''
    counter = Counter()
    current_element = None

    # iterate the sequence
    for element in sequence:

        if current_element == None:
            current_element = element
            current_rep = 1
        elif element == current_element:
            current_rep += 1
        elif element != current_element:
            # update the element with the longest rep 
            if counter[current_element]  < current_rep:
                counter[current_element] = current_rep
            current_rep = 1
            current_element = element
    # update the element with the longest rep outside the loop
    if len(sequence) > 0 and counter[current_element]  < current_rep:
        counter[current_element] = current_rep

    return counter




def qcut_safe(prices, q):
    nbins=min(q, len(prices))
    result = pd.qcut(prices, nbins, labels=np.arange(nbins) )

    return result



class GaussRankScaler():

    def __init__( self ):
        self.epsilon = 1e-9
        self.lower = -1 + self.epsilon
        self.upper =  1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform( self, X ):

        i = np.argsort( X, axis = 0 )
        j = np.argsort( i, axis = 0 )

        assert ( j.min() == 0 ).all()
        assert ( j.max() == len( j ) - 1 ).all()

        j_range = len( j ) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = scipy.special.erfinv( transformed )
        ############
        # transformed = transformed - np.mean(transformed)

        return transformed
    
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_rank(inp, to_np=False):
    sorted_inp = sorted(inp)
    out = [sorted_inp.index(i) for i in inp]
    if to_np:
        out = np.array(out)
    return out

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


class CategoricalEncoder():
    '''
    This class is for those operating on large data, in which sklearn's LabelEncoder class may take too much time.
    This encoder is only suitable for 1-d array/ list. You may modify it to become n-d compatible.
    '''
    def __init__(self):
        self.f_dict = {}
        self.r_dict = {}

    def fit(self, array):
        '''

        :param array: list or np array
        :return: None
        '''

        unique_elements = OrderedSet(array)
        # unique_elements = sorted(unique_elements)
        # print(DUMMY_ITEM in unique_elements)
        # print('-1' in unique_elements)
        self.n_elements = 0
        self.f_dict = {}
        self.r_dict = {}

        for e in unique_elements:
            self.f_dict[e] = self.n_elements
            self.r_dict[self.n_elements] = e
            self.n_elements += 1


    def continue_fit(self, array):
        '''
        Do not refresh n_elements, count from the latest n_elements.
        :param array:
        :return: None
        '''
        unique_elements = set(array)
        for e in unique_elements:
            if e not in self.f_dict:
                self.f_dict[e] = self.n_elements
                self.r_dict[self.n_elements] = e
                self.n_elements += 1


    def reverse_transform(self, transformed_array, to_np=False):
        '''

        :param transformed_array: list or np array
        :return: array: np array with the same shape as input
        '''


        array = [self.r_dict[e] for e in transformed_array]
        if to_np:
            array = np.array(array)
        return array


    def transform(self, array, to_np=False):
        '''

        :param array: array list or np array
        :return: list or np array with the same shape as the input
        '''
        transformed_array = [self.f_dict[e] for e in array]
        if to_np:
            transformed_array = np.array(transformed_array)
        return transformed_array

    def fit_transform(self, array, to_np=False):
        '''

        :param array: array list or np array
        :return: list or np array with the same shape as the input
        '''
        self.fit(array)
        return self.transform(array, to_np)

def str2bool(v):
    return v.lower() in ('true')

def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['learning_rate'] , weight_decay=params['weight_decay'],  eps=1e-07, amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['learning_rate'],)
    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    return optimizer

def get_attn_key_pad_mask(seq_k, seq_q, transformed_dummy_value):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(transformed_dummy_value)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def compute_mean_reciprocal_rank(rs):
    '''
    rs: 2d array

    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    '''

    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.5f} s'.format(name,(time.time() - t0)))


