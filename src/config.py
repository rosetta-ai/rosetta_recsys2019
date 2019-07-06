import numpy as np 
from datetime import datetime
import torch
import inspect
from loss import *

class Configuration(object):

    def __init__(self):
        self.alias = None
        self.num_epochs = 1
        self.batch_size = 1024
        self.optimizer = 'adam'
        self.use_cuda = True
        self.device_id = 0 
        self.early_stopping = 1
        self.loss = torch.nn.BCELoss
        self.debug = True
        self.sub_sample = True
        self.slack = True
        self.use_test = True if not self.sub_sample else False

    def __getitem__(cls, x):
        '''make configuration subscriptable'''
        return getattr(cls, x)

    def __setitem__(cls, x, v):
        '''make configuration subscriptable'''
        return setattr(cls, x, v)

    def get_attributes(self): 
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
 
        # store only not the default attribute __xx__
        attribute_tuple_list = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

        attribute_dict = {}
        for tup in attribute_tuple_list:
            key = tup[0]
            value = tup[1]
            if key == 'loss':
                value = str(value)
            # convert numpy value to float
            if type(value) == np.float64:
                value = float(value)
            attribute_dict[key] = value

        return attribute_dict

    def set_model_dir(self):
        now = datetime.now()

        time_info = f'{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
        self.model_dir = f'model_weights/{self.alias}-{time_info}.model'

    def attribute_to_integer(self):
        '''Convert the attributes in self.integer_attribute_list to integer'''

        for attribute in self.integer_attribute_list:
            self[attribute] = int(self[attribute])

    def set_config(self, config):
        for key in config:
            self[key] = config[key]


class NNConfiguration(Configuration):

    def __init__(self):
        super(NNConfiguration, self).__init__()
        self.categorical_emb_dim = 128

        self.alias = 'NN'
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.weight_decay = 0
        self.sequence_length = 10
        self.sess_length = 30
        self.num_embeddings = {}
        self.verbose = True
        self.hidden_dims = [256 , 128]
        self.dropout_rate = 0
        self.loss = torch.nn.BCELoss


class LGBConfiguration(Configuration):

    def __init__(self):
        super(LGBConfiguration, self).__init__()
        self.categorical_emb_dim = 128
        self.alias = 'LGB'
        self.sequence_length = 10
        self.sess_length = 30

class XGBConfiguration(Configuration):

    def __init__(self):
        super(XGBConfiguration, self).__init__()
        self.categorical_emb_dim = 128
        self.alias = 'XGB'
        self.sequence_length = 10
        self.sess_length = 30        

