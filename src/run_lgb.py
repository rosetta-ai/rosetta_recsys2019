import torch
import numpy as np
import pandas as pd
import pickle
import gc
from constant import *
from utils import *
from config import *
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import lightgbm as lgb
import scipy
from sklearn.decomposition import TruncatedSVD
import multiprocessing
from ordered_set import OrderedSet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import log_loss
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import itertools
from scipy.sparse import csr_matrix

configuration = LGBConfiguration()

model_name='lgb_imp_cooc_v2'

if configuration.sub_sample:
    model_name += '_140k'
else:
    model_name += '_all'

if configuration.use_test:
    model_name += '_ut'

seed_everything(42)

########################################################### Load data ######################################################################
with open( f'{input_dir}/train_v2.p', 'rb') as f:
    train = pickle.load(f)
    train['id']= np.arange(len(train))

with open(f'{input_dir}/test_v2.p', 'rb') as f:
    test = pickle.load(f)
    test['id'] = np.arange( len(train), len(train)+ len(test))

with open('../input/item_metadata.p', 'rb') as f:
    item_meta = pickle.load(f)
    item_meta['properties'] = item_meta.properties.apply(lambda x: x.split('|'))
    item_meta['item_id'] = item_meta['item_id'].apply(str)

# whether to use sub sample of the data to speed up the evaluation
if configuration.sub_sample:    
    with open('../input/selected_users_140k.p', 'rb') as f:
        selected_users = pickle.load(f)
    
    train = train.loc[train.user_id.isin(selected_users),:]

# check if the code can run with debug mode
if configuration.debug:
    train = train.sample(1000)
    test = test.sample(1000)

with timer("preprocessing"):
    
    # change columns name
    train.rename(columns={'reference': 'item_id', 'action_type': 'action'}, inplace=True)
    test.rename(columns={'reference': 'item_id', 'action_type': 'action'}, inplace=True)

    # concatenate the action and reference in string format as these refernce are not actually item id
    train.loc[train.action=='change of sort order','action'] = train.loc[train.action=='change of sort order'].apply(lambda row: row.action + str(row.item_id), axis=1)
    test.loc[test.action=='change of sort order','action'] = test.loc[test.action=='change of sort order'].apply(lambda row: row.action + str(row.item_id), axis=1)


    train.loc[train.action=='filter selection','action'] = train.loc[train.action=='filter selection'].apply(lambda row: row.action + str(row.item_id), axis=1)
    test.loc[test.action=='filter selection','action'] = test.loc[test.action=='filter selection'].apply(lambda row: row.action + str(row.item_id), axis=1)


    # wipe out the item id associated with these actions, reason same as the above
    train.loc[train.action.str.contains('change of sort order'), 'item_id'] = DUMMY_ITEM
    test.loc[test.action.str.contains('change of sort order'), 'item_id'] = DUMMY_ITEM

    train.loc[train.action.str.contains('search for poi'), 'item_id'] = DUMMY_ITEM
    test.loc[test.action.str.contains('search for poi'), 'item_id'] = DUMMY_ITEM        

    train.loc[train.action.str.contains('filter selection'), 'item_id'] = DUMMY_ITEM
    test.loc[test.action.str.contains('filter selection'), 'item_id'] = DUMMY_ITEM        

    train.loc[train.action.str.contains('search for destination'), 'item_id'] = DUMMY_ITEM
    test.loc[test.action.str.contains('search for destination'), 'item_id'] = DUMMY_ITEM  

    # remove training example where clicked item is not in the impressions
    train['in_impressions'] = True
    train.loc[~train.impressions.isna(), 'in_impressions'] = train.loc[~train.impressions.isna()].apply(lambda row:row.item_id in row.impressions.split('|'), axis=1)
    train = train.loc[train.in_impressions].drop('in_impressions', axis=1).reset_index(drop=True)

    test['in_impressions'] = True
    test.loc[(~test.impressions.isna()) & (~test.item_id.isna()), 'in_impressions'] = test.loc[(~test.impressions.isna())& (~test.item_id.isna())].apply(lambda row:row.item_id in row.impressions.split('|'), axis=1)
    test = test.loc[test.in_impressions].drop('in_impressions', axis=1).reset_index(drop=True)

    # parse impressions and prices list from string to list
    train['item_id'] = train['item_id'].apply(str)
    train.loc[~train.impressions.isna(),'impressions'] = train.loc[~train.impressions.isna()].impressions.apply(lambda x: x.split('|'))
    train.loc[~train.prices.isna(), 'prices'] = train.loc[~train.prices.isna()].prices.apply(lambda x: x.split('|')).apply(lambda x: [int(p) for p in x])

    test['item_id'] = test['item_id'].apply(str)
    test.loc[~test.impressions.isna(),'impressions'] = test.loc[~test.impressions.isna()].impressions.apply(lambda x: x.split('|'))
    test.loc[~test.prices.isna(),'prices'] = test.loc[~test.prices.isna()].prices.apply(lambda x: x.split('|')).apply(lambda x: [int(p) for p in x])
    
    # compute the last interacted item by shifted the item_id by 1 position
    train['last_item'] = np.nan
    test['last_item'] = np.nan
    
    train_shifted_item_id = [DUMMY_ITEM] + train.item_id.values[:-1].tolist()
    test_shifted_item_id = [DUMMY_ITEM] + test.item_id.values[:-1].tolist()

    # compute the last interacted item by shifted the item_id by 2 position
    train['last_item'] = train_shifted_item_id
    test['last_item'] = test_shifted_item_id

    train_shifted_item_id = [DUMMY_ITEM] *2 + train.item_id.values[:-2].tolist()
    test_shifted_item_id = [DUMMY_ITEM] *2  + test.item_id.values[:-2].tolist()

    # compute the last interacted item by shifted the item_id by 3 position
    train['second_last_item'] = train_shifted_item_id
    test['second_last_item'] = test_shifted_item_id

    train_shifted_item_id = [DUMMY_ITEM] *3 + train.item_id.values[:-3].tolist()
    test_shifted_item_id = [DUMMY_ITEM] *3  + test.item_id.values[:-3].tolist()

    train['third_last_item'] = train_shifted_item_id
    test['third_last_item'] = test_shifted_item_id

    # mask out the last interacted item if that interaction comes first in its session
    train['step_rank'] = train.groupby('session_id')['step'].rank(method='max', ascending=True)
    test['step_rank'] = test.groupby('session_id')['step'].rank(method='max', ascending=True)

    # fill the invalid shifted last n item with a constant number
    train.loc[(train.step_rank == 1) & (train.action == 'clickout item'), 'last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 1) & (test.action == 'clickout item'), 'last_item'] = DUMMY_ITEM

    train.loc[(train.step_rank == 2) & (train.action == 'clickout item'), 'second_last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 2) & (test.action == 'clickout item'), 'second_last_item'] = DUMMY_ITEM

    train.loc[(train.step_rank == 3) & (train.action == 'clickout item'), 'third_last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 3) & (test.action == 'clickout item'), 'third_last_item'] = DUMMY_ITEM
    
    
    # ignore this
    keep_columns = ['session_id', 'user_id','item_id', 'impressions','prices', 'city', 'step', 'last_item']
    all_cat_columns = ['item_id', 'city', 'platform', 'device','country','country_platform','action','device_platform']

    
    # generate country from city
    train['country'] = train.city.apply(lambda x:x.split(',')[-1])
    test['country'] = test.city.apply(lambda x:x.split(',')[-1])
    
    # concate country and platform in string format as a new feature
    train['country_platform'] = train.apply(lambda row: row.country + row.platform, axis=1)
    test['country_platform'] = test.apply(lambda row: row.country + row.platform, axis=1)

    train['device_platform'] = train.apply(lambda row: row.device + row.platform, axis=1)
    test['device_platform'] = test.apply(lambda row: row.device + row.platform, axis=1)
    # filter out rows where reference doesn't present in impression
    # train = train.loc[train.apply(lambda row:row.item_id in row.impressions, axis=1),:]

print("train shape",train.shape)

    
# concat train and test
data = pd.concat([train, test], axis=0)
data = data.reset_index(drop=True)

# compute a dicationary that maps session id to the sequence of item ids in that session
train_session_interactions = dict(train.groupby('session_id')['item_id'].apply(list))
test_session_interactions = dict(test.groupby('session_id')['item_id'].apply(list))


# compute a dicationary that maps session id to the sequence of action in that session
train_session_actions = dict(train.groupby('session_id')['action'].apply(list))
test_session_actions = dict(test.groupby('session_id')['action'].apply(list))

# compute session session step since the "step" column in some session is not correctly order
train['sess_step'] = train.groupby('session_id')['timestamp'].rank(method='max').apply(int)
test['sess_step'] = test.groupby('session_id')['timestamp'].rank(method='max').apply(int)




data_feature = data.loc[:,['id','step','session_id', 'timestamp','platform','country']].copy()

# compute the time difference between each step
data_feature['time_diff'] = data.groupby('session_id')['timestamp'].diff()

# compute the difference of time difference between each step
data_feature['time_diff_diff'] = data_feature.groupby('session_id')['time_diff'].diff()

# compute the difference of the difference of time difference between each step
data_feature['time_diff_diff_diff'] = data_feature.groupby('session_id')['time_diff_diff'].diff()

# compute the time difference from 2 steps ahead
data_feature['time_diff_2'] = data.groupby('session_id')['timestamp'].diff().shift(1)

# compute the time difference from 3 steps ahead
data_feature['time_diff_3'] = data.groupby('session_id')['timestamp'].diff().shift(2)

data_feature['hour']= pd.to_datetime(data_feature.timestamp, unit='s').dt.hour//4

# map platform to country
data_feature['mapped_country'] = data_feature.platform.apply(platform2country)


# load the precomputed country to utc offsets from geopy
with open('../input/country2offsets_dict.p','rb') as f:
    platform_country2offsets_dict = pickle.load(f)
data_feature['platform2country_utc_offsets'] = data_feature.mapped_country.map(platform_country2offsets_dict)


# trasnform time difference with rank gauss
data_feature['rg_time_diff'] = GaussRankScaler().fit_transform(data_feature['time_diff'].values)

# compute the log of step
data_feature['step_log'] = np.log1p(data_feature['step'])

# drop the useless columns
data_feature = data_feature.drop(['session_id','step','timestamp','hour','platform','country','mapped_country'], axis=1)



    
# merge train, test with data_feature
train = train.merge(data_feature, on='id', how='left')
test = test.merge(data_feature, on='id', how='left')


# compute the sequence of time difference in each session
train_session_time_diff = dict(train.groupby('session_id')['time_diff'].apply(list))
test_session_time_diff = dict(test.groupby('session_id')['time_diff'].apply(list))

# encode the categorical feture
cat_encoders = {}
for col in all_cat_columns:
    cat_encoders[col] = CategoricalEncoder()


all_items = []
for imp in data.loc[~data.impressions.isna()].impressions.tolist() + [data.item_id.apply(str).tolist()] :
    all_items += imp

unique_items = OrderedSet(all_items)
unique_actions = OrderedSet(data.action.values)

cat_encoders['item_id'].fit(list(unique_items) + [DUMMY_ITEM])
cat_encoders['action'].fit( list(unique_actions) + [DUMMY_ACTION])
for col in  ['city', 'platform', 'device','country','country_platform', 'device_platform']:

    cat_encoders[col].fit(data[col].tolist() )


# transform all the categorical columns to continuous integer
for col in all_cat_columns:
    train[col] = cat_encoders[col].transform(train[col].values)
    test[col] = cat_encoders[col].transform(test[col].values)


# get the encoded action
transformed_clickout_action = cat_encoders['action'].transform(['clickout item'])[0]
transformed_dummy_item = cat_encoders['item_id'].transform([DUMMY_ITEM])[0]
transformed_dummy_action = cat_encoders['action'].transform([DUMMY_ACTION])[0]
transformed_interaction_image = cat_encoders['action'].transform(['interaction item image'])[0]
transformed_interaction_deals = cat_encoders['action'].transform(['interaction item deals'])[0]
transformed_interaction_info = cat_encoders['action'].transform(['interaction item info'])[0]
transformed_interaction_rating = cat_encoders['action'].transform(['interaction item rating'])[0]

# transform session interactions and pad dummy in front of all of them
for session_id, item_list in train_session_interactions.items():
    train_session_interactions[session_id] = [transformed_dummy_item] * configuration.sess_length + cat_encoders['item_id'].transform(item_list)

for session_id, item_list in test_session_interactions.items():
    test_session_interactions[session_id] = [transformed_dummy_item] * configuration.sess_length + cat_encoders['item_id'].transform(item_list)
    
for session_id, action_list in train_session_actions.items():
    train_session_actions[session_id] = [transformed_dummy_action] * configuration.sess_length + cat_encoders['action'].transform(action_list)

for session_id, action_list in test_session_actions.items():
    test_session_actions[session_id] = [transformed_dummy_action] * configuration.sess_length + cat_encoders['action'].transform(action_list) 
    

### compute co-occurence matrix
implicit_train = train.loc[train.action != transformed_clickout_action, :]
implicit_test = test.loc[test.action != transformed_clickout_action, :]

# get all interacted items in a session
implicit_all = pd.concat([implicit_train , implicit_test], axis=0)
# a list of list containing items in the same session
co_occ_items = implicit_all.groupby('session_id').item_id.apply(list).to_dict().values()
co_occ_permutes = [list(itertools.permutations(set(items), 2)) for items in co_occ_items]

#aggregate co-ocurrence across sessions
co_occ_coordinates = []
for coordinates in  co_occ_permutes:
    co_occ_coordinates += coordinates

#construct csr
row, col, values = zip(*((i,j,1) for i,j in co_occ_coordinates ))
co_occ_matrix= csr_matrix((values, (row, col)), shape=(cat_encoders['item_id'].n_elements, cat_encoders['item_id'].n_elements), dtype=np.float32)

co_occ_matrix_csc = co_occ_matrix.tocsc()

print("max entry: ", co_occ_matrix.max())


### compute co-occurence matrix for imp list

# imp_co_occ_items = train.loc[~train.impressions.isna()].impressions.apply(lambda x: cat_encoders['item_id'].transform(x)).values.tolist() + test.loc[~test.impressions.isna()].impressions.apply(lambda x: cat_encoders['item_id'].transform(x)).values.tolist()
# imp_co_occ_permutes = [list(itertools.permutations(set(items), 2)) for items in imp_co_occ_items]

# #aggregate co-ocurrence across sessions
# imp_co_occ_coordinates = []
# for coordinates in  imp_co_occ_permutes:
#     imp_co_occ_coordinates += coordinates

# #construct csr
# row, col, values = zip(*((i,j,1) for i,j in imp_co_occ_coordinates ))
# imp_co_occ_matrix= csr_matrix((values, (row, col)), shape=(cat_encoders['item_id'].n_elements, cat_encoders['item_id'].n_elements), dtype=np.float32)

# imp_co_occ_matrix_csc = imp_co_occ_matrix.tocsc()

# print("max entry: ", imp_co_occ_matrix.max())

# categorically encode last, second last and third item
train['last_item'] = cat_encoders['item_id'].transform(train['last_item'].values)
test['last_item'] = cat_encoders['item_id'].transform(test['last_item'].values)

train['second_last_item'] = cat_encoders['item_id'].transform(train.second_last_item.values)
test['second_last_item'] = cat_encoders['item_id'].transform(test.second_last_item.values)

train['third_last_item'] = cat_encoders['item_id'].transform(train.third_last_item.values)
test['third_last_item'] = cat_encoders['item_id'].transform(test.third_last_item.values)




# genetate item properties features 
item_meta = item_meta.loc[item_meta.item_id.isin(unique_items),:]
# item_meta multi-hot
item_meta['item_id'] = cat_encoders['item_id'].transform(item_meta['item_id'].values)
item_meta['star'] = np.nan
item_meta.loc[item_meta.properties.apply(lambda x: '1 Star' in x), 'star'] = 1
item_meta.loc[item_meta.properties.apply(lambda x: '2 Star' in x), 'star'] = 2
item_meta.loc[item_meta.properties.apply(lambda x: '3 Star' in x), 'star'] = 3
item_meta.loc[item_meta.properties.apply(lambda x: '4 Star' in x), 'star'] = 4
item_meta.loc[item_meta.properties.apply(lambda x: '5 Star' in x), 'star'] = 5
item_meta.loc[(item_meta.star.isna()) & (item_meta.properties.apply(lambda y: 'Excellent Rating' in y) ), 'star'] = 9
item_meta.loc[(item_meta.star.isna()) & (item_meta.properties.apply(lambda y: 'Very Good Rating' in y) ), 'star'] = 8
item_meta.loc[(item_meta.star.isna()) & (item_meta.properties.apply(lambda y: 'Good Rating' in y) ), 'star'] = 7
item_meta.loc[(item_meta.star.isna()) & (item_meta.properties.apply(lambda y: 'Satisfactory Rating' in y) ), 'star'] = 6

item_meta['rating'] = np.nan
item_meta.loc[item_meta.properties.apply(lambda x: 'Satisfactory Rating' in x), 'rating'] = 7.0
item_meta.loc[item_meta.properties.apply(lambda x: 'Good Rating' in x), 'rating'] = 7.5
item_meta.loc[item_meta.properties.apply(lambda x: 'Very Good Rating' in x), 'rating'] = 8.0
item_meta.loc[item_meta.properties.apply(lambda x: 'Excellent Rating' in x), 'rating'] = 8.5

# get binary properties feature
item_properties_df = pd.DataFrame()
item_properties_df['item_id'] = item_meta.item_id
item_properties_df['num_properties'] = item_meta.properties.apply(len)
item_properties_df['star'] = item_meta.star
item_properties_df['item_Beach'] = item_meta.properties.apply(lambda x: 'Beach' in x).astype(np.float16)
item_properties_df['item_Bed & Breakfast'] = item_meta.properties.apply(lambda x: 'Bed & Breakfast' in x).astype(np.float16)
item_properties_df['rating'] = item_meta['rating']


item_star_map = item_properties_df.loc[:,['item_id','star']].set_index('item_id').to_dict()['star']
item_rating_map = item_properties_df.loc[:,['item_id','rating']].set_index('item_id').to_dict()['rating']



del  item_meta
gc.collect()

# ignore filter_df , not using, consume huge memory yet increase a little
filter_df = data.loc[ ~data.current_filters.isna(), ['id', 'current_filters']]
filter_df['current_filters'] = filter_df.current_filters.apply(lambda x:x.split('|'))

# filter_df.loc[filter_df.current_filters.apply(lambda x: '3 Star' in x), 'nights'] = 3
filter_df['nights']=np.nan
filter_df.loc[filter_df.current_filters.apply(lambda x: '2 Nights' in x), 'nights'] = 1
filter_df.loc[filter_df.current_filters.apply(lambda x: '3 Nights' in x), 'nights'] = 2

filter_set = list(set(np.hstack(filter_df['current_filters'].to_list())))

cat_encoders['filters'] = CategoricalEncoder()
cat_encoders['filters'].fit(filter_set)

# get binary filter feature
filters_df = pd.DataFrame()
filters_df['id'] = filter_df.id
filters_df['num_filters'] = filter_df.current_filters.apply(len)
filters_df['breakfast_included'] = filter_df.current_filters.apply( lambda x: 'Breakfast Included' in x).astype(np.float16)
filters_df['filters_Sort By Price'] = filter_df.current_filters.apply( lambda x: 'Sort by Price' in x).astype(np.float16)
filters_df['filters_Sort By Popularity'] = filter_df.current_filters.apply( lambda x: 'Sort By Popularity' in x).astype(np.float16)



# compute interaction image count for each item across train/ test
interaction_image_item_ids = train.loc[train.action == transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_image_items, counts = np.unique(interaction_image_item_ids, return_counts=True)
global_image_count_dict = dict(zip(unique_interaction_image_items, counts))  

# compute interaction count for each item across train/ test
interaction_item_ids = train.loc[train.action != transformed_clickout_action, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action != transformed_clickout_action, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_items, counts = np.unique(interaction_item_ids, return_counts=True)
global_interaction_count_dict = dict(zip(unique_interaction_items, counts))  

# compute interaction deals count for each item across train/ test
interaction_deals_item_ids = train.loc[train.action == transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_deals_items, counts = np.unique(interaction_deals_item_ids, return_counts=True)
global_deals_count_dict = dict(zip(unique_interaction_deals_items, counts))


# compute step rank to identify the last row in each session for train/ val split
train = train.loc[train.action == transformed_clickout_action,:]
test = test.loc[test.action == transformed_clickout_action,:]
train['step_rank'] = train.groupby('session_id')['step'].rank(method='max', ascending=False)

# compute the impression count for each item
item_ids = np.hstack([np.hstack(train['impressions'].values), np.hstack(test.impressions.values)])
unique_items, counts = np.unique(item_ids, return_counts=True)
impression_count_dict = dict(zip(unique_items, counts))

# compute the rank gauss transformed prices
unique_prices = np.unique(np.hstack([np.hstack(train.prices.values), np.hstack(test.prices.values)]) )
rg_unique_prices = GaussRankScaler().fit_transform(unique_prices)
price_rg_price_dict = dict(zip(unique_prices, rg_unique_prices))


#train/ val split
if configuration.debug:
    val = train.loc[train.step_rank == 1,:].iloc[:5]
else:
    val = train.loc[train.step_rank == 1,:].iloc[:50000]

val_index = val.index
train = train.loc[~train.index.isin(val_index),:]

train = train.drop('step_rank', axis=1)
val = val.drop('step_rank', axis=1)


# get the encoded nan item
transformed_nan_item = cat_encoders['item_id'].transform(['nan'])[0]




from collections import defaultdict, Counter
session_clickout_count_dict = {}
past_interaction_dict = {}
last_click_sess_dict = {}
last_impressions_dict = {}
sess_last_imp_idx_dict={}
sess_last_price_dict  = {}
sess_time_diff_dict ={}
sess_step_diff_dict = {}

cumulative_click_dict = defaultdict(lambda : 0)




def parse_impressions(df, session_interactions, session_actions, session_time_diff, training=True):
    # parse the data into a binary classification task, generate 1 example for each item in the impression list
    df_list = []
    label_test_df_list = []
    # parse impressions for train set
    past_interaction_rows = []
    past_interaction_columns = []
    for idx, row in enumerate(tqdm(df.itertuples())):

        if row.session_id not in session_clickout_count_dict:
            session_clickout_count_dict[row.session_id] = 0

        if row.user_id not in past_interaction_dict:
            past_interaction_dict[row.user_id] = []
        
        
        sess_step = row.sess_step
        session_id = row.session_id

        # compute the categorically encoded impression list
        transformed_impressions = cat_encoders['item_id'].transform(row.impressions, to_np=True)

        current_rows = np.zeros([len(row.impressions), 66], dtype=object)

        # compute rank of price this clickout
        price_rank = compute_rank(row.prices)

        #compute the number of interactions associated with the last interacted item in this session
        equal_last_item_indices = np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1]) == row.last_item
        last_item_interaction = len(set(np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1])[equal_last_item_indices]))

        #compute the local interaction count for each item id
        interaction_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) != transformed_clickout_action
        interaction_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_indices]
        sess_unique_items, counts = np.unique(interaction_item, return_counts=True)
        interaction_count_dict = dict(zip(sess_unique_items, counts))

        #compute the local interaction image count for each item id
        interaction_image_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_image
        interaction_image_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_image_indices]
        sess_unique_image_items, counts = np.unique(interaction_image_item, return_counts=True)
        interaction_image_count_dict = dict(zip(sess_unique_image_items, counts))

        #compute the local interaction deals count for each item id
        interaction_deals_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_deals
        interaction_deals_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_deals_indices]
        sess_unique_deals_items, counts = np.unique(interaction_deals_item, return_counts=True)
        interaction_deals_count_dict = dict(zip(sess_unique_deals_items, counts))

        #compute the local clickout count for each item id
        interaction_clickout_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_clickout_action
        interaction_clickout_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_clickout_indices]
        sess_unique_clickout_items, counts = np.unique(interaction_clickout_item, return_counts=True)
        interaction_clickout_count_dict = dict(zip(sess_unique_clickout_items, counts))

        #compute the local interaction rating count for each item id
        interaction_rating_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_rating
        interaction_rating_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_rating_indices]
        sess_unique_rating_items, counts = np.unique(interaction_rating_item, return_counts=True)
        interaction_rating_count_dict = dict(zip(sess_unique_rating_items, counts))

        
        # get the time diffference array in this session for later computing the average of it
        finite_time_diff_indices = np.isfinite(session_time_diff[session_id][:sess_step -1])
        finite_time_diff_array = np.array(session_time_diff[session_id][:sess_step -1])[finite_time_diff_indices]

        # unpad the interactions
        unpad_interactions = session_interactions[session_id][configuration.sess_length:configuration.sess_length+ sess_step -1]
        unique_interaction = pd.unique(session_interactions[session_id][:configuration.sess_length+ sess_step -1])
        
        # time elapse of within two steps for each item before the clickout
        item_time_elapse_dict = {}
        for it, elapse in zip(unpad_interactions[:-1], session_time_diff[session_id][1:sess_step -1]):
            if it not in item_time_elapse_dict: 
                item_time_elapse_dict[it] = [elapse]
                
            else:
                item_time_elapse_dict[it].append(elapse)

        # compute time_diff for each item in the session
        interact_diff = [unpad_interactions[::-1].index(imp) if imp in unpad_interactions else np.nan for imp in transformed_impressions]
        item_time_diff =  np.array([ sum(session_time_diff[session_id][sess_step - diff -1 :sess_step]) if np.isfinite(diff) else np.nan for diff in interact_diff])

        target_index = transformed_impressions.tolist().index(row.item_id) if training else np.nan

        #(imp len, num items)        
        current_co_occ = co_occ_matrix[transformed_impressions,:]

        
        #(imp len, num unique items in the session b4 this clickout)
        current_co_occ = current_co_occ[:,sess_unique_items].toarray()

        # (1, num unique items in the session b4 this clickout)
        # print(current_co_occ.dtype)

        norm =  (1 + co_occ_matrix_csc[:, sess_unique_items].sum(axis=0).reshape(-1))

        # #(imp len, num items)        
        # imp_current_co_occ = imp_co_occ_matrix[transformed_impressions,:]

        
        # #(imp len, num unique items in the session b4 this clickout)
        # imp_current_co_occ = imp_current_co_occ[:,sess_unique_items].toarray()

        # # (1, num unique items in the session b4 this clickout)
        # # print(current_co_occ.dtype)

        # imp_norm =  (1 + imp_co_occ_matrix_csc[:, sess_unique_items].sum(axis=0).reshape(-1))

        # norm_imp_current_co_occ = imp_current_co_occ / imp_norm

        # the position of the last interacted item in the current impression list
        if row.last_item in transformed_impressions:
            last_interact_index = transformed_impressions.tolist().index(row.last_item)
        else:
            last_interact_index = np.nan

        # the position of the second last interacted item in the current impression list
        if row.second_last_item in transformed_impressions:
            second_last_interact_index = transformed_impressions.tolist().index(row.second_last_item)
        else:
            second_last_interact_index = np.nan

        # the position of the third last interacted item in the current impression list
        if row.third_last_item in transformed_impressions:
            third_last_interact_index = transformed_impressions.tolist().index(row.third_last_item)
        else:
            third_last_interact_index = np.nan

        # initialize dictionaries
        if row.session_id not in last_click_sess_dict:
            last_click_sess_dict[row.session_id] = transformed_dummy_item

        if row.session_id not in last_impressions_dict:
            last_impressions_dict[row.session_id] = None

        if row.session_id not in sess_last_imp_idx_dict:
            sess_last_imp_idx_dict[row.session_id] = DUMMY_IMPRESSION_INDEX

        if row.session_id not in sess_last_price_dict:
            sess_last_price_dict[row.session_id] = None
        
        if row.session_id not in sess_time_diff_dict:
            sess_time_diff_dict[row.session_id] = None
        
        if row.session_id not in sess_step_diff_dict:
            sess_step_diff_dict[row.session_id] = None

        
        # item id
        current_rows[:, 0] = transformed_impressions
        
        # label
        current_rows[:, 1] = transformed_impressions == row.item_id
        current_rows[:, 2] = row.session_id
        
        # whether current item id equal to the last interacted item id
        current_rows[:, 3] = transformed_impressions == row.last_item 
        current_rows[:, 4] = price_rank
        current_rows[:, 5] = row.platform
        current_rows[:, 6] = row.device
        current_rows[:, 7] = row.city
        current_rows[:, 8] = row.prices
        current_rows[:, 9] = row.country
        
        # impression index
        current_rows[:, 10] = np.arange(len(row.impressions))
        current_rows[:, 11] = row.step
        current_rows[:, 12] = row.id
        
        # last_click_item: last clickout item id
        current_rows[:, 13] = last_click_sess_dict[row.session_id]
        
        # equal_last_impressions: current impression list is eactly the same as the last one that the user encountered 
        current_rows[:, 14] = last_impressions_dict[row.session_id] == transformed_impressions.tolist() 

         
        current_rows[:, 15] = sess_last_imp_idx_dict[row.session_id]
        # last_interact_index
        current_rows[:, 16] = last_interact_index

        # price_diff
        current_rows[:, 17] = row.prices - sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np.nan

        # last_price
        current_rows[:, 18] = sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np.nan

        # price_ratio
        current_rows[:, 19] = row.prices / sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np. nan

        # clickout_time_diff
        current_rows[:, 20] = row.timestamp - sess_time_diff_dict[row.session_id] if sess_time_diff_dict[row.session_id] else np.nan

        # country_platform
        current_rows[:, 21] = row.country_platform

        # impression_count
        current_rows[:, 22] = [impression_count_dict[imp] for imp in row.impressions]
        
        # is_interacted: if that item has been interaced in the current session
        current_rows[:, 23] = [imp in session_interactions[session_id][:configuration.sess_length+ sess_step -1] for imp in transformed_impressions]
        
        # local_interaction_image_count
        current_rows[:, 24] = [interaction_image_count_dict[imp] if imp in interaction_image_count_dict else 0 for imp in transformed_impressions] 
        # local_interaction_deals_count
        current_rows[:, 25] = [interaction_deals_count_dict[imp] if imp in interaction_deals_count_dict else 0 for imp in transformed_impressions] 

        # local_interaction_clickout_count
        current_rows[:, 26] = [interaction_clickout_count_dict[imp] if imp in interaction_clickout_count_dict else 0 for imp in transformed_impressions] 

        # global_interaction_image_count
        current_rows[:, 27] = [global_image_count_dict[imp] if imp in global_image_count_dict else 0 for imp in transformed_impressions] 

        # global_interaction_deals_count
        current_rows[:, 28] = [global_deals_count_dict[imp] if imp in global_deals_count_dict else 0 for imp in transformed_impressions] 

        # is_clicked
        current_rows[:, 29] = [imp in past_interaction_dict[row.user_id] for imp in transformed_impressions]

        # click_diff
        current_rows[:, 30] = [past_interaction_dict[row.user_id][::-1].index(imp) if imp in past_interaction_dict[row.user_id] else np.nan for imp in transformed_impressions]

        # average of the previous features
        for i in range(31, 38):
            current_rows[:, i]  = np.mean(current_rows[:, i-8])

        # impression_avg_prices
        current_rows[:, 38] = np.mean(row.prices)
        current_rows[:, 39] = row.device_platform

        # equal_max_liic: euqal the maximum of local interaction image count
        current_rows[:, 40] = np.array(current_rows[:, 24]) == np.max(current_rows[:, 24]) if sum(current_rows[:, 24]) >0 else False

        # num_interacted_items
        current_rows[:, 41] = len(np.unique(session_interactions[session_id][:configuration.sess_length+ sess_step -1]))

        # equal_second_last_item
        current_rows[:, 42] = transformed_impressions == row.second_last_item 

        # last_action
        current_rows[:, 43] = session_actions[session_id][configuration.sess_length+ sess_step -2]

        # last_second_last_imp_idx_diff
        current_rows[:, 44] = last_interact_index - second_last_interact_index

        # predicted_next_imp_idx (the idea is to trace your eyeball, last_interact_index + (last_interact_index - second_last_interact_index))
        current_rows[:, 45] = 2 * last_interact_index - second_last_interact_index

        # list_len
        current_rows[:, 46] = len(row.impressions)
        
        # imp_idx_velocity
        current_rows[:, 47] = last_interact_index - 2 * second_last_interact_index + third_last_interact_index

        # time_diff_sess_avg
        current_rows[:, 48] = np.mean(finite_time_diff_array)
        
        # max_time_elapse
        current_rows[:, 49] = [ max(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]

        # sum_time_elapse
        current_rows[:, 50] = [ sum(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]

        # avg_time_elapse
        current_rows[:, 51] = [ np.mean(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]

        # item_time_diff  
        current_rows[:, 52] = item_time_diff

        # global_interaction_count
        current_rows[:, 53] = [global_interaction_count_dict[imp] if imp in global_interaction_count_dict else 0 for imp in transformed_impressions] 

        # average global_interaction_count
        current_rows[:, 54] = np.mean(current_rows[:, 53])

        # std of global interaction image count
        current_rows[:, 55] = np.std(current_rows[:, 27])
        
        # std of glocal interaction conut
        current_rows[:, 56] = np.std(current_rows[:, 53])

        # local_interaction_count
        current_rows[:, 57] = [interaction_count_dict[imp] if imp in interaction_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 58] = target_index

        # target price
        current_rows[:, 59] = row.prices[target_index] if not np.isnan(target_index) else np.nan

        # normalized co-occurence statistics
        current_rows[:, 60] = np.mean(current_co_occ/ norm, axis=1).reshape(-1)
        current_rows[:, 61] = np.min(current_co_occ/ norm, axis=1).reshape(-1)
        current_rows[:, 62] = np.max(current_co_occ/norm, axis=1).reshape(-1)
        current_rows[:, 63] = np.median(current_co_occ/norm, axis=1).reshape(-1)

        # last_item_interaction
        current_rows[:, 64] = last_item_interaction

        # target price rank
        current_rows[:, 65] = price_rank[target_index] if not np.isnan(target_index) else np.nan
        # current_rows[:, 66] = np.mean(norm_imp_current_co_occ, axis=1).reshape(-1)
        # current_rows[:, 67] = np.min(norm_imp_current_co_occ, axis=1).reshape(-1)
        # current_rows[:, 68] = np.max(norm_imp_current_co_occ, axis=1).reshape(-1)
        # current_rows[:, 69] = np.median(norm_imp_current_co_occ, axis=1).reshape(-1)
        

        
        
        
        if training or  row.item_id == transformed_nan_item:
            df_list.append(current_rows)
        else:
            label_test_df_list.append(current_rows) 
        # cumulative_click_dict[row.item_id] += 1
        past_interaction_dict[row.user_id].append(row.item_id)
        last_click_sess_dict[row.session_id] = row.item_id
        last_impressions_dict[row.session_id] = transformed_impressions.tolist()
        sess_time_diff_dict[row.session_id] = row.timestamp
        sess_step_diff_dict[row.session_id] = row.step
        if row.item_id != transformed_nan_item:
            sess_last_imp_idx_dict[row.session_id] = (transformed_impressions == row.item_id).tolist().index(True)
            sess_last_price_dict[row.session_id] = np.array(row.prices)[ transformed_impressions == row.item_id ][0]
            # cumulative_click_dict[row.item_id]  += 1
    data = np.vstack(df_list)
    df_columns = ['item_id', 'label', 'session_id', 'equal_last_item', 'price_rank', 'platform', 'device', 'city', 'price', 'country', 'impression_index','step', 'id','last_click_item','equal_last_impressions', 'last_click_impression','last_interact_index','price_diff','last_price','price_ratio','clickout_time_diff','country_platform','impression_count','is_interacted','local_interaction_image_count','local_interaction_deals_count','local_interaction_clickout_count','global_interaction_image_count','global_interaction_deals_count','is_clicked','click_diff', 'avg_is_interacted','avg_liic', 'avg_lidc','avg_licc','avg_giic','avg_gdc','avg_is_clicked','impression_avg_prices','device_platform','equal_max_liic','num_interacted_items','equal_second_last_item','last_action','last_second_last_imp_idx_diff','predicted_next_imp_idx', 'list_len','imp_idx_velocity','time_diff_sess_avg','max_time_elapse','sum_time_elapse','avg_time_elapse','item_time_diff','global_interaction_count','avg_gic','std_giic','std_gic','local_interaction_count','target_index','target_price','co_occ_mean_norm','co_occ_min_norm','co_occ_max_norm','co_occ_median_norm','last_item_interaction','target_price_rank']
    dtype_dict = {"item_id":"int32", "label": "int8", "equal_last_item":"int8", "step":"int16", "price_rank": "int32","impression_index":"int32", "platform":"int32","device":"int32","city":"int32", "id":"int32", "country":"int32", "price":"int16", "last_click_item":"int32", "equal_last_impressions":"int8", 'last_click_impression':'int16', 'last_interact_index':'float32', 'price_diff':'float16','last_price':'float16','price_ratio':'float32','clickout_time_diff':'float16','country_platform':'int32','impression_count':'int32','is_interacted':'int8','local_interaction_image_count':'int32','local_interaction_deals_count':'int32','local_interaction_clickout_count':'int32','global_interaction_image_count':'int32','global_interaction_deals_count':'int32','is_clicked':'int8','click_diff':'float32'\
                , 'avg_is_interacted':'float16' ,'avg_liic':'float16', 'avg_lidc':'float32','avg_licc':'float32','avg_giic':'float32','avg_gdc':'float32','avg_is_clicked':'float32','impression_avg_prices':'float32','device_platform':'int32','equal_max_liic':'int8','num_interacted_items':'int32','equal_second_last_item':'int8','last_action':'int32','last_second_last_imp_idx_diff':'float32', 'predicted_next_imp_idx': 'float32','list_len':'int16','imp_idx_velocity':'float32','time_diff_sess_avg':'float32','max_time_elapse':'float32','sum_time_elapse':'float32','avg_time_elapse':'float32','item_time_diff':'float32','global_interaction_count':'float32','avg_gic':'float32','std_giic':'float32','std_gic':'float32','local_interaction_count':'int32','target_index':'float32','target_price':'float32','co_occ_mean_norm':'float32','co_occ_min_norm':'float32','co_occ_max_norm':'float32','co_occ_median_norm':'float32','last_item_interaction':'int32','target_price_rank':'float32'} 
    df = pd.DataFrame(data, columns=df_columns)
    df = df.astype(dtype=dtype_dict )
    if training:
        return df
    else:
        label_test = np.vstack(label_test_df_list)
        label_test = pd.DataFrame(label_test, columns=df_columns)
        label_test = label_test.astype(dtype= dtype_dict)
        return df, label_test
    



train.sort_values('timestamp',inplace=True)
val.sort_values('timestamp',inplace=True)
test.sort_values('timestamp',inplace=True)

# print("sorted!!")
train = parse_impressions(train, train_session_interactions, train_session_actions, train_session_time_diff)
val = parse_impressions(val, train_session_interactions, train_session_actions, train_session_time_diff)
test, label_test = parse_impressions(test, test_session_interactions, test_session_actions, test_session_time_diff, training=False)

if configuration.use_test:
    train = pd.concat([train, label_test], axis=0)






print("test before merge", test.shape)
train = train.merge(item_properties_df, on="item_id", how="left")
val = val.merge(item_properties_df, on="item_id", how="left")
test = test.merge(item_properties_df, on="item_id", how="left")


print("test ", test.shape)
train = train.merge(filters_df, on='id', how="left")
val = val.merge(filters_df, on='id', how="left")
test = test.merge(filters_df, on='id', how="left")


# print("test ", test.shape)
# print("test before merge data_feature", test.shape)

train = train.merge(data_feature, on='id', how="left")
val = val.merge(data_feature, on='id', how="left")
test = test.merge(data_feature, on='id', how="left")
print("test ", test.shape)

del filters_df, data_feature
del data
gc.collect()

# target encoding
agg_cols = [ 'price_rank', 'city', 'platform', 'device', 'country', 'impression_index','star']
for c in agg_cols:
    gp = train.groupby(c)['label']
    mean = gp.mean()
    train[f'{c}_label_avg'] = train[c].map(mean)
    val[f'{c}_label_avg'] = val[c].map(mean)
    test[f'{c}_label_avg'] = test[c].map(mean)

  





agg_cols = ['city','impression_index', 'platform']
for c in agg_cols:
    gp = train.groupby(c)['price']
    mean = gp.mean()
    train[f'{c}_price_avg'] = train[c].map(mean)
    val[f'{c}_price_avg'] = val[c].map(mean)
    test[f'{c}_price_avg'] = test[c].map(mean)



agg_cols = ['city']
for c in agg_cols:
    gp = train.groupby(c)['rg_time_diff']
    mean = gp.mean()
    train[f'{c}_td_avg'] = train[c].map(mean)
    val[f'{c}_td_avg'] = val[c].map(mean)
    test[f'{c}_td_avg'] = test[c].map(mean)

  

train['rg_price'] = train.price.map(price_rg_price_dict)
val['rg_price'] = val.price.map(price_rg_price_dict)
test['rg_price'] = test.price.map(price_rg_price_dict)



#price cut within city

data = pd.concat([train,val,test], axis=0).reset_index()
data = data.loc[:,['city','price']].drop_duplicates(['city','price'])
data['city_price_bin'] = data.groupby('city').price.apply(lambda x: qcut_safe(x, q = 40).astype(str))
data['city_price_bin'] = data.apply( lambda x: str(x.city) + x.city_price_bin,axis=1)
data['city_price_bin'] = data['city_price_bin'].factorize()[0]


train = train.merge(data,  on=['city','price'], how='left')
val = val.merge(data,  on=['city','price'], how='left')
test = test.merge(data,  on=['city','price'], how='left')

  

print("train", train.shape)
print("val", val.shape)
print("test", test.shape)
# test = test.merge(item_properties_df, on="item_id", how="left")





data_drop_columns= ['label', 'session_id', 'step', 'id']
data_drop_columns+= ['target_index','target_price','target_price_rank']

train_label = train.label
val_label = val.label

# build lgbm dataset
d_train = lgb.Dataset(data=train.drop(data_drop_columns, axis=1), label=train_label, free_raw_data=True, silent=True)
d_val = lgb.Dataset(data=val.drop(data_drop_columns, axis=1), label=val_label, free_raw_data=True, silent=True)





del  train
gc.collect()

# params = {
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'nthread': multiprocessing.cpu_count() // 3 if configuration.sub_sample else 24,
#     'num_leaves': 200,
#     'max_depth':10,
#     'learning_rate': 0.05 if configuration.sub_sample else 0.01 ,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'feature_fraction':0.7,
#     'seed': 0,
#     'verbose': -1,

# }
params = {'objective': 'binary', 
'boosting_type': 'gbdt', 
'colsample_bytree': 0.76, 
'learning_rate': 0.01, 
'nthread': multiprocessing.cpu_count() -1, 
'max_depth': 13, 
'min_child_weight': 33, 
'min_data_in_leaf': 94, 
'num_leaves': 302, 
'seed': 30, 
'verbose': -1
}



clf = lgb.train(
    params=params,
    train_set=d_train,
    num_boost_round=50000,
    valid_sets=[d_train, d_val],
    early_stopping_rounds=200 if configuration.sub_sample else 500,
    verbose_eval=500,
    
)



# evaluation
def evaluate(val_df, clf):
    incorrect_session = {}
    val_df['scores'] = clf.predict(val_df.drop(data_drop_columns, axis=1))

    loss = log_loss(val_df.label.values, val_df.scores.values)
    grouped_val = val_df.groupby('session_id')
    rss_group = {i:[] for i in range(1,26)}
    rss = []
    for session_id, group in grouped_val:

        scores = group.scores
        sorted_arg = np.flip(np.argsort(scores))
        rss.append( group['label'].values[sorted_arg])
        rss_group[len(group)].append(group['label'].values[sorted_arg])
        if group['label'].values[sorted_arg][0] != 1:
            incorrect_session[session_id] = (sorted_arg.values, group['label'].values[sorted_arg])
    mrr = compute_mean_reciprocal_rank(rss)
    mrr_group = {i:(len(rss_group[i]), compute_mean_reciprocal_rank(rss_group[i])) for i in range(1,26)}
    print(mrr_group)
    if not configuration.debug:
        pickle.dump( incorrect_session, open(f'../output/{model_name}_val_incorrect_order.p','wb'))
    return mrr, mrr_group, loss



mrr, mrr_group, val_log_loss = evaluate(val, clf)

print("MRR score: ", mrr)



imp = clf.feature_importance('gain')
fn =clf.feature_name()
imp_df = pd.DataFrame()
imp_df['importance'] = imp
imp_df['name'] = fn
imp_df.sort_values('importance', ascending=False, inplace=True)


print(imp_df.head(20))



del d_train, d_val
gc.collect()


if configuration.debug:
    exit(0)    

predictions = []
session_ids = []

test['score'] = clf.predict(test.drop(data_drop_columns, axis=1))
save_test = test.copy()
save_test['item_id'] = cat_encoders['item_id'].reverse_transform(save_test.item_id.values)
with open(f'../output/{model_name}_test_score.p', 'wb') as f:
    pickle.dump( save_test.loc[:,['score', 'session_id', 'item_id', 'step']],f, protocol=4)
    
grouped_test = test.groupby('session_id')
for session_id, group in grouped_test:
    scores = group['score']
    sorted_arg = np.flip(np.argsort(scores))
    sorted_item_ids = group['item_id'].values[sorted_arg]
    sorted_item_ids = cat_encoders['item_id'].reverse_transform(sorted_item_ids)
    sorted_item_string = ' '.join([str(i) for i in sorted_item_ids])
    predictions.append(sorted_item_string)
    session_ids.append(session_id)
        
prediction_df = pd.DataFrame()
prediction_df['session_id'] = session_ids
prediction_df['item_recommendations'] = predictions

print("pred df shape", prediction_df.shape)
sub_df = pd.read_csv('../input/submission_popular.csv')
sub_df.drop('item_recommendations', axis=1, inplace=True)
sub_df = sub_df.merge(prediction_df, on="session_id")
# sub_df['item_recommendations'] = predictions

sub_df.to_csv(f'../output/{model_name}.csv', index=None)   



