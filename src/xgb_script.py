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
import xgboost as xgb
import scipy
from sklearn.decomposition import TruncatedSVD
import multiprocessing
import slack
from ordered_set import OrderedSet
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
configuration = XGBConfiguration()
client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])
model_name='xgb_gic_lic_wosh_lf350_lr002_v2'

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

    # sort by first timestamp of each session
    
    # sess_timestamp = train.loc[:,['timestamp','session_id']].groupby('session_id').first().timestamp.sort_values().reset_index()
    # sess_timestamp['time_order'] = sess_timestamp.timestamp.rank(method='max')
    # train = train.merge(sess_timestamp.drop('timestamp', axis=1), on='session_id').sort_values(['time_order','id']).drop('time_order', axis=1).reset_index(drop=True)
    
    
    # sess_timestamp = test.loc[:,['timestamp','session_id']].groupby('session_id').first().timestamp.sort_values().reset_index()
    # sess_timestamp['time_order'] = sess_timestamp.timestamp.rank(method='max')
    # test = test.merge(sess_timestamp.drop('timestamp', axis=1), on='session_id').sort_values(['time_order','id']).drop('time_order', axis=1).reset_index(drop=True)
    

    # filter out useless action
    # train.loc[~train.action.isin(['change of sort order','filter selection'])]
    # test.loc[~test.action.isin(['change of sort order','filter selection'])]

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
    train['last_item'] = train_shifted_item_id
    test['last_item'] = test_shifted_item_id

    train_shifted_item_id = [DUMMY_ITEM] *2 + train.item_id.values[:-2].tolist()
    test_shifted_item_id = [DUMMY_ITEM] *2  + test.item_id.values[:-2].tolist()

    train['second_last_item'] = train_shifted_item_id
    test['second_last_item'] = test_shifted_item_id

    train_shifted_item_id = [DUMMY_ITEM] *3 + train.item_id.values[:-3].tolist()
    test_shifted_item_id = [DUMMY_ITEM] *3  + test.item_id.values[:-3].tolist()

    train['third_last_item'] = train_shifted_item_id
    test['third_last_item'] = test_shifted_item_id

    # mask out the last interacted item if that interaction comes first in its session
    train['step_rank'] = train.groupby('session_id')['step'].rank(method='max', ascending=True)
    test['step_rank'] = test.groupby('session_id')['step'].rank(method='max', ascending=True)


    train.loc[(train.step_rank == 1) & (train.action == 'clickout item'), 'last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 1) & (test.action == 'clickout item'), 'last_item'] = DUMMY_ITEM

    train.loc[(train.step_rank == 2) & (train.action == 'clickout item'), 'second_last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 2) & (test.action == 'clickout item'), 'second_last_item'] = DUMMY_ITEM

    train.loc[(train.step_rank == 3) & (train.action == 'clickout item'), 'third_last_item'] = DUMMY_ITEM
    test.loc[(test.step_rank == 3) & (test.action == 'clickout item'), 'third_last_item'] = DUMMY_ITEM
    
    # train.sort_values(['timestamp','step'],inplace=True)
    # test.sort_values(['timestamp','step'],inplace=True)  
    
    

    
    
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

train_session_interactions = dict(train.groupby('session_id')['item_id'].apply(list))
test_session_interactions = dict(test.groupby('session_id')['item_id'].apply(list))

# train_user_interactions = dict(train.groupby('user_id')['item_id'].apply(list))
# test_user_interactions = dict(test.groupby('user_id')['item_id'].apply(list))

train_session_actions = dict(train.groupby('session_id')['action'].apply(list))
test_session_actions = dict(test.groupby('session_id')['action'].apply(list))

train['sess_step'] = train.groupby('session_id')['timestamp'].rank(method='max').apply(int)
test['sess_step'] = test.groupby('session_id')['timestamp'].rank(method='max').apply(int)


# train['user_step'] = train.groupby('user_id')['timestamp'].rank(method='first').apply(int)
# test['user_step'] = test.groupby('user_id')['timestamp'].rank(method='first').apply(int)

# print(train.loc[:,['session_id','user_step', 'sess_step']].head())

# train['user_step'] = train.groupby('user_id')['timestamp'].rank(method='max').apply(int)
# test['user_step'] = test.groupby('user_id')['timestamp'].rank(method='max').apply(int)

# with open('../input/lgb_data_feature.p','rb') as f:
#     data_feature  = pickle.load(f)

data_feature = data.loc[:,['id','step','session_id', 'timestamp']].copy()
# first_timestamp = data.groupby('session_id')['timestamp'].first()
# data_feature['first_timestamp'] = data_feature.session_id.map(first_timestamp)
# data_feature['time_elapse'] = data_feature['timestamp'] - data_feature['first_timestamp']
# print(data_feature.loc[:,['first_timestamp','time_elapse','timestamp']].head())
data_feature['time_diff'] = data.groupby('session_id')['timestamp'].diff()
data_feature['time_diff_diff'] = data_feature.groupby('session_id')['time_diff'].diff()
data_feature['time_diff_diff_diff'] = data_feature.groupby('session_id')['time_diff_diff'].diff()
data_feature['time_diff_2'] = data.groupby('session_id')['timestamp'].diff().shift(1)
data_feature['time_diff_3'] = data.groupby('session_id')['timestamp'].diff().shift(2)


# data_feature['time_diff_3'] = data.groupby('session_id')['timestamp'].diff().shift(2)
# data_feature['time_diff_min'] = data_feature.time_diff // 60
data_feature['rg_time_diff'] = GaussRankScaler().fit_transform(data_feature['time_diff'].values)
# data_feature['rg_timestamp'] = GaussRankScaler().fit_transform(data_feature['timestamp'].values)
data_feature['step_log'] = np.log1p(data_feature['step'])


data_feature = data_feature.drop(['session_id','step','timestamp'], axis=1)


# with open('../input/lgb_data_feature.p','wb') as f:
#     pickle.dump(data_feature, f, protocol=4)    
    
# get time diff    
train = train.merge(data_feature, on='id', how='left')
test = test.merge(data_feature, on='id', how='left')

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


# d2v = pickle.load(open('../input/dict_doc2vec_property_sort.p','rb')).rename(columns={'row_id':'id'})
# d2v['item_id'] = d2v['item_id'].apply(str)
# d2v = d2v.loc[d2v.item_id.isin(all_items)]

# d2v['item_id'] = cat_encoders['item_id'].transform(d2v['item_id'].values)
# fm_file_name = '../input/fm_item_embedding_d32_e30_warp_nocc.p'
# # fm_file_name = '../input/ncf_xnn_int_diff_v2_140k_0_ie.p'

# with open(fm_file_name,'rb') as f:
#     fm_df = pickle.load(f)


# fm_df = fm_df.loc[fm_df.item_id.isin(unique_items)]

# # # # fm_df['item_id'] = cat_
# print(fm_file_name)

for col in all_cat_columns:
    train[col] = cat_encoders[col].transform(train[col].values)
    test[col] = cat_encoders[col].transform(test[col].values)

# fm_df['item_id'] = cat_encoders['item_id'].transform(fm_df.item_id.values)
# fm_df_ids = fm_df.item_id.values
# fm_df_embedding = fm_df.drop('item_id',axis=1).values

# fm_dict = { item_id: fm_df_embedding[idx] for idx, item_id in enumerate(fm_df.item_id)}
# print(fm_dict)

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
    
# for user_id, item_list in train_user_interactions.items():
#     train_user_interactions[user_id] = [transformed_dummy_item] * configuration.sess_length + cat_encoders['item_id'].transform(item_list)

# for user_id, item_list in test_user_interactions.items():
#     test_user_interactions[user_id] = [transformed_dummy_item] * configuration.sess_length + cat_encoders['item_id'].transform(item_list)   
train['last_item'] = cat_encoders['item_id'].transform(train['last_item'].values)
test['last_item'] = cat_encoders['item_id'].transform(test['last_item'].values)

train['second_last_item'] = cat_encoders['item_id'].transform(train.second_last_item.values)
test['second_last_item'] = cat_encoders['item_id'].transform(test.second_last_item.values)

train['third_last_item'] = cat_encoders['item_id'].transform(train.third_last_item.values)
test['third_last_item'] = cat_encoders['item_id'].transform(test.third_last_item.values)

# compute step_rank for train/ val split, use put the last clickout in the session in the validation set


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


item_meta['rating'] = np.nan
item_meta.loc[item_meta.properties.apply(lambda x: 'Satisfactory Rating' in x), 'rating'] = 7.0
item_meta.loc[item_meta.properties.apply(lambda x: 'Good Rating' in x), 'rating'] = 7.5
item_meta.loc[item_meta.properties.apply(lambda x: 'Very Good Rating' in x), 'rating'] = 8.0
item_meta.loc[item_meta.properties.apply(lambda x: 'Excellent Rating' in x), 'rating'] = 8.5



# item_meta['nights'] = 1
# item_meta.loc[item_meta.properties.apply(lambda x: '2 Nights' in x), 'nights'] = 2
# item_meta.loc[item_meta.properties.apply(lambda x: '3 Nights' in x), 'nights'] = 3



# unique_property = list(set(np.hstack(item_meta.properties.tolist())))

# cat_encoders['item_property'] = CategoricalEncoder()
# cat_encoders['item_property'].fit(unique_property)
# all_item_list = []
# for row in item_meta.itertuples():
#     current_row = np.zeros(len(unique_property) + 1)
#     one_indices = cat_encoders['item_property'].transform(row.properties)
#     current_row[one_indices] = 1
#     current_row[-1] = row.item_id
#     all_item_list.append(current_row)

# item_properties_array = np.vstack(all_item_list)

# item_properties_df = pd.DataFrame(item_properties_array, columns=unique_property + ['item_id'])
# item_properties_df = item_properties_df.astype(dtype= {"item_id":"int32"})
item_properties_df = pd.DataFrame()
item_properties_df['item_id'] = item_meta.item_id
item_properties_df['num_properties'] = item_meta.properties.apply(len)
item_properties_df['star'] = item_meta.star
item_properties_df['item_Beach'] = item_meta.properties.apply(lambda x: 'Beach' in x).astype(np.float16)
item_properties_df['item_Bed & Breakfast'] = item_meta.properties.apply(lambda x: 'Bed & Breakfast' in x).astype(np.float16)
item_properties_df['rating'] = item_meta['rating']


item_star_map = item_properties_df.loc[:,['item_id','star']].set_index('item_id').to_dict()['star']
# item_properties_df['nights'] = item_meta.nights
# item_properties_df['item_Hostel'] = item_meta.properties.apply(lambda x: 'Hostel' in x).astype(np.float16)
# item_properties_df['item_Pet Friendly'] = item_meta.properties.apply(lambda x: 'Pet Friendly' in x).astype(np.float16)
# item_properties_df['Business Hotel'] = item_meta.properties.apply(lambda x: 'Business Hotel' in x).astype(np.float16)


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
all_filter_array = []

for row in filter_df.itertuples():
    current_row = np.zeros(len(filter_set) + 1, dtype=object)
    current_filters = row.current_filters
    one_indices = cat_encoders['filters'].transform(row.current_filters)
    current_row[one_indices] = 1
    current_row[-1] = row.id
    all_filter_array.append(current_row)
    
all_filter_array = np.vstack(all_filter_array)
# filters_df = pd.DataFrame(all_filter_array, columns=  [f'ft_{f}' for f in filter_set] + [ 'id'])
# dtype_dict = {"id":"int32"}
# for f in filter_set:
#     dtype_dict[f'ft_{f}'] = "int32"
# filters_df = filters_df.astype(dtype= dtype_dict)
filters_df = pd.DataFrame()
filters_df['id'] = filter_df.id
filters_df['num_filters'] = filter_df.current_filters.apply(len)
filters_df['breakfast_included'] = filter_df.current_filters.apply( lambda x: 'Breakfast Included' in x).astype(np.float16)
filters_df['filters_Sort By Price'] = filter_df.current_filters.apply( lambda x: 'Sort by Price' in x).astype(np.float16)
filters_df['filters_Sort By Popularity'] = filter_df.current_filters.apply( lambda x: 'Sort By Popularity' in x).astype(np.float16)

# filters_df['filters_Swimming Pool (Combined Filter)'] = filter_df.current_filters.apply( lambda x: 'Swimming Pool (Combined Filter)' in x).astype(np.float16)


interaction_image_item_ids = train.loc[train.action == transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_image_items, counts = np.unique(interaction_image_item_ids, return_counts=True)
global_image_count_dict = dict(zip(unique_interaction_image_items, counts))  

# interaction_image_item_ids = train.loc[train.action == transformed_interaction_image, :].item_id.tolist() + test.loc[test.action == transformed_interaction_image, :].item_id.tolist()
# unique_interaction_image_items, counts = np.unique(interaction_image_item_ids, return_counts=True)
# global_image_count_dup_dict = dict(zip(unique_interaction_image_items, counts))  


interaction_item_ids = train.loc[train.action != transformed_clickout_action, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action != transformed_clickout_action, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_items, counts = np.unique(interaction_item_ids, return_counts=True)
global_interaction_count_dict = dict(zip(unique_interaction_items, counts))  

interaction_deals_item_ids = train.loc[train.action == transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
unique_interaction_deals_items, counts = np.unique(interaction_deals_item_ids, return_counts=True)
global_deals_count_dict = dict(zip(unique_interaction_deals_items, counts))

# interaction_deals_item_ids = train.loc[train.action == transformed_interaction_deals, :].item_id.tolist() + test.loc[test.action == transformed_interaction_deals, :].item_id.tolist()
# unique_interaction_deals_items, counts = np.unique(interaction_deals_item_ids, return_counts=True)
# global_deals_count_dup_dict = dict(zip(unique_interaction_deals_items, counts))

# interaction_info_item_ids = train.loc[train.action == transformed_interaction_info, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_info, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
# unique_interaction_info_items, counts = np.unique(interaction_info_item_ids, return_counts=True)
# global_info_count_dict = dict(zip(unique_interaction_info_items, counts))  

# interaction_rating_item_ids = train.loc[train.action == transformed_interaction_rating, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == transformed_interaction_rating, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
# unique_interaction_rating_items, counts = np.unique(interaction_rating_item_ids, return_counts=True)
# global_rating_count_dict = dict(zip(unique_interaction_rating_items, counts))


#global_interaction_count

# filter actions
train = train.loc[train.action == transformed_clickout_action,:]
test = test.loc[test.action == transformed_clickout_action,:]
train['step_rank'] = train.groupby('session_id')['step'].rank(method='max', ascending=False)

# occurrence in impression count
item_ids = np.hstack([np.hstack(train['impressions'].values), np.hstack(test.impressions.values)])
unique_items, counts = np.unique(item_ids, return_counts=True)
impression_count_dict = dict(zip(unique_items, counts))



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

# clickout count
# clickout_item_ids = train.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
# unique_clickout_items, counts = np.unique(clickout_item_ids, return_counts=True)
# clickout_count_dict = dict(zip(unique_clickout_items, counts))  

# clickout_count_df = pd.DataFrame()
# clickout_count_df['clickout_count'] = counts
# clickout_count_df['item']
# clickout_count_df['quantile'] = pd.qcut(clickout_count_df['clickout_count'], 30, duplicates='drop')
# clickout_count_df = clickout_count_df.sort_values('quantile') 
# clickout_count_df['quantile'] = clickout_count_df['quantile'].factorize()[0]
# print(clickout_count_df.loc[clickout_count_df['clickout_count'] == clickout_count_df.clickout_count.min()].head())
# clickout_quantile_dict = clickout_count_df.loc[:,['item_id','quantile']].set_index('item_id').to_dict()['quantile']
# clickout_count_df = train.item_id.value_counts().reset_index().rename(columns={'item_id':'clickout_count','index':'item_id'})

# item_ids = cat_encoders['item_id'].transform(np.hstack(train['impressions'].values))
# unique_items, counts = np.unique(item_ids, return_counts=True)
# impression_count_df = pd.DataFrame()
# impression_count_df['item_id'] = unique_items
# impression_count_df['impressions_count'] = counts


# clickout_count_df = pd.DataFrame()
# clickout_count_df['item_id'] = unique_clickout_items
# clickout_count_df['clickout_count'] = counts

# # filter item clickout less than 10 times






# click through rate
# ctr_df = impression_count_df.merge(clickout_count_df, how='left',on='item_id').fillna(0)
# ctr_df['ctr'] = ctr_df.clickout_count / ctr_df.impressions_count
# ctr_dict = ctr_df.loc[:,['item_id','ctr']].to_dict()['ctr']

# print(ctr_dict)
# clickout_item_ids = train.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() #+ test.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() 
# unique_clickout_items, counts = np.unique(clickout_item_ids, return_counts=True)
# clickout_count_dict = dict(zip(unique_clickout_items, counts))  


# compute the nan item later used to distinguish labled or unlabeld test data
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
# user_click_dict = {}



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

        transformed_impressions = cat_encoders['item_id'].transform(row.impressions, to_np=True)
        current_rows = np.zeros([len(row.impressions), 58], dtype=object)
        price_rank = compute_rank(row.prices)

        interaction_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) != transformed_clickout_action
        interaction_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_indices]
        sess_unique_items, counts = np.unique(interaction_item, return_counts=True)
        interaction_count_dict = dict(zip(sess_unique_items, counts))


        interaction_image_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_image
        interaction_image_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_image_indices]
        sess_unique_image_items, counts = np.unique(interaction_image_item, return_counts=True)
        interaction_image_count_dict = dict(zip(sess_unique_image_items, counts))


        interaction_deals_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_deals
        interaction_deals_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_deals_indices]
        sess_unique_deals_items, counts = np.unique(interaction_deals_item, return_counts=True)
        interaction_deals_count_dict = dict(zip(sess_unique_deals_items, counts))


        interaction_clickout_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_clickout_action
        interaction_clickout_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_clickout_indices]
        sess_unique_clickout_items, counts = np.unique(interaction_clickout_item, return_counts=True)
        interaction_clickout_count_dict = dict(zip(sess_unique_clickout_items, counts))

        interaction_rating_indices = np.array(session_actions[session_id][:configuration.sess_length+ sess_step -1]) == transformed_interaction_rating
        interaction_rating_item =  np.array(session_interactions[session_id][:configuration.sess_length+ sess_step -1])[interaction_rating_indices]
        sess_unique_rating_items, counts = np.unique(interaction_rating_item, return_counts=True)
        interaction_rating_count_dict = dict(zip(sess_unique_rating_items, counts))

        # padded_prices =  np.array([ 0] * 2 +  row.prices + [0]*2)
        # padded_image_counts =[interaction_image_count_dict[imp] if imp in interaction_image_count_dict else 0 for imp in transformed_impressions] 
        # global_clickout_count = np.array([clickout_count_dict[imp] if imp in clickout_count_dict else 0 for imp in transformed_impressions])
        # unleaked_clickout_count = [unleaked_clickout_count[idx] -1 if imp == row.item_id else unleaked_clickout_count[idx] for idx, imp in enumerate(transformed_impressions)]

        finite_time_diff_indices = np.isfinite(session_time_diff[session_id][:sess_step -1])
        finite_time_diff_array = np.array(session_time_diff[session_id][:sess_step -1])[finite_time_diff_indices]

        unpad_interactions = session_interactions[session_id][configuration.sess_length:configuration.sess_length+ sess_step -1]
        
        
        unique_interaction = pd.unique(session_interactions[session_id][:configuration.sess_length+ sess_step -1])
        
        # time elapse of within two steps for each item before the clickout
        item_time_elapse_dict = {}

        for it, elapse in zip(unpad_interactions[:-1], session_time_diff[session_id][1:sess_step -1]):
            if it not in item_time_elapse_dict: #or elapse > item_time_elapse_dict[it]:

                item_time_elapse_dict[it] = [elapse]
                
            else:
                item_time_elapse_dict[it].append(elapse)

        # compute time_diff for each item in the session
        interact_diff = [unpad_interactions[::-1].index(imp) if imp in unpad_interactions else np.nan for imp in transformed_impressions]
        item_time_diff =  np.array([ sum(session_time_diff[session_id][sess_step - diff -1 :sess_step]) if np.isfinite(diff) else np.nan for diff in interact_diff])


        # last_star = item_star_map[row.last_item] if row.last_item in item_star_map else np.nan

        if row.last_item in transformed_impressions:
            last_interact_index = transformed_impressions.tolist().index(row.last_item)
        else:
            last_interact_index = np.nan

        if row.second_last_item in transformed_impressions:
            second_last_interact_index = transformed_impressions.tolist().index(row.second_last_item)
        else:
            second_last_interact_index = np.nan

        if row.third_last_item in transformed_impressions:
            third_last_interact_index = transformed_impressions.tolist().index(row.third_last_item)
        else:
            third_last_interact_index = np.nan

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

        # cumulative_click_sum = sum(cumulative_click_dict.values()) + 1.0
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
        # last clickout item id
        current_rows[:, 13] = last_click_sess_dict[row.session_id]
        # 
        current_rows[:, 14] = last_impressions_dict[row.session_id] == transformed_impressions.tolist() 
        current_rows[:, 15] = sess_last_imp_idx_dict[row.session_id]
        current_rows[:, 16] = last_interact_index
        current_rows[:, 17] = row.prices - sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np.nan
        current_rows[:, 18] = sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np.nan
        current_rows[:, 19] = row.prices / sess_last_price_dict[row.session_id] if sess_last_price_dict[row.session_id] else np. nan
        current_rows[:, 20] = row.timestamp - sess_time_diff_dict[row.session_id] if sess_time_diff_dict[row.session_id] else np.nan
        current_rows[:, 21] = row.country_platform
        current_rows[:, 22] = [impression_count_dict[imp] for imp in row.impressions]
        # print(session_interactions.keys())
        # if that item has been interaced in the current session
        current_rows[:, 23] = [imp in session_interactions[session_id][:configuration.sess_length+ sess_step -1] for imp in transformed_impressions]
        current_rows[:, 24] = [interaction_image_count_dict[imp] if imp in interaction_image_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 25] = [interaction_deals_count_dict[imp] if imp in interaction_deals_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 26] = [interaction_clickout_count_dict[imp] if imp in interaction_clickout_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 27] = [global_image_count_dict[imp] if imp in global_image_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 28] = [global_deals_count_dict[imp] if imp in global_deals_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 29] = [imp in past_interaction_dict[row.user_id] for imp in transformed_impressions]
        current_rows[:, 30] = [past_interaction_dict[row.user_id][::-1].index(imp) if imp in past_interaction_dict[row.user_id] else np.nan for imp in transformed_impressions]

        for i in range(31, 38):
            current_rows[:, i]  = np.mean(current_rows[:, i-8])

        current_rows[:, 38] = np.mean(row.prices)
        current_rows[:, 39] = row.device_platform
        current_rows[:, 40] = np.array(current_rows[:, 24]) == np.max(current_rows[:, 24]) if sum(current_rows[:, 24]) >0 else False
        current_rows[:, 41] = len(np.unique(session_interactions[session_id][:configuration.sess_length+ sess_step -1]))
        current_rows[:, 42] = transformed_impressions == row.second_last_item 
        current_rows[:, 43] = session_actions[session_id][configuration.sess_length+ sess_step -2]
        current_rows[:, 44] = last_interact_index - second_last_interact_index
        current_rows[:, 45] = 2 * last_interact_index - second_last_interact_index
        current_rows[:, 46] = len(row.impressions)
        
        current_rows[:, 47] = last_interact_index - 2 * second_last_interact_index + third_last_interact_index
        current_rows[:, 48] = np.mean(finite_time_diff_array)
        current_rows[:, 49] = [ max(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]
        current_rows[:, 50] = [ sum(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]
        current_rows[:, 51] = [ np.mean(item_time_elapse_dict[imp]) if imp in item_time_elapse_dict else np.nan for imp in transformed_impressions]
        current_rows[:, 52] = item_time_diff
        current_rows[:, 53] = [global_interaction_count_dict[imp] if imp in global_interaction_count_dict else 0 for imp in transformed_impressions] 
        current_rows[:, 54] = np.mean(current_rows[:, 53])
        current_rows[:, 55] = np.std(current_rows[:, 27])
        current_rows[:, 56] = np.std(current_rows[:, 53])
        
        # current_rows[:, 57] = current_rows[:, 53].tolist()[:-1] + [np.nan]
        # current_rows[:, 58] = [np.nan] + current_rows[:, 53].tolist()[1:]
        current_rows[:, 57] = [interaction_count_dict[imp] if imp in interaction_count_dict else 0 for imp in transformed_impressions] 
        
        # current_rows[:, 59] = len(row.impressions) - current_rows[:, 23][::-1].tolist().index(True)  if True in current_rows[:, 23] else np.nan
        
        # current_rows[:, 55] = row.impressions == last_click_sess_dict[row.session_id]
        # current_rows[:, 54] = (last_interact_index - second_last_interact_index) / (1 + session_time_diff[session_id][sess_step -2])
        # current_rows[:, 54] = current_rows[:, 53] - np.mean(current_rows[:, 53])
        # current_rows[:, 53] = np.nanargmax(current_rows[:, 50]) if not np.all(np.isnan(current_rows[:, 50].astype(np.float16))) else False
        # current_rows[:, 53] = [ np.min(item_time_diff[np.arange(len(row.impressions)) != i]) for i in range(len(row.impressions))] if len(row.impressions) > 1 else np.nan
        # current_rows[:, 53] = np.mean(current_rows[:, 49])
        # current_rows[:, 49] = transformed_impressions == row.third_last_item
        # current_rows[:, 49] = (last_interact_index + second_last_interact_index) / 2
        
        # current_rows[:, 49] = len(row.impressions) == 25
        # current_rows[:, 50] = len(row.impressions) < 11

        
        
        
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
    df_columns = ['item_id', 'label', 'session_id', 'equal_last_item', 'price_rank', 'platform', 'device', 'city', 'price', 'country', 'impression_index','step', 'id','last_click_item','equal_last_impressions', 'last_click_impression','last_interact_index','price_diff','last_price','price_ratio','clickout_time_diff','country_platform','impression_count','is_interacted','local_interaction_image_count','local_interaction_deals_count','local_interaction_clickout_count','global_interaction_image_count','global_interaction_deals_count','is_clicked','click_diff', 'avg_is_interacted','avg_liic', 'avg_lidc','avg_licc','avg_giic','avg_gdc','avg_is_clicked','impression_avg_prices','device_platform','equal_max_liic','num_interacted_items','equal_second_last_item','last_action','last_second_last_imp_idx_diff','predicted_next_imp_idx', 'list_len','imp_idx_velocity','time_diff_sess_avg','max_time_elapse','sum_time_elapse','avg_time_elapse','item_time_diff','global_interaction_count','avg_gic','std_giic','std_gic','local_interaction_count']
    dtype_dict = {"item_id":"int32", "label": "int8", "equal_last_item":"int8", "step":"int16", "price_rank": "int32","impression_index":"int32", "platform":"int32","device":"int32","city":"int32", "id":"int32", "country":"int32", "price":"int16", "last_click_item":"int32", "equal_last_impressions":"int8", 'last_click_impression':'int16', 'last_interact_index':'float32', 'price_diff':'float16','last_price':'float16','price_ratio':'float32','clickout_time_diff':'float16','country_platform':'int32','impression_count':'int32','is_interacted':'int8','local_interaction_image_count':'int32','local_interaction_deals_count':'int32','local_interaction_clickout_count':'int32','global_interaction_image_count':'int32','global_interaction_deals_count':'int32','is_clicked':'int8','click_diff':'float32'\
                , 'avg_is_interacted':'float16' ,'avg_liic':'float16', 'avg_lidc':'float32','avg_licc':'float32','avg_giic':'float32','avg_gdc':'float32','avg_is_clicked':'float32','impression_avg_prices':'float32','device_platform':'int32','equal_max_liic':'int8','num_interacted_items':'int32','equal_second_last_item':'int8','last_action':'int32','last_second_last_imp_idx_diff':'float32', 'predicted_next_imp_idx': 'float32','list_len':'int16','imp_idx_velocity':'float32','time_diff_sess_avg':'float32','max_time_elapse':'float32','sum_time_elapse':'float32','avg_time_elapse':'float32','item_time_diff':'float32','global_interaction_count':'float32','avg_gic':'float32','std_giic':'float32','std_gic':'float32','local_interaction_count':'int32'} 
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

# past_interaction_rows = train_past_interaction_rows + val_past_interaction_rows + test_past_interaction_rows
# past_interaction_columns = train_past_interaction_columns + val_past_interaction_columns + test_past_interaction_columns


# form sparse matrix
# data = np.ones(len(train_past_interaction_rows))
# train_interaction_matrix = scipy.sparse.coo_matrix((data, (train_past_interaction_rows, train_past_interaction_columns)), shape=(len(train) , cat_encoders['item_id'].n_elements))

# data = np.ones(len(val_past_interaction_rows))
# val_interaction_matrix = scipy.sparse.coo_matrix((data, (val_past_interaction_rows, val_past_interaction_columns)), shape=(len(val) , cat_encoders['item_id'].n_elements))

# data = np.ones(len(test_past_interaction_rows))
# test_interaction_matrix = scipy.sparse.coo_matrix((data, (test_past_interaction_rows, test_past_interaction_columns)), shape=(len(test) , cat_encoders['item_id'].n_elements))

# svd to learn the latent variable from interaction matrix
# svd = TruncatedSVD(n_components=5, n_iter=5, random_state=42)
# train_svd_matrix = svd.fit_transform(train_interaction_matrix)
# val_svd_matrix = svd.transform(val_interaction_matrix)
# test_svd_matrix = svd.transform(test_interaction_matrix)

# print("explained ratio", svd.explained_variance_ratio_.sum())

# train_svd_matrix = svd_matrix[:len(train),:]
# val_svd_matrix = svd_matrix[len(train):len(train) + len(val),:]
# test_svd_matrix = svd_matrix[len(train) + len(val):,:]

# for i in range(train_svd_matrix.shape[1]):
#     train[f'svd_{i}'] = train_svd_matrix[:,i]
#     val[f'svd_{i}'] = val_svd_matrix[:,i]
#     test[f'svd_{i}'] = test_svd_matrix[:,i]




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


# city_star_mean_price = train.groupby(['city','star'])['price'].mean().reset_index().rename(columns={'price':'city_star_price_avg'})

# train = train.merge(city_star_mean_price, on=['city','star'], how='left')
# val = val.merge(city_star_mean_price, on=['city','star'], how='left')
# test = test.merge(city_star_mean_price, on=['city','star'], how='left')

    # train[f'{c}_price_diff'] = train['price'] - train[f'{c}_price_avg']
    # val[f'{c}_price_diff'] = val['price'] - val[f'{c}_price_avg']
    # test[f'{c}_price_diff'] = test['price'] - test[f'{c}_price_avg']




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

# agg_cols = ['city','impression_index']
# for c in agg_cols:
#     gp = train.groupby(c)['price']
#     skew = gp.skew()
#     train[f'{c}_price_avg'] = train[c].map(skew)
#     val[f'{c}_price_avg'] = val[c].map(skew)
#     test[f'{c}_price_avg'] = test[c].map(skew)

# agg_cols = ['item_id']
# for c in agg_cols:
#     gp = train.groupby(c)['impression_index']
#     mean = gp.mean()
#     train[f'{c}_impression_index_avg'] = train[c].map(mean)
#     val[f'{c}_impression_index_avg'] = val[c].map(mean)
#     test[f'{c}_impression_index_avg'] = test[c].map(mean)    






# nuniq = pd.concat([train, val, test], axis=0).groupby('item_id')['city'].nunique()
# train[f'item_city_count'] = train[c].map(nuniq)
# val[f'item_city_count'] = val[c].map(nuniq)
# test[f'item_city_count'] = test[c].map(nuniq)

#price cut within city

data = pd.concat([train,val,test], axis=0).reset_index()
data = data.loc[:,['city','price']].drop_duplicates(['city','price'])
data['city_price_bin'] = data.groupby('city').price.apply(lambda x: qcut_safe(x, q = 40).astype(str))
data['city_price_bin'] = data.apply( lambda x: str(x.city) + x.city_price_bin,axis=1)
data['city_price_bin'] = data['city_price_bin'].factorize()[0]


train = train.merge(data,  on=['city','price'], how='left')
val = val.merge(data,  on=['city','price'], how='left')
test = test.merge(data,  on=['city','price'], how='left')


# with open(f'../output/{model_name}_train_processed.p','wb') as f:
#     pickle.dump(train, f, protocol=4)

# with open(f'../output/{model_name}_val_processed.p','wb') as f:
#     pickle.dump(val, f, protocol=4)

# with open(f'../output/{model_name}_test_processed.p','wb') as f:
#     pickle.dump(test, f, protocol=4)        


# train = train.merge(d2v, on=['id','item_id'], how='left')
# val = val.merge(d2v, on=['id','item_id'], how='left')
# test = test.merge(d2v, on=['id','item_id'], how='left')

# train = train.merge(fm_df, on='item_id',how='left')
# val = val.merge(fm_df, on='item_id',how='left')
# test = test.merge(fm_df, on='item_id',how='left')


# with open('../input/doc2vec_32_df.p','rb') as f:
#     doc2vec_df = pickle.load(f)

# train = train.merge(doc2vec_df, on='id',how='left')
# val = val.merge(doc2vec_df, on='id',how='left')
# test = test.merge(doc2vec_df, on='id',how='left')

# with open('../input/user_embedding_maxpool_df.p' ,'rb') as f:
#     user_embedding_df = pickle.load(f)

# train = train.merge(user_embedding_df, on='user_id',how='left')
# val = val.merge(user_embedding_df, on='user_id',how='left')
# test = test.merge(user_embedding_df, on='user_id',how='left')    

print("train", train.shape)
print("val", val.shape)
print("test", test.shape)
# test = test.merge(item_properties_df, on="item_id", how="left")





data_drop_columns= ['label', 'session_id', 'step', 'id']
# data_drop_columns+= ['avg_lidc','avg_licc']

train_label = train.label

val_label = val.label


d_train = xgb.DMatrix(data=train.drop(data_drop_columns, axis=1), label=train_label.values, silent=True, nthread=-1, feature_names=train.drop(data_drop_columns, axis=1).columns.tolist())
d_val = xgb.DMatrix(data=val.drop(data_drop_columns, axis=1), label=val_label.values, silent=True, nthread= -1, feature_names=train.drop(data_drop_columns, axis=1).columns.tolist())
d_test = xgb.DMatrix(test.drop(data_drop_columns, axis=1), nthread=-1, feature_names=train.drop(data_drop_columns, axis=1).columns.tolist())

cat_cols = [ 'item_id', "price_rank", 'city', 'platform', 'device', 'country', 'impression_index','star','last_click_impression','last_click_item','last_interact_index','country_platform']

for col in cat_cols:
    if (train[col] < 0).sum() > 0:
        print("contains negative ", col)

del  train
gc.collect()

# params = {
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'nthread': multiprocessing.cpu_count() //2,
#     'num_leaves': 200,
#     'max_depth':10,
#     'learning_rate': 0.05,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'feature_fraction':0.7,
#     'seed': 0,
#     'verbose': -1,

# }

params={
    'eta': 0.02,  # 0.03,
  "booster": "gbtree",
  'tree_method':'hist',
  'max_leaves': 350, 
  'max_depth': 10,  # 18
  "nthread": multiprocessing.cpu_count() -1,
  'subsample': 0.9,
  'colsample_bytree': 0.8,
  'colsample_bylevel': 0.8,
  'min_child_weight': 2,
  'alpha': 1,
  'objective': 'binary:logistic',
  'eval_metric': 'logloss',
  'random_state': 5478,
  'verbosity': 0,
}


watchlist = [ (d_train, 'train'), (d_val, 'valid')]
clf = xgb.train(
    params=params,
    dtrain=d_train,
    num_boost_round=50000, #11927
    evals= watchlist,
    early_stopping_rounds=500,
    verbose_eval=500,
    # categorical_feature= cat_cols
)


# clf.save_model('../weights/lgb-10000-200-01.model')

def evaluate(val_df, clf):
    val_df['scores'] = clf.predict(d_val)
    grouped_val = val_df.groupby('session_id')
    rss = []
    for _, group in grouped_val:

        scores = group.scores
        sorted_arg = np.flip(np.argsort(scores))
        rss.append( group['label'].values[sorted_arg])
        
    mrr = compute_mean_reciprocal_rank(rss)
    return mrr



mrr = evaluate(val, clf)

print("MRR score: ", mrr)



imp = clf.get_score( importance_type='gain')
imp_df = pd.DataFrame.from_dict(imp, orient='index').reset_index()

imp_df.columns=['name','importance']
imp_df.sort_values('importance', ascending=False, inplace=True)



print(imp_df.head(20))


# del d_train
# gc.collect()

if configuration.slack:
    response = client.chat_postMessage(
    channel='#recsys2019',
    blocks=[

    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*configuration* :\n" + str(configuration.get_attributes())

        },

    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*Features* :\n```{test.columns.tolist()} ```"
            }
        ]
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*Best mrr* :\n{mrr}"
            }
        ]
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": "*Feature importance* :\n" + str(imp_df.head(20)) 
            }
        ]
    }
  ]) 
# if configuration.debug:
#     exit(0)    

predictions = []
session_ids = []

test['score'] = clf.predict(d_test)
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






