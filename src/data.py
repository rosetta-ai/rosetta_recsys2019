import torch
import numpy as np
import pandas as pd
import pickle
import gc
from constant import *
from utils import *
from config import *
import torch
from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
from ordered_set import OrderedSet
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class NCFDataLoader():
    def __init__(self, data, config, shuffle=True, batch_size=128, continuous_features=None):
        self.item_id = torch.LongTensor(data.item_id.values)
        self.config = config
        self.label = torch.FloatTensor(data.label.values)
        self.past_interactions = torch.LongTensor(np.vstack(data.past_interactions.values))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.item_id))
        self.past_interaction_masks = self.past_interactions != self.config.transformed_dummy_item
        self.price_rank = torch.LongTensor(data.price_rank.values)
        self.city = torch.LongTensor(data.city.values)
        self.last_item = torch.LongTensor(data.last_item.values)
        self.impression_index = torch.LongTensor(data.impression_index)
        
        self.continuous_features = torch.FloatTensor(data.loc[:,continuous_features].values)

        self.neighbor_prices = torch.FloatTensor(np.vstack(data.neighbor_prices))
        # other_is_interacted = torch.FloatTensor(np.vstack(data.other_is_interacted))
        # other_is_clicked = torch.FloatTensor(np.vstack(data.other_is_clicked))
        
        # self.continuous_features = torch.cat([self.continuous_features, other_is_interacted, other_is_clicked], dim=1)
        # print(data.neighbor_prices)
        # print(neighbor_prices.shape)
        # self.continuous_features = torch.cat([self.continuous_features, neighbor_prices], dim=1)
        

        self.star = torch.LongTensor(data.star)
        
        self.past_interactions_sess = torch.LongTensor(np.vstack(data.past_interactions_sess.values))
        self.past_actions_sess = torch.LongTensor(np.vstack(data.past_actions_sess.values))
        self.last_click_item = torch.LongTensor(data.last_click_item.values)
        self.last_click_impression = torch.LongTensor(data.last_click_impression.values)
        self.last_interact_index = torch.LongTensor(data.last_interact_index.values)
        self.other_item_ids = torch.LongTensor(np.vstack(data.other_item_ids.values))
        self.city_platform = torch.LongTensor(data.city_platform.values)
        # self.log_price = torch.LongTensor(data.log_price.values)
        # self.user_id = torch.LongTensor(data.user_id.values)
        # self.other_item_impressions = torch.LongTensor(np.vstack(data.other_item_impressions.values))
        
        assert len(self.item_id) == len(self.past_interactions)
        assert len(self.past_interactions) == len(self.label)
    def __len__(self):
        return len(self.item_id) // self.batch_size

    def __iter__(self):
        self.batch_id = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.batch_id * self.batch_size <= len(self.indices):
            current_indices = self.indices[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
            result = [self.item_id[current_indices], self.label[current_indices], self.past_interactions[current_indices]\
                     , self.past_interaction_masks[current_indices], self.price_rank[current_indices], self.city[current_indices]\
                     , self.last_item[current_indices], self.impression_index[current_indices], self.continuous_features[current_indices]\
                     , self.star[current_indices], self.past_interactions_sess[current_indices], self.past_actions_sess[current_indices]\
                     , self.last_click_item[current_indices], self.last_click_impression[current_indices], self.last_interact_index[current_indices]\
                     , self.neighbor_prices[current_indices], self.other_item_ids[current_indices], self.city_platform[current_indices]]
                     #, self.other_item_ids[current_indices], self.other_item_impressions[current_indices]]
            self.batch_id += 1
            return result
        else:
            raise StopIteration



class NCFDataGenerator():
    """Construct dataset for NCF"""
    def __init__(self, config):
        """
        args:
            target_action: the target action at the next timestep. Can be 'buy', 'select', 'click', 'view'
            monitor_actions: the action that we should keep track with
        """

        self.config = config

        self.target_action = self.config.target_action = 'clickout item'
        # self.config.keep_columns = self.keep_columns = ['session_id', 'user_id','item_id', 'impressions','prices', 'city', 'step', 'last_item']
        self.config.all_cat_columns = self.all_cat_columns = ['user_id', 'item_id', 'city','action', 'city_platform']
        
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

        # with open('../output/ncf_xnn_intpop_clickout_v2_all_ut_lgb_bf_beach_v2_all_ut_xgb_4_8_8_pseudo_label.p','rb') as f:
        #     pseudo_label_test = pickle.load(f)


        if config.sub_sample:
            with open('../input/selected_users_140k.p', 'rb') as f:
                selected_users = pickle.load(f)

            train = train.loc[train.user_id.isin(selected_users),:]
        
        if config.debug:
            train = train.sample(1000)
            test = test.sample(1000)
        
        train.rename(columns={'reference': 'item_id', 'action_type':'action'}, inplace=True)
        test.rename(columns={'reference': 'item_id', 'action_type':'action'}, inplace=True)
        
        
        # test = test.merge(pseudo_label_test, on=['session_id'], how='left')
        # align columns
        # train['pseudo_label'] = np.nan
        # train = train.loc[:, test.columns]
        

        # fill item_id with DUMMY
        train.loc[train.action=='change of sort order','action'] = train.loc[train.action=='change of sort order'].apply(lambda row: row.action + str(row.item_id), axis=1)
        test.loc[test.action=='change of sort order','action'] = test.loc[test.action=='change of sort order'].apply(lambda row: row.action + str(row.item_id), axis=1)


        train.loc[train.action=='filter selection','action'] = train.loc[train.action=='filter selection'].apply(lambda row: row.action + str(row.item_id), axis=1)
        test.loc[test.action=='filter selection','action'] = test.loc[test.action=='filter selection'].apply(lambda row: row.action + str(row.item_id), axis=1)






        train.loc[train.action.str.contains('change of sort order'), 'item_id'] = DUMMY_ITEM
        test.loc[test.action.str.contains('change of sort order'), 'item_id'] = DUMMY_ITEM

        train.loc[train.action.str.contains('search for poi'), 'item_id'] = DUMMY_ITEM
        test.loc[test.action.str.contains('search for poi'), 'item_id'] = DUMMY_ITEM        

        train.loc[train.action.str.contains('filter selection'), 'item_id'] = DUMMY_ITEM
        test.loc[test.action.str.contains('filter selection'), 'item_id'] = DUMMY_ITEM        

        train.loc[train.action.str.contains('search for destination'), 'item_id'] = DUMMY_ITEM
        test.loc[test.action.str.contains('search for destination'), 'item_id'] = DUMMY_ITEM  
        
        
        

        
        # filter out rows where reference doesn't present in impression
        train['in_impressions'] = True
        train.loc[~train.impressions.isna(), 'in_impressions'] = train.loc[~train.impressions.isna()].apply(lambda row:row.item_id in row.impressions.split('|'), axis=1)
        train = train.loc[train.in_impressions].drop('in_impressions', axis=1).reset_index(drop=True)

        test['in_impressions'] = True
        test.loc[(~test.impressions.isna()) & (~test.item_id.isna()), 'in_impressions'] = test.loc[(~test.impressions.isna())& (~test.item_id.isna())].apply(lambda row:row.item_id in row.impressions.split('|'), axis=1)
        test = test.loc[test.in_impressions].drop('in_impressions', axis=1).reset_index(drop=True)
        
       
        train['item_id'] = train['item_id'].apply(str)
        train.loc[~train.impressions.isna(),'impressions'] = train.loc[~train.impressions.isna()].impressions.apply(lambda x: x.split('|'))
        train.loc[~train.prices.isna(), 'prices'] = train.loc[~train.prices.isna()].prices.apply(lambda x: x.split('|')).apply(lambda x: [int(p) for p in x])
        
        
  
        test['item_id'] = test['item_id'].apply(str)
        test.loc[~test.impressions.isna(),'impressions'] = test.loc[~test.impressions.isna()].impressions.apply(lambda x: x.split('|'))
        test.loc[~test.prices.isna(),'prices'] = test.loc[~test.prices.isna()].prices.apply(lambda x: x.split('|')).apply(lambda x: [int(p) for p in x])
        


        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)
        all_items = []
        
        for imp in data.loc[~data.impressions.isna()].impressions.tolist() + [data.item_id.apply(str).tolist()]:
            all_items += imp

        unique_items = OrderedSet(all_items)
        unique_actions = OrderedSet(data.action.values)

        train_session_interactions = dict(train.groupby('session_id')['item_id'].apply(list))
        test_session_interactions = dict(test.groupby('session_id')['item_id'].apply(list))

        
        train_session_actions = dict(train.groupby('session_id')['action'].apply(list))
        test_session_actions = dict(test.groupby('session_id')['action'].apply(list))


        train['sess_step'] = train.groupby('session_id')['timestamp'].rank(method='max').apply(int)
        test['sess_step'] = test.groupby('session_id')['timestamp'].rank(method='max').apply(int)
        

        train['city_platform'] = train.apply(lambda x: x['city'] + x['platform'], axis=1)
        test['city_platform'] = test.apply(lambda x: x['city'] + x['platform'], axis=1)
        # get last item
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
        # train.loc[train.action != 'clickout item','last_item'] = \
        # train.loc[train.action != 'clickout item', 'item_id']
        
        # test.loc[test.action != 'clickout item' ,'last_item'] = \
        # test.loc[test.action != 'clickout item', 'item_id']
        
        train['step_rank'] = train.groupby('session_id')['timestamp'].rank(method='max', ascending=True)
        test['step_rank'] = test.groupby('session_id')['timestamp'].rank(method='max', ascending=True)
        
       
        
        # train["last_item"].fillna(method='ffill', inplace=True)
        # test["last_item"].fillna(method='ffill', inplace=True)
        


        # filter actions        
        
        

        
        train.loc[(train.step_rank == 1) , 'last_item'] = DUMMY_ITEM
        test.loc[(test.step_rank == 1) , 'last_item'] = DUMMY_ITEM

        
        train.loc[(train.step_rank == 2) , 'second_last_item'] = DUMMY_ITEM
        test.loc[(test.step_rank == 2) , 'second_last_item'] = DUMMY_ITEM
        


        
        


        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)
        
        data_feature = data.loc[:,['id','session_id','timestamp', 'step']].copy()
        data_feature['time_diff'] = data_feature.groupby('session_id')['timestamp'].diff()
        data_feature['time_diff_diff'] = data_feature.groupby('session_id')['time_diff'].diff()
        data_feature['time_diff'] = GaussRankScaler().fit_transform(data_feature['time_diff'].values)
        data_feature['time_diff_diff'] = GaussRankScaler().fit_transform(data_feature['time_diff_diff'].values)
        # data_feature.loc[:,'time_diff'].fillna(0, inplace=True)
        # print("na sum",data_feature['time_diff'].isna().sum())
        # data_feature['time_diff'] = MinMaxScaler().fit_transform(data_feature['time_diff'].values.reshape(-1,1))
        data_feature['mm_step'] = GaussRankScaler().fit_transform(data_feature['step'].values)
        data_feature['day'] = MinMaxScaler().fit_transform(pd.to_datetime(data.timestamp, unit='s').dt.day.values.reshape(-1,1) )
        data_feature['rg_timestamp'] = GaussRankScaler().fit_transform(data_feature['timestamp'].values)
        # data_feature['dayofweek'] = MinMaxScaler().fit_transform(pd.to_datetime(data.timestamp, unit='s').dt.dayofweek.values.reshape(-1,1) )
        # data_feature['hour'] = MinMaxScaler().fit_transform(pd.to_datetime(data.timestamp, unit='s').dt.hour.values.reshape(-1,1) )

        
        
        # data_feature['day_of_week'] = MinMaxScaler().fit_transform(pd.to_datetime(data.timestamp, unit='s').dt.day.values.reshape(-1,1) )

        data_feature = data_feature.drop( ['session_id','timestamp','step'],axis=1)

        
        # get time diff    
        train = train.merge(data_feature, on='id', how='left')
        test = test.merge(data_feature, on='id', how='left')

        train_session_time_diff = dict(train.groupby('session_id')['time_diff'].apply(list))
        test_session_time_diff = dict(test.groupby('session_id')['time_diff'].apply(list))

        self.cat_encoders = {}

        for col in self.all_cat_columns:
            self.cat_encoders[col] = CategoricalEncoder()
                
        
        
        
        
        


        self.cat_encoders['item_id'].fit(list(unique_items) + [DUMMY_ITEM] )
        # with open('../input/ncf_item_enc.p', 'wb') as f:
        #     pickle.dump(self.cat_encoders['item_id'],f)
        
        # with open('../input/ncf_item_enc.p', 'rb') as f:
        #     self.cat_encoders['item_id'] = pickle.load(f)

        self.cat_encoders['city'].fit(data.city.values)
        self.cat_encoders['city_platform'].fit(data.city_platform.values)
        self.cat_encoders['action'].fit( list(unique_actions) + [DUMMY_ACTION])

        with open('../input/user_encoder.p','rb') as f:
            self.cat_encoders['user_id'] = pickle.load(f)
        # self.cat_encoders['user_id'].fit(data.user_id.tolist() )


        for col in self.all_cat_columns:
            
            train[col] = self.cat_encoders[col].transform(train[col].values)
            test[col] = self.cat_encoders[col].transform(test[col].values)
            self.config.num_embeddings[col] = self.cat_encoders[col].n_elements


        #this is an integer
        self.config.transformed_clickout_action = self.transformed_clickout_action = self.cat_encoders['action'].transform(['clickout item'])[0]
        self.config.transformed_dummy_action = self.transformed_dummy_action = self.cat_encoders['action'].transform([DUMMY_ACTION])[0]
        self.transformed_interaction_image = self.cat_encoders['action'].transform(['interaction item image'])[0]
        self.transformed_interaction_deals = self.cat_encoders['action'].transform(['interaction item deals'])[0]
        self.transformed_interaction_info = self.cat_encoders['action'].transform(['interaction item info'])[0]
        self.transformed_interaction_rating = self.cat_encoders['action'].transform(['interaction item rating'])[0]

        self.config.transformed_dummy_item = self.transformed_dummy_item = self.cat_encoders['item_id'].transform([DUMMY_ITEM])[0]
        self.config.transformed_nan_item = self.transformed_nan_item = self.cat_encoders['item_id'].transform(['nan'])[0]
        

        # transform last item
        train['last_item'] = self.cat_encoders['item_id'].transform(train['last_item'].values)
        test['last_item'] = self.cat_encoders['item_id'].transform(test['last_item'].values)

        train['second_last_item'] = self.cat_encoders['item_id'].transform(train.second_last_item.values)
        test['second_last_item'] = self.cat_encoders['item_id'].transform(test.second_last_item.values)
        
        # transform session interactions and pad dummy in front of all of them
        for session_id, item_list in train_session_interactions.items():
            train_session_interactions[session_id] = [self.transformed_dummy_item] * self.config.sess_length + self.cat_encoders['item_id'].transform(item_list)

        for session_id, item_list in test_session_interactions.items():
            test_session_interactions[session_id] = [self.transformed_dummy_item] * self.config.sess_length + self.cat_encoders['item_id'].transform(item_list)

        for session_id, action_list in train_session_actions.items():
            train_session_actions[session_id] = [self.transformed_dummy_action] * self.config.sess_length + self.cat_encoders['action'].transform(action_list)

        for session_id, action_list in test_session_actions.items():
            test_session_actions[session_id] = [self.transformed_dummy_action] * self.config.sess_length + self.cat_encoders['action'].transform(action_list)


        implicit_train = train.loc[train.action != self.transformed_clickout_action, :]
        implicit_test = test.loc[test.action != self.transformed_clickout_action, :]

        # implicit_train = implicit_train.drop_duplicates(subset=['session_id','item_id','action'])
        # implicit_test = implicit_test.drop_duplicates(subset=['session_id','item_id','action'])

        


        # get interaction count for all item
        interaction_item_ids = implicit_train.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + implicit_test.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        unique_interaction_items, counts = np.unique(interaction_item_ids, return_counts=True)
        self.interaction_count_dict = dict(zip(unique_interaction_items, counts))        

        # get interaction count for all item
        interaction_image_item_ids = train.loc[train.action == self.transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == self.transformed_interaction_image, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        unique_interaction_image_items, counts = np.unique(interaction_image_item_ids, return_counts=True)
        self.image_count_dict = dict(zip(unique_interaction_image_items, counts))        

        # interaction_deals_item_ids = train.loc[train.action == self.transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == self.transformed_interaction_deals, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        # unique_interaction_deals_items, counts = np.unique(interaction_deals_item_ids, return_counts=True)
        # self.deals_count_dict = dict(zip(unique_interaction_deals_items, counts))        


        # interaction_info_item_ids = train.loc[train.action == self.transformed_interaction_info, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == self.transformed_interaction_info, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        # unique_interaction_info_items, counts = np.unique(interaction_info_item_ids, return_counts=True)
        # self.info_count_dict = dict(zip(unique_interaction_info_items, counts))        

        # interaction_rating_item_ids = train.loc[train.action == self.transformed_interaction_rating, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.loc[test.action == self.transformed_interaction_rating, :].drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        # unique_interaction_rating_items, counts = np.unique(interaction_rating_item_ids, return_counts=True)
        # self.rating_count_dict = dict(zip(unique_interaction_rating_items, counts))        

        # get only the clickout
        train = train.loc[train.action ==self.transformed_clickout_action,:]
        test = test.loc[test.action == self.transformed_clickout_action,:]


        train['step_rank'] = train.groupby('session_id')['step'].rank(method='max', ascending=False)
        


        # compute global item-price DataFrame
        # prices = np.hstack([np.hstack(train['prices'].values), np.hstack(test.prices.values)])
        item_ids = np.hstack([np.hstack(train['impressions'].values), np.hstack(test.impressions.values)])
        
        unique_items, counts = np.unique(item_ids, return_counts=True)
        self.item_popularity_dict = dict(zip(unique_items, counts))

        clickout_item_ids = train.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist() + test.drop_duplicates(subset=['session_id','item_id','action']).item_id.tolist()
        unique_clickout_items, counts = np.unique(clickout_item_ids, return_counts=True)

        self.clickout_count_dict = dict(zip(unique_clickout_items, counts))        
        
        self.platform_clickout_count = pd.concat([train, test], axis=0).groupby(['platform','item_id']).size()
        # compute mean value for each item
        # self.item_mean_price_dict = dict(price_table.groupby('item_id')['prices'].mean())




        if config.debug:
            val = train.loc[train.step_rank == 1,:].iloc[:5]
        else:
            val = train.loc[train.step_rank == 1,:].iloc[:50000]

        val_index = val.index
        train = train.loc[~train.index.isin(val_index),:]
        

        
        
        
        # {'user_id':[11,2,5,9,]}
        self.past_interaction_dict = {}
        self.past_interaction_dict_sess = {}
        self.last_click_sess_dict = {}
        self.last_impressions_dict = {}
        self.sess_impressions_dict = {}
        self.sess_last_step_dict = {}
        self.sess_last_imp_idx_dict = {}
        self.sess_last_price_dict = {}
        self.sess_time_diff_dict = {}
        


        
         # split the interaction df into train/ val and construct training sequences
        self.train_data = self.build_user_item_interactions(train, train_session_interactions, train_session_actions, train_session_time_diff)
        self.val_data = self.build_user_item_interactions(val, train_session_interactions, train_session_actions, train_session_time_diff)
        self.test_data, labeled_test = self.build_user_item_interactions(test, test_session_interactions,  test_session_actions, test_session_time_diff, training=False)

        # standard scale price

        price_sc = StandardScaler() 
        
        
        self.train_data['price_diff'] = price_sc.fit_transform(self.train_data.price_diff.values.reshape(-1,1))
        self.val_data['price_diff'] = price_sc.transform(self.val_data.price_diff.values.reshape(-1,1))
        self.test_data['price_diff'] = price_sc.transform(self.test_data.price_diff.values.reshape(-1,1))


        # self.train_data['last_price'] = price_sc.fit_transform(self.train_data.last_price.values.reshape(-1,1))
        # self.val_data['last_price'] = price_sc.transform(self.val_data.last_price.values.reshape(-1,1))
        # self.test_data['last_price'] = price_sc.transform(self.test_data.last_price.values.reshape(-1,1))

        price_mm = MinMaxScaler()
        self.train_data['price_ratio'] = price_mm.fit_transform(self.train_data.price_ratio.values.reshape(-1,1))
        self.val_data['price_ratio'] = price_mm.transform(self.val_data.price_ratio.values.reshape(-1,1))
        self.test_data['price_ratio'] = price_mm.transform(self.test_data.price_ratio.values.reshape(-1,1))

        # mm = MinMaxScaler()
        # self.train_data['interaction_count'] = mm.fit_transform(self.train_data.interaction_count.values.reshape(-1,1))
        # self.val_data['interaction_count'] = mm.transform(self.val_data.interaction_count.values.reshape(-1,1))
        # self.test_data['interaction_count'] = mm.transform(self.test_data.interaction_count.values.reshape(-1,1))        

        # self.train_data['mean_price'] = price_mm.fit_transform(self.train_data.mean_price.values.reshape(-1,1))
        # self.val_data['mean_price'] = price_mm.transform(self.val_data.mean_price.values.reshape(-1,1))
        # self.test_data['mean_price'] = price_mm.transform(self.test_data.mean_price.values.reshape(-1,1))



        
        # self.train_data['clickout_time_diff'] = price_mm.fit_transform(self.train_data.clickout_time_diff.values.reshape(-1,1))
        # self.val_data['clickout_time_diff'] = price_mm.transform(self.val_data.clickout_time_diff.values.reshape(-1,1))
        # self.test_data['clickout_time_diff'] = price_mm.transform(self.test_data.clickout_time_diff.values.reshape(-1,1))        
            


        price_mm.fit(np.hstack([np.hstack(self.train_data.neighbor_prices.values), np.hstack(self.val_data.neighbor_prices.values),\
            np.hstack(self.test_data.neighbor_prices.values)]).reshape(-1,1) )
        # print(self.train_data['neighbor_prices'].head(5))
        self.train_data['neighbor_prices'] = self.train_data['neighbor_prices'].apply(lambda x: price_mm.transform(np.array(x).reshape(-1,1)).reshape(-1))
        self.val_data['neighbor_prices'] = self.val_data['neighbor_prices'].apply(lambda x: price_mm.transform(np.array(x).reshape(-1,1)).reshape(-1))
        self.test_data['neighbor_prices'] = self.test_data['neighbor_prices'].apply(lambda x: price_mm.transform(np.array(x).reshape(-1,1)).reshape(-1))

        # print(self.train_data['price_diff'].min())
        # print(self.val_data['price_diff'].min())
        # print(self.test_data['price_diff'].min())

        # print(self.train_data['price_diff'].max())
        # print(self.val_data['price_diff'].max())
        # print(self.test_data['price_diff'].max())

        # print(self.train_data['price_diff'].isna().sum())
        # print(self.val_data['price_diff'].isna().sum())
        # print(self.test_data['price_diff'].isna().sum())

        if config.use_test:
            self.train_data = pd.concat([self.train_data, labeled_test], axis=0)
        
        sampled_test_session = self.test_data.session_id.sample(frac=0.3)

        # self.train_data = pd.concat([self.train_data, self.test_data.loc[self.test_data.session_id.isin(sampled_test_session)]], axis=0)
        # item_meta multi-hot
        item_meta = item_meta.loc[item_meta.item_id.isin(unique_items),:]
        item_meta['item_id'] = self.cat_encoders['item_id'].transform(item_meta['item_id'].values)
        item_meta['star'] = 0
        item_meta.loc[item_meta.properties.apply(lambda x: '1 Star' in x), 'star'] = 1
        item_meta.loc[item_meta.properties.apply(lambda x: '2 Star' in x), 'star'] = 2
        item_meta.loc[item_meta.properties.apply(lambda x: '3 Star' in x), 'star'] = 3
        item_meta.loc[item_meta.properties.apply(lambda x: '4 Star' in x), 'star'] = 4
        item_meta.loc[item_meta.properties.apply(lambda x: '5 Star' in x), 'star'] = 5

        unique_property = list(OrderedSet(np.hstack(item_meta.properties.tolist())))
        self.unique_property = unique_property
        
        self.cat_encoders['item_property'] = CategoricalEncoder()
        self.cat_encoders['item_property'].fit(unique_property)
        item_properties_array = []
        for row in item_meta.itertuples():
            current_row = np.zeros(len(unique_property) + 2)
            one_indices = self.cat_encoders['item_property'].transform(row.properties)
            current_row[one_indices] = 1
            current_row[-1] = row.item_id
            current_row[-2] = row.star
            item_properties_array.append(current_row)

        item_properties_array = np.vstack(item_properties_array)
        item_properties_df = pd.DataFrame(item_properties_array, columns=unique_property + ['star', 'item_id'])
        
        
        

        #tfidf
        # item_meta['properties'] = item_meta['properties'].apply(lambda props: ' '.join([ '_'.join(p.split(' ')) for p in props]))
        
        # vectorizer = TfidfVectorizer(token_pattern=r'(?u)[^\s]+')
        # X = vectorizer.fit_transform(item_meta.properties.values)
        # print("tfidf output shape", X.shape)
        # item_properties_df = pd.DataFrame(X.todense(), columns=np.arange(X.shape[1]) )
        # item_properties_df['star'] = item_meta.star
        # item_properties_df['item_id'] = item_meta.item_id

        item_properties_item_id = item_properties_df.item_id.values
        item_properties_star = item_properties_df.star.values
        
        tsvd = TruncatedSVD(n_components=30, n_iter=10, random_state=None)
        svd_matrix = tsvd.fit_transform(item_properties_df.drop( ['star', 'item_id'],axis=1).values)
        print("explained ratio", tsvd.explained_variance_ratio_.sum())
        svd_ip_columns = [ f'svd_ip_{i}' for i in np.arange(30)]
        item_properties_df = pd.DataFrame(svd_matrix, columns=svd_ip_columns)
        item_properties_df['item_id'] = item_properties_item_id
        item_properties_df['star'] = item_properties_star
        item_properties_df['pet_friendly'] = item_meta.properties.apply(lambda x: 'Pet Friendly' in x)
        item_properties_df['parking'] = item_meta.properties.apply(lambda x: 'Car Park' in x)
        item_properties_df = item_properties_df.astype(dtype= {"item_id":"int32","pet_friendly":"float32", "parking":"float32"})
            
            
        filter_df = data.loc[ ~data.current_filters.isna(), ['id', 'current_filters']]
        filter_df['current_filters'] = filter_df.current_filters.apply(lambda x:x.split('|'))
        filter_set = list(OrderedSet(np.hstack(filter_df['current_filters'].to_list())))

        self.cat_encoders['filters'] = CategoricalEncoder()
        self.cat_encoders['filters'].fit(filter_set)
        all_filter_array = []

        for row in filter_df.itertuples():
            current_row = np.zeros(len(filter_set) + 1, dtype=object)
            current_filters = row.current_filters
            one_indices = self.cat_encoders['filters'].transform(row.current_filters)
            current_row[one_indices] = 1
            current_row[-1] = row.id
            all_filter_array.append(current_row)
        
        
        all_filter_array = np.vstack(all_filter_array)
        filters_df = pd.DataFrame(all_filter_array, columns=  [f'ft_{f}' for f in filter_set] + ['id'])
        dtype_dict = {"id":"int32"}
        for f in filter_set:
            dtype_dict[f'ft_{f}'] = "int32"
        filters_df = filters_df.astype(dtype= dtype_dict)
        
        filters_id = filters_df.id.values

        
        tsvd = TruncatedSVD(n_components=10, n_iter=10, random_state=None)
        svd_matrix = tsvd.fit_transform(filters_df.drop( ['id'],axis=1).values)
        print("explained ratio", tsvd.explained_variance_ratio_.sum())
        svd_ft_columns = [ f'svd_ft_{i}' for i in np.arange(10)]
        filters_df = pd.DataFrame(svd_matrix, columns=svd_ft_columns)
        for c in svd_ft_columns:
            filters_df[c] = MinMaxScaler().fit_transform(filters_df[c].values.reshape(-1,1))
        filters_df['id'] = filters_id

        del train, test, data
        gc.collect()

        self.train_data = self.train_data.merge(item_properties_df, on="item_id", how="left")
        self.val_data = self.val_data.merge(item_properties_df, on="item_id", how="left")
        self.test_data = self.test_data.merge(item_properties_df, on="item_id", how="left")
        
        self.train_data = self.train_data.merge(filters_df, on=['id'], how="left")
        self.val_data = self.val_data.merge(filters_df, on=['id'], how="left")
        self.test_data = self.test_data.merge(filters_df, on=['id'], how="left")

        self.train_data = self.train_data.merge(data_feature, on=['id'], how="left")
        self.val_data = self.val_data.merge(data_feature, on=['id'], how="left")
        self.test_data = self.test_data.merge(data_feature, on=['id'], how="left")
        
        self.train_data['interaction_image_count'] = self.train_data.item_id.map(self.image_count_dict)
        self.val_data['interaction_image_count'] = self.val_data.item_id.map(self.image_count_dict)
        self.test_data['interaction_image_count'] = self.test_data.item_id.map(self.image_count_dict)

        # self.train_data['interaction_deals_count'] = self.train_data.item_id.map(self.deals_count_dict)
        # self.val_data['interaction_deals_count'] = self.val_data.item_id.map(self.deals_count_dict)
        # self.test_data['interaction_deals_count'] = self.test_data.item_id.map(self.deals_count_dict)

        # self.train_data['interaction_info_count'] = self.train_data.item_id.map(self.info_count_dict)
        # self.val_data['interaction_info_count'] = self.val_data.item_id.map(self.info_count_dict)
        # self.test_data['interaction_info_count'] = self.test_data.item_id.map(self.info_count_dict)

        # self.train_data['interaction_rating_count'] = self.train_data.item_id.map(self.rating_count_dict)
        # self.val_data['interaction_rating_count'] = self.val_data.item_id.map(self.rating_count_dict)
        # self.test_data['interaction_rating_count'] = self.test_data.item_id.map(self.rating_count_dict)

        # max_step_diff = max(self.train_data['step_diff'].values.max(),self.val_data['step_diff'].values.max(),self.test_data['step_diff'].values.max()  ) + 1
        
        # print("max_step_diff", max_step_diff)
        # self.train_data['step_diff'] /= max_step_diff
        # self.val_data['step_diff'] /= max_step_diff
        # self.test_data['step_diff'] /= max_step_diff

        # with open('../input/doc2vec_ft_32_df.p','rb') as f:
        #     doc2vec_df = pickle.load(f)
        # d2v_columns = [c for c in doc2vec_df.columns if c != 'id']
        # self.train_data = self.train_data.merge(doc2vec_df, on='id',how='left')
        # self.val_data = self.val_data.merge(doc2vec_df, on='id',how='left')
        # self.test_data = self.test_data.merge(doc2vec_df, on='id',how='left')

        train_other_is_interacted = np.vstack(self.train_data.other_is_interacted.values).astype(np.float32)
        val_other_is_interacted = np.vstack(self.val_data.other_is_interacted.values).astype(np.float32)
        test_other_is_interacted = np.vstack(self.test_data.other_is_interacted.values).astype(np.float32)

        is_interacted_columns = []
        for i in range(train_other_is_interacted.shape[1]):
            col = f'is_int_{i}'
            is_interacted_columns.append(col)
            self.train_data[col] = train_other_is_interacted[:,i]
            self.val_data[col] = val_other_is_interacted[:,i]
            self.test_data[col] = test_other_is_interacted[:,i]

        self.train_data.drop('other_is_interacted',axis=1, inplace=True)
        self.val_data.drop('other_is_interacted',axis=1, inplace=True)
        self.test_data.drop('other_is_interacted',axis=1, inplace=True)

        train_other_is_clicked = np.vstack(self.train_data.other_is_clicked.values).astype(np.float32)
        val_other_is_clicked = np.vstack(self.val_data.other_is_clicked.values).astype(np.float32)
        test_other_is_clicked = np.vstack(self.test_data.other_is_clicked.values).astype(np.float32)
        

        is_clicked_columns = []
        for i in range(train_other_is_clicked.shape[1]):
            col = f'is_cl_{i}'
            is_clicked_columns.append(col)
            self.train_data[col] = train_other_is_clicked[:,i]
            self.val_data[col] = val_other_is_clicked[:,i]
            self.test_data[col] = test_other_is_clicked[:,i]

        self.train_data.drop('other_is_clicked',axis=1, inplace=True)
        self.val_data.drop('other_is_clicked',axis=1, inplace=True)
        self.test_data.drop('other_is_clicked',axis=1, inplace=True)

        # rank gauss transform
        train_len = self.train_data.shape[0]
        val_len = self.val_data.shape[0]

        # for c in ['sum_time_elapse']:
        #     feature = self.train_data[c].values.tolist() + self.val_data[c].values.tolist() + self.test_data[c].values.tolist()
        #     transformed_feature = GaussRankScaler().fit_transform(feature)
        #     self.train_data[c] = transformed_feature[:train_len]
        #     self.val_data[c] = transformed_feature[train_len:val_len+train_len]
        #     self.test_data[c] = transformed_feature[val_len+train_len:]

        self.continuous_features = svd_ip_columns + svd_ft_columns + is_interacted_columns + is_clicked_columns + ['mm_step','time_diff', 'day', 'mm_price', 'equal_last_impressions', 'price_diff','price','last_price','price_ratio','is_clicked','is_interacted','item_popularity','is_interacted_image','is_interacted_deals','interaction_count','clickout_count','interaction_image_count','click_diff','rg_timestamp','equal_last_item','global_clickout_count_rank','rg_price','interaction_count_avg','avg_is_interacted_image','avg_is_interacted']


        # normalize num_impressions


        # target encoding
        agg_cols = ['impression_index','price_rank']
        for c in agg_cols:
            gp = self.train_data.groupby(c)['label']
            mean = gp.mean()
            self.train_data[f'{c}_label_avg'] = self.train_data[c].map(mean)
            self.val_data[f'{c}_label_avg'] = self.val_data[c].map(mean)
            self.test_data[f'{c}_label_avg'] = self.test_data[c].map(mean)

            self.continuous_features.append(f'{c}_label_avg')    

        # agg_cols = ['city']
        # for c in agg_cols:
        #     gp = self.train_data.loc[self.train_data.label == 1].groupby(c)['impression_index']
        #     mean = gp.mean()
        #     self.train_data[f'{c}_imp_idx_avg'] = self.train_data[c].map(mean)
        #     self.val_data[f'{c}_imp_idx_avg'] = self.val_data[c].map(mean)
        #     self.test_data[f'{c}_imp_idx_avg'] = self.test_data[c].map(mean)

        #     self.continuous_features.append(f'{c}_imp_idx_avg')    
        
        agg_cols = ['city']
        for c in agg_cols:
            gp = self.train_data.groupby(c)['price']
            mean = gp.mean()
            self.train_data[f'{c}_price_avg'] = self.train_data[c].map(mean)
            self.val_data[f'{c}_price_avg'] = self.val_data[c].map(mean)
            self.test_data[f'{c}_price_avg'] = self.test_data[c].map(mean)

            self.continuous_features.append(f'{c}_price_avg')                

        agg_cols = ['city']
        for c in agg_cols:
            gp = self.train_data.groupby(c)['price']
            mean = gp.std()
            self.train_data[f'{c}_price_std'] = self.train_data[c].map(mean)
            self.val_data[f'{c}_price_std'] = self.val_data[c].map(mean)
            self.test_data[f'{c}_price_std'] = self.test_data[c].map(mean)

            self.continuous_features.append(f'{c}_price_std') 
        
        #normalize 
        self.train_data['global_clickout_count_rank'] /= 25
        self.val_data['global_clickout_count_rank'] /= 25
        self.test_data['global_clickout_count_rank'] /= 25

        
               
        # agg_cols = ['city']
        # for c in agg_cols:
        #     gp = self.train_data.groupby(c)['time_diff']
        #     mean = gp.mean()
        #     self.train_data[f'{c}_td_avg'] = self.train_data[c].map(mean)
        #     self.val_data[f'{c}_td_avg'] = self.val_data[c].map(mean)
        #     self.test_data[f'{c}_td_avg'] = self.test_data[c].map(mean)

        #     self.continuous_features.append(f'{c}_td_avg')                    

        # fill zero
        for col in ['star','time_diff']:

            self.train_data.loc[:,col].fillna(0, inplace=True)
            self.val_data.loc[:,col].fillna(0, inplace=True)
            self.test_data.loc[:,col].fillna(0, inplace=True)

        

        for up in self.continuous_features :
            mean_value = self.train_data.loc[ ~self.train_data[up].isna() , up].mean()
            self.train_data.loc[:,up].fillna(mean_value, inplace=True)
            self.val_data.loc[:,up].fillna(mean_value, inplace=True)
            self.test_data.loc[:,up].fillna(mean_value, inplace=True)
        

        for c in self.continuous_features:
            if self.train_data[c].isna().sum() >0 or self.val_data[c].isna().sum() >0 or self.test_data[c].isna().sum() >0:
                print("is null!!", c)

        self.config.num_embeddings['price_rank'] = 25
        self.config.num_embeddings['impression_index'] = 26
        
        # self.config.num_embeddings['day_of_week'] = 7
        self.config.num_embeddings['star'] = 6

        self.config.all_cat_columns+= ['price_rank', 'impression_index', 'star']
        
        self.config.continuous_size = len(self.continuous_features) 
        self.config.neighbor_size = 5
        
        self.all_cat_columns = self.config.all_cat_columns
        
        if self.config.verbose:
            print(f"Number of training data: {self.train_data.shape}")
            print(f"Number of validation data: {self.val_data.shape}")
            print(f"Number of test data: {self.test_data.shape}")
    
    def get_features(self):
        return ', '.join([c  for c in self.continuous_features if 'svd' not in c])

    def build_user_item_interactions(self, df, session_interactions, session_actions, session_time_diff, training=True):
        df_list = []
        label_test_df_list = []
        # parse impressions for train set
        for idx, row in enumerate(tqdm(df.itertuples())):
            if row.user_id not in self.past_interaction_dict:
                self.past_interaction_dict[row.user_id] = [self.transformed_dummy_item] * self.config.sequence_length
            # if row.session_id not in self.past_interaction_dict_sess:
            #     self.past_interaction_dict_sess[row.session_id] = [self.transformed_dummy_item] * self.config.sess_length
            if row.session_id not in self.last_click_sess_dict:
                self.last_click_sess_dict[row.session_id] = self.transformed_dummy_item

            if row.session_id not in self.last_impressions_dict:
                self.last_impressions_dict[row.session_id] = None

            if row.session_id not in self.sess_last_imp_idx_dict:
                self.sess_last_imp_idx_dict[row.session_id] = DUMMY_IMPRESSION_INDEX

            if row.session_id not in self.sess_last_price_dict:
                self.sess_last_price_dict[row.session_id] = None
            
            if row.session_id not in self.sess_time_diff_dict:
                self.sess_time_diff_dict[row.session_id] = None

            # if row.session_id not in self.sess_impressions_dict:
            #     self.sess_impressions_dict[row.session_id] = set()
            # if row.session_id not in self.sess_last_step_dict:
            #     self.sess_last_step_dict[row.session_id] = None
            
            transformed_impressions = self.cat_encoders['item_id'].transform(row.impressions, to_np=True)

            # compute session_interaction
            sess_step = row.sess_step
            session_id = row.session_id
            
            current_session_interactions = session_interactions[session_id][:self.config.sess_length+ sess_step -1] # -1 for excluding the current row
            current_session_interactions = current_session_interactions[-self.config.sess_length:]
            
            current_session_actions = session_actions[session_id][:self.config.sess_length+ sess_step -1]
            current_session_actions = current_session_actions[-self.config.sess_length:]

            assert len(current_session_interactions) == self.config.sess_length
            
            if row.last_item in transformed_impressions:
                last_interact_index = transformed_impressions.tolist().index(row.last_item)
            else:
                last_interact_index = DUMMY_IMPRESSION_INDEX
            
            if row.second_last_item in transformed_impressions:
                second_last_interact_index = transformed_impressions.tolist().index(row.second_last_item)
            else:
                second_last_interact_index = DUMMY_IMPRESSION_INDEX

            # if row.item_id != self.transformed_nan_item:
                # training
            label = transformed_impressions == row.item_id
            # else:

            # last3_impression_idices = [ transformed_impressions.index(imp)   for imp in session_interactions[session_id][self.config.sess_length+ sess_step -4:self.config.sess_length+ sess_step -1] if imp in transformed_impressions else DUMMY_IMPRESSION_INDEX]
            # #     # test
            #     label = row.pseudo_label
            # if len(transformed_impressions) < 25:
            #     padded_transformed_impressions = np.array(transformed_impressions.tolist() + [self.transformed_dummy_item] * (25 - len(transformed_impressions)))
            # else:
            #     padded_transformed_impressions = transformed_impressions.copy()
            interaction_image_indices = np.array(session_actions[session_id][:self.config.sess_length+ sess_step -1]) == self.transformed_interaction_image
            interaction_image_item =  np.array(session_interactions[session_id][:self.config.sess_length+ sess_step -1])[interaction_image_indices]
            sess_unique_items, counts = np.unique(interaction_image_item, return_counts=True)
            interaction_image_count_dict = dict(zip(sess_unique_items, counts))


            interaction_deals_indices = np.array(session_actions[session_id][:self.config.sess_length+ sess_step -1]) == self.transformed_interaction_deals
            interaction_deals_item =  np.array(session_interactions[session_id][:self.config.sess_length+ sess_step -1])[interaction_deals_indices]
            sess_unique_deals_items, counts = np.unique(interaction_deals_item, return_counts=True)
            interaction_deals_count_dict = dict(zip(sess_unique_deals_items, counts))


            interaction_clickout_indices = np.array(session_actions[session_id][:self.config.sess_length+ sess_step -1]) == self.transformed_clickout_action
            interaction_clickout_item =  np.array(session_interactions[session_id][:self.config.sess_length+ sess_step -1])[interaction_clickout_indices]
            sess_unique_clickout_items, counts = np.unique(interaction_clickout_item, return_counts=True)
            interaction_clickout_count_dict = dict(zip(sess_unique_clickout_items, counts))
                
            finite_time_diff_indices = np.isfinite(session_time_diff[session_id][:sess_step -1])
            finite_time_diff_array = np.array(session_time_diff[session_id][:sess_step -1])[finite_time_diff_indices]

            # don't leak the current clickout info
            unleaked_clickout_count = [self.clickout_count_dict[imp] if imp in self.clickout_count_dict else 0 for imp in transformed_impressions]
            unleaked_clickout_count = [unleaked_clickout_count[idx] -1 if imp == row.item_id else unleaked_clickout_count[idx] for idx, imp in enumerate(transformed_impressions)]

            # unleaked_platform_clickout_count = [self.platform_clickout_count[row.platform, imp] if (row.platform, imp) in self.platform_clickout_count else 0 for imp in transformed_impressions]
            # unleaked_platform_clickout_count = [unleaked_platform_clickout_count[idx] -1 if imp == row.item_id else unleaked_platform_clickout_count[idx] for idx, imp in enumerate(transformed_impressions)]

            other_is_interacted = [imp in session_interactions[session_id][:self.config.sess_length+ sess_step -1] for imp in transformed_impressions]
            padded_other_is_interacted = other_is_interacted + [False] * (25 - len(other_is_interacted))

            other_is_clicked = [imp in self.past_interaction_dict[row.user_id] for imp in transformed_impressions]
            padded_other_is_clicked = other_is_clicked + [False] * (25 - len(other_is_clicked))            


            unpad_interactions = session_interactions[session_id][self.config.sess_length:self.config.sess_length+ sess_step -1]
        
        
            unique_interaction = pd.unique(session_interactions[session_id][:self.config.sess_length+ sess_step -1])
            
            # time elapse of within two steps for each item before the clickout
            item_time_elapse_dict = {}

            for it, elapse in zip(unpad_interactions[:-1], session_time_diff[session_id][1:sess_step -1]):
                if it not in item_time_elapse_dict: #or elapse > item_time_elapse_dict[it]:

                    item_time_elapse_dict[it] = elapse
                else:
                    item_time_elapse_dict[it] += elapse
                

            if len(transformed_impressions) < 25:
                padded_transformed_impressions = np.array(transformed_impressions.tolist() + [self.transformed_dummy_item] * (25 - len(transformed_impressions)))
            else:
                padded_transformed_impressions = transformed_impressions.copy()
            # padded_transformed_impressions = np.array([transformed_impressions[0]] * 2 + transformed_impressions.tolist() + [transformed_impressions[-1]] * 2)
            padded_prices =  [ row.prices[0]] * 2 +  row.prices + [row.prices[-1]]*2
            price_rank = compute_rank(row.prices)
            current_rows = np.zeros([len(row.impressions), 41], dtype=object)
            current_rows[:, 0] = row.user_id
            current_rows[:, 1] = transformed_impressions
            current_rows[:, 2] = label
            current_rows[:, 3] = row.session_id
            current_rows[:, 4] = [np.array(self.past_interaction_dict[row.user_id])] * len(row.impressions)
            current_rows[:, 5] = price_rank
            current_rows[:, 6] = row.city
            current_rows[:, 7] = row.last_item

            # impression index               
            current_rows[:, 8] = np.arange(len(transformed_impressions))
            current_rows[:, 9] = row.step
            current_rows[:, 10] = row.id
            
            current_rows[:, 11] = [np.array(current_session_interactions)] * len(row.impressions)
            current_rows[:, 12] = [np.array(current_session_actions)] * len(row.impressions)
            current_rows[:, 13] = MinMaxScaler().fit_transform(np.array(row.prices).reshape(-1,1)).reshape(-1)
            current_rows[:, 14] = row.prices

            # last click item id
            current_rows[:, 15] = self.last_click_sess_dict[row.session_id]

            # equal_last_impressions
            current_rows[:, 16] = self.last_impressions_dict[row.session_id] == transformed_impressions.tolist() 

            # impression index of last clicked item
            current_rows[:, 17] = self.sess_last_imp_idx_dict[row.session_id]

            #impression index of last interacted item 
            current_rows[:, 18] = last_interact_index

            # price difference with last interacted item
            current_rows[:, 19] = row.prices - self.sess_last_price_dict[row.session_id] if self.sess_last_price_dict[row.session_id] else 0


            current_rows[:, 20] = self.sess_last_price_dict[row.session_id] if self.sess_last_price_dict[row.session_id] else 0
            current_rows[:, 21] = row.prices / self.sess_last_price_dict[row.session_id] if self.sess_last_price_dict[row.session_id] else 0
            current_rows[:, 22] = [ padded_prices[i:i+5] for i in range(len(row.impressions))]

            current_rows[:, 23] = [np.concatenate([padded_transformed_impressions[:i], padded_transformed_impressions[i+1:]]) for i in range(len(row.impressions))]
            current_rows[:, 24] = row.city_platform

            # if that item has been clicked  by the current user
            current_rows[:, 25] = [imp in self.past_interaction_dict[row.user_id] for imp in transformed_impressions]

            # if that item has been interaced in the current session
            current_rows[:, 26] = [imp in session_interactions[session_id][:self.config.sess_length+ sess_step -1] for imp in transformed_impressions]

            # note that the impressions here was not transformed
            current_rows[:, 27] = [self.item_popularity_dict[imp] for imp in row.impressions]

            current_rows[:, 28] = [1 if imp in interaction_image_count_dict else 0 for imp in transformed_impressions] 
            current_rows[:, 29] = [1 if imp in interaction_deals_count_dict else 0 for imp in transformed_impressions] 

            current_rows[:, 30] = [self.interaction_count_dict[imp] if imp in self.interaction_count_dict else 0 for imp in transformed_impressions]
            current_rows[:, 31] = unleaked_clickout_count
            current_rows[:, 32] = [self.past_interaction_dict[row.user_id][::-1].index(imp) if imp in self.past_interaction_dict[row.user_id] else 0 for imp in transformed_impressions]
            current_rows[:, 33] = [np.array(padded_other_is_interacted)] * len(row.impressions)
            current_rows[:, 34] = [np.array(padded_other_is_clicked)] * len(row.impressions)
            current_rows[:, 35] = transformed_impressions == row.last_item
            current_rows[:, 36] = np.argsort(np.argsort(unleaked_clickout_count))
            current_rows[:, 37] = GaussRankScaler().fit_transform(row.prices)
            current_rows[:, 38] = np.mean(current_rows[:, 30])
            current_rows[:, 39] = np.mean(current_rows[:, 28])
            current_rows[:, 40] = np.mean(current_rows[:, 26])

            # current_rows[:, 41] = np.mean(finite_time_diff_array)
            # current_rows[:, 41] = np.std(current_rows[:, 30])
            # current_rows[:, 41] = 2 * last_interact_index - second_last_interact_index


            # current_rows[:, 41] = second_last_interact_index

            #TODO: Rank  of statistics

            # print(unleaked_platform_clickout_count)
                        
            # current_rows[:, 35] = [session_interactions[session_id][:self.config.sess_length+ sess_step -1][::-1].index(imp) if imp in session_interactions[session_id][:self.config.sess_length+ sess_step -1] else 0 for imp in transformed_impressions]
            # for i in range(35, 42):
            #     current_rows[:, i]  = np.mean(current_rows[:, i-10])

            # current_rows[:, 29] = [interaction_clickout_count_dict[imp] if imp in interaction_clickout_count_dict else 0 for imp in transformed_impressions]             

            
            # neighboring item
            # current_rows[:, 23] = [np.concatenate([padded_transformed_impressions[:i], padded_transformed_impressions[i+1:]]) for i in range(len(row.impressions))]
            
            
            # current_rows[:, 20] = row.prices - np.concatenate([row.prices[1:], [row.prices[-1]]], axis=0)
            # current_rows[:, 21] = row.prices - np.concatenate([[row.prices[0]], row.prices[:-1]], axis=0)
            
            # current_rows[:, 17] = row.step - self.sess_last_step_dict[row.session_id] if self.sess_last_step_dict[row.session_id] else 0

            # back pad transformed impressions
            

            
            # current_rows[:, 16] = [np.delete(np.arange(25), i) for i in range(len(row.impressions))]
            # print(self.last_click_sess_dict[row.session_id], self.last_impressions_dict[row.session_id] == transformed_impressions.tolist())

            if training or  row.item_id == self.transformed_nan_item:

                df_list.append(current_rows)
            else:
                label_test_df_list.append(current_rows)
            #include both labeled and pseudo-labelled
            
                
            
            # pad current item_id to default dict
            self.past_interaction_dict[row.user_id] = self.past_interaction_dict[row.user_id][1:]
            self.past_interaction_dict[row.user_id].append(row.item_id)
            
            
            self.last_click_sess_dict[row.session_id] = row.item_id
            self.last_impressions_dict[row.session_id] = transformed_impressions.tolist()
            self.sess_last_step_dict[row.session_id] = row.step
            self.sess_time_diff_dict[row.session_id] = row.timestamp

            

            # update last impression index
            if row.item_id != self.transformed_nan_item:
                self.sess_last_imp_idx_dict[row.session_id] = (transformed_impressions == row.item_id).tolist().index(True)
                self.sess_last_price_dict[row.session_id] = np.array(row.prices)[ transformed_impressions == row.item_id ][0]
            
            
            
            

            
            
            
            
            

        data = np.vstack(df_list)
        dtype_dict = {"city":"int32", "last_item":"int32", 'impression_index':'int32', "step":"int32","id":"int32", "user_id":"int32",
                "item_id":"int32", "label": "int32", "price_rank":"int32", "mm_price":"float32", 'price':'float32', "last_click_item":"int32", "equal_last_impressions":"int8", 'last_click_impression':'int16', 'last_interact_index':'int16', 'price_diff':'float32','last_price':'float32','price_ratio':'float32','city_platform':'int32', 'is_clicked':'int8', 'is_interacted':'int8','item_popularity':'int32', 'is_interacted_image':'int8','is_interacted_deals':'int8','interaction_count':'int32','clickout_count':'int32','click_diff':'float32','equal_last_item':'int8','global_clickout_count_rank':'int8','rg_price':'float32','interaction_count_avg':'float32','avg_is_interacted_image':'float32','avg_is_interacted':'float32'}
        df_columns= ['user_id', 'item_id', 'label', 'session_id', 'past_interactions', 'price_rank', 'city', 'last_item', 'impression_index', 'step', 'id', 'past_interactions_sess', 'past_actions_sess', 'mm_price','price','last_click_item','equal_last_impressions', 'last_click_impression','last_interact_index','price_diff','last_price','price_ratio','neighbor_prices','other_item_ids','city_platform', 'is_clicked', 'is_interacted', 'item_popularity','is_interacted_image','is_interacted_deals','interaction_count','clickout_count','click_diff','other_is_interacted','other_is_clicked','equal_last_item','global_clickout_count_rank','rg_price','interaction_count_avg','avg_is_interacted_image', 'avg_is_interacted']
        df = pd.DataFrame(data, columns=df_columns)
        df = df.astype(dtype= dtype_dict)
        if training:
            return df
        else:
            label_test = np.vstack(label_test_df_list)
            label_test = pd.DataFrame(label_test, columns=df_columns)
            label_test = label_test.astype(dtype= dtype_dict)
            return df, label_test
    def instance_a_train_loader(self):


        train_data = self.train_data

        return NCFDataLoader(train_data, self.config, shuffle=True, batch_size=self.config.batch_size, continuous_features=self.continuous_features)
    def evaluate_data_valid(self):
        val_data = self.val_data
        return NCFDataLoader(val_data, self.config, shuffle=False, batch_size=self.config.batch_size, continuous_features=self.continuous_features)
            
    def instance_a_test_loader(self):
        test_data = self.test_data
        return NCFDataLoader(test_data, self.config, shuffle=False, batch_size=self.config.batch_size,continuous_features=self.continuous_features)


if __name__ =='__main__':
    conf = NCFConfiguration()
    data_gen = NCFDataGenerator(conf)
    with timer("gen"):
        for result in data_gen.instance_a_train_loader(128):
            print(result[-1])
            print(torch.LongTensor(result[-1]))
        
        for result in data_gen.instance_a_train_loader(128):
            print(result[-1])
            print(torch.LongTensor(result[-1]))
            