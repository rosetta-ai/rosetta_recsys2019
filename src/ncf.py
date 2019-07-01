import torch
from utils import *
from data import *
import torch.nn as nn

class SelfAttention(torch.nn.Module):
    def __init__(self, temperature):
        super(SelfAttention, self).__init__()
        self.temperature = temperature
        # self.q_linear = torch.nn.Linear(dim, dim)
        # self.v_linear = torch.nn.Linear(dim, dim)
        # self.k_linear = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask):
        
        # q = self.q_linear(past_interactions)
        # k = self.k_linear(past_interactions)
        # v = self.v_linear(past_interactions)
        

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
#         attn = attn / self.temperature
        
        
        attn = attn.masked_fill(mask, -np.inf)
        
        attn = self.softmax(attn)

        # fill invalid with 0
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
#         attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
        
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config, emb_dim):
        super(MultiHeadAttention, self).__init__()

        self.config = config
        self.n_head = n_head = 4
        self.d_model = d_model = emb_dim
        self.d_k = d_k = emb_dim // self.n_head
        self.d_v = d_v = emb_dim // self.n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = SelfAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(self.config.categorical_emb_dim)

        self.fc = nn.Linear(n_head * d_v, self.config.categorical_emb_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output

class Attention(torch.nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, q, k):
        '''
        q: (batch, feature_size)
        k: (batch, step_size, feature_size)
        '''

        # att (batch, step_size, 1)
        
        att = torch.bmm(k, q.unsqueeze(2)).squeeze(2)
        att = self.softmax(att)

        output = k * att.unsqueeze(2).repeat(1, 1, k.size(2))
        output = output.sum(dim=1)
        return output




class Dense(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):

        x = self.linear(x)
        x = torch.nn.ReLU()(x)

        return x

class NCF(torch.nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        
        self.num_embeddings = config.num_embeddings
        self.cross_layer_sizes = config.cross_layer_sizes
        self.fast_CIN_d = config.fast_CIN_d

        # self.conv_w_dict = {}
        # self.conv_v_dict = {}
        # for idx, layer_size in enumerate(self.cross_layer_sizes):
        #     idx = str(idx)
        #     # conv_w = torch.nn.Conv1d(7, layer_size * self.fast_CIN_d, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #     conv_w = torch.nn.Parameter(torch.Tensor( layer_size * self.fast_CIN_d, 7, 1)).cuda()
        #     # uniform init
        #     torch.nn.init.uniform_(conv_w)
        #     self.conv_w_dict[idx] = conv_w
        #     if idx != '0' :
        #         # conv_v = torch.nn.Conv1d(last_layer_size, layer_size * self.fast_CIN_d, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #         conv_v = torch.nn.Parameter(torch.Tensor(layer_size * self.fast_CIN_d, last_layer_size, 1)).cuda()
        #         # uniform init
        #         torch.nn.init.uniform_(conv_v)
        #         self.conv_v_dict[idx] = conv_v

        #     last_layer_size = layer_size

        


        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':
                # weights = np.load('../input/fasttext_e10_ws5_mc1_dim128.npy')
                # weights = torch.FloatTensor(weights)
                # self.emb_dict[cat_col] =  torch.nn.Embedding.from_pretrained(weights, freeze=True, padding_idx= self.config.transformed_dummy_item)
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            # elif cat_col == 'user_id' :
            #     model_name='user2vec_wo_ts_nonitem_nan_clickoutonly_mc5_e10_lr0.05'
            #     weights= np.load(f'../input/{model_name}_vecs.npy') 
            #     weights = torch.FloatTensor(weights)
            #     self.emb_dict[cat_col] = torch.nn.Embedding.from_pretrained(weights, freeze=True)
            else:    
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim)

        # self.att_sess = MultiHeadAttention(config, self.categorical_emb_dim*2)
            # torch.nn.init.xavier_normal_(self.emb_dict[cat_col].weight)
        self.gru_sess = torch.nn.GRU(input_size = self.categorical_emb_dim *2, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=2, batch_first=True)
        self.other_item_gru = torch.nn.GRU(input_size = self.categorical_emb_dim, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=1, batch_first=True)
        # self.other_item_item_att = Attention(config)
        # self.other_item_item_gru = torch.nn.GRU(input_size = self.categorical_emb_dim, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=1, batch_first=True)
        # minus 2 for user_id and action
        # self.other_item_dense = Dense(self.categorical_emb_dim +1, self.categorical_emb_dim)
        # self.all_other_item_dense = Dense(self.categorical_emb_dim * 24, self.categorical_emb_dim)

        self.cont_linear = torch.nn.Linear(config.continuous_size,self.categorical_emb_dim )
        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim*17 , self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size*2 + 3 + config.neighbor_size, self.hidden_dims[1] )
        self.dropout = torch.nn.Dropout(p=config.dropout_rate, inplace=False)
        self.output = torch.nn.Linear(self.hidden_dims[1]   , 1)
        # self.cin_linear = torch.nn.Linear(sum(self.cross_layer_sizes[1:]), self.categorical_emb_dim)
        # self.conv = torch.nn.Conv1d(self.cateembdim, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.att = SelfAttention(config)
        # self.att_sess = SelfAttention(config.categorical_emb_dim )
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim*17)
        self.bn_hidden = torch.nn.BatchNorm1d(self.hidden_dims[0] + config.continuous_size*2+ 3 + config.neighbor_size )
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, neighbor_prices, other_item_ids, city_platform):
        embeddings = []
        user_embeddings = []
        batch_size = item_id.size(0)
        # batch_size = categorical_data.size(0)

        

        
        emb_item = self.emb_dict['item_id'](item_id)
        emb_past_interactions = self.emb_dict['item_id'](past_interactions)
        emb_price_rank = self.emb_dict['price_rank'](price_rank)
        emb_city = self.emb_dict['city'](city)
        emb_last_item = self.emb_dict['item_id'](last_item)
        emb_impression_index = self.emb_dict['impression_index'](impression_index)
        emb_star = self.emb_dict['star'](star)
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess)
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess)
        
        emb_last_click_item = self.emb_dict['item_id'](last_click_item)
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression)
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index)

        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids)
        
        emb_other_item_ids_gru, _ = self.other_item_gru(emb_other_item_ids)


        # emb_other_item_ids_item = emb_other_item_ids * emb_item.unsqueeze(1).repeat(1, emb_other_item_ids.size(1),1)
        # emb_other_item_ids_item, _ = self.other_item_item_gru(emb_other_item_ids_item)
        # pooled_emb_other_item_ids_item = F.max_pool1d(emb_other_item_ids_item.permute(0,2,1), kernel_size=emb_other_item_ids_item.size(1)).squeeze(2)
        # pooled_emb_other_item_ids_item = self.other_item_item_att(emb_item, emb_other_item_ids)


        emb_city_platform = self.emb_dict['city_platform'](city_platform)
        # emb_user_id = self.emb_dict['user_id'](user_id)

        # emb_other_items = self.other_item_dense(emb_other_items)

        pooled_other_item_ids = F.max_pool1d(emb_other_item_ids_gru.permute(0,2,1), kernel_size=emb_other_item_ids_gru.size(1)).squeeze(2)

        emb_past_interactions = emb_past_interactions.permute(0,2,1)
        pooled_interaction = F.max_pool1d(emb_past_interactions, kernel_size=self.config.sequence_length).squeeze(2)
        # pooled_interaction = self.att(emb_past_interactions)
        
        emb_past_interactions_sess = torch.cat( [emb_past_interactions_sess, emb_past_actions_sess], dim=2)

        # self_att_mask =  get_attn_key_pad_mask(past_interactions_sess, past_interactions_sess, self.config.transformed_dummy_item)
        # att_past_interactions_sess = self.att_sess(emb_past_interactions_sess, emb_past_interactions_sess, emb_past_interactions_sess, self_att_mask)
        # pooled_att_past_interactions_sess = torch.mean(att_past_interactions_sess, dim=1, keepdim=False)

        # print(att_past_interactions_sess.shape, emb_past_interactions_sess.shape)
        emb_past_interactions_sess , _ = self.gru_sess(emb_past_interactions_sess)

        # att_past_interactions_sess = self.att_sess(emb_past_interactions_sess)
        # att_interaction_sess = F.max_pool1d(att_past_interactions_sess.permute(0,2,1), kernel_size=self.config.sess_length).squeeze(2)

        emb_past_interactions_sess = emb_past_interactions_sess.permute(0,2,1)
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess, kernel_size=self.config.sess_length).squeeze(2)
        # pooled_interaction_sess = torch.nn.ReLU()(pooled_interaction_sess)
        

        item_interaction =  emb_item * pooled_interaction
        item_last_item = emb_item * emb_last_item
        item_last_click_item = emb_item * emb_last_click_item
        imp_last_idx = emb_impression_index * emb_last_interact_index
        # other_item_ids_interaction_sess =  pooled_other_item_ids * pooled_interaction_sess
        
        # pooled_emb_other_item_ids_pooled_interaction_sess = self.other_item_item_att(pooled_interaction_sess, emb_other_item_ids)

        emb_list = [emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, emb_star]
        emb_concat = torch.cat(emb_list, dim=1)
        
        sum_squared = torch.pow( torch.sum( emb_concat, dim=1) , 2).unsqueeze(1)
        squared_sum = torch.sum( torch.pow( emb_concat, 2) , dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        
        squared_cont = torch.pow(cont_features, 2)
#         cont = self.cont_linear(cont_features)

        
        concat = torch.cat([emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, item_interaction, item_last_item, emb_star, pooled_interaction_sess, emb_last_click_item, emb_last_click_impression, emb_last_interact_index, item_last_click_item, imp_last_idx, pooled_other_item_ids, emb_city_platform] , dim=1)
        concat = self.bn(concat)
        
        hidden = torch.nn.ReLU()(self.hidden1(concat))

        hidden = torch.cat( [cont_features, hidden, sum_squared, squared_sum, second_order, squared_cont, neighbor_prices] , dim=1)
        
        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))
        
        # target_item = emb_item * hidden
        # other_item = emb_other_item_ids*  hidden.unsqueeze(1).repeat(1,24,1)
        # other_item = F.max_pool1d(other_item.permute(0,2,1), kernel_size=24).squeeze(2)

        
        
        # hidden = self.dropout(hidden)

        output = torch.sigmoid(self.output(hidden)).squeeze()
        
        
        return output
