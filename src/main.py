from config import *
from data import *
from utils import *
from constant import *
from rnn import *
from ncf import *
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import slack
import os
from torchviz import make_dot
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
import pytz 


client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])



model_name = 'ncf_xnn_time_diff_v2'


torch.backends.cudnn.deterministic = True
seed_everything(42)

configuration = NCFConfiguration()

# now = pytz.timezone('Asia/Taipei').localize(datetime.now())
# logdir = f'../runs/{now.month}-{now.day} {now.hour}:{now.minute} {configuration.device_id}'


os.environ["CUDA_VISIBLE_DEVICES"] = str(configuration.device_id)
print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

if configuration.sub_sample:
    model_name += '_140k'
else:
    model_name += '_all'

if configuration.use_test:
    model_name += '_ut'

if configuration.debug:
    model_name += '_db'

model_name += f'_{configuration.device_id}'

# model_name += f'_{now.month}-{now.day}-{now.hour}-{now.minute}'
weight_path = f"../weights/{model_name}.model"




print(configuration.get_attributes())
# with open(f'../input/ncf_xnn_int_diff_v2_all_ut_0_dg.p','rb') as f:
#     data_gen = pickle.load( f)  

# data_gen.config.begin_iter = configuration.begin_iter
# data_gen.config.learning_rate = configuration.learning_rate
# data_gen.config.device_id = configuration.device_id
# data_gen.config.max_iters = configuration.max_iters
# configuration = data_gen.config

data_gen = NCFDataGenerator(configuration)



print(configuration.get_attributes())



valid_data = data_gen.val_data
train_data= data_gen.train_data



if configuration.use_cuda:
    net = NCF(configuration).cuda()
else:
    net = NCF(configuration)

optim = use_optimizer(net, configuration)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',min_lr=0.0005, factor=0.7, verbose=True)
print(net)

def get_prediction(loader, net):
    net.eval()
    all_scores = []
    validation_loss = []
    for batch_id, data in enumerate(loader):
        with torch.no_grad():
            item_ids = Variable(data[0]).to(device=device_type)
            targets = Variable(data[1]).to(device=device_type)
            past_interactions = Variable(data[2]).to(device=device_type)

            past_interaction_masks = (data[3])

            price_rank = Variable(data[4]).to(device=device_type)
            city = Variable(data[5]).to(device=device_type)
            last_item =  Variable(data[6]).to(device=device_type)
            impression_index = Variable(data[7]).to(device=device_type)
            continuous_features = Variable(data[8]).to(device=device_type)

            star = Variable(data[9]).to(device=device_type)
            
            past_interactions_sess = Variable(data[10]).to(device=device_type)
            past_actions_sess = Variable(data[11]).to(device=device_type)

            
            last_click_item = Variable(data[12]).to(device=device_type)
            last_click_impression = Variable(data[13]).to(device=device_type)
            last_interact_index = Variable(data[14]).to(device=device_type)
            neighbor_prices = Variable(data[15]).to(device=device_type)
            other_item_ids = Variable(data[16]).to(device=device_type)
            city_platform = Variable(data[17]).to(device=device_type)

            prediction = net(item_ids, past_interactions, past_interaction_masks, price_rank, city, last_item, impression_index, continuous_features, star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, neighbor_prices, other_item_ids, city_platform)
            loss = crit(prediction,targets).item()
            prediction = prediction.detach().cpu().numpy().tolist()
            all_scores += prediction
            validation_loss.append(loss)
    validation_loss = np.mean(validation_loss)
    return all_scores, validation_loss

def evaluate_valid(val_loader, val_df, net ):
    
            
    val_df['score'], val_loss = get_prediction(val_loader, net)
    # train_data['score'], _ = get_prediction(train_loader, net)
    # pickle.dump( train_data, open(f'../output/{model_name}_train_prediction.p','wb'))
    # pickle.dump( val_df, open(f'../output/{model_name}_val_prediction.p','wb'))
    
    grouped_val = val_df.groupby('session_id')
    rss = []
    rss_group = {i:[] for i in range(1,26)}
    incorrect_session = {}
    for session_id, group in grouped_val:
        
        scores = group['score']
        sorted_arg = np.flip(np.argsort(scores))

        if group['label'].values[sorted_arg][0] != 1:
            incorrect_session[session_id] = (sorted_arg.values, group['label'].values[sorted_arg])

        rss.append( group['label'].values[sorted_arg])
        rss_group[len(group)].append(group['label'].values[sorted_arg])

    mrr = compute_mean_reciprocal_rank(rss)
    mrr_group = {i:(len(rss_group[i]), compute_mean_reciprocal_rank(rss_group[i])) for i in range(1,26)}
    # print(mrr_group)
    pickle.dump( incorrect_session, open(f'../output/{model_name}_val_incorrect_order.p','wb'))

    return mrr, mrr_group, val_loss



device_type='cuda'



crit = configuration.loss()


best_mrr = 0
early_stopping = configuration.early_stopping
not_improve_round = 0
val_loader = data_gen.evaluate_data_valid()
test_loader =data_gen.instance_a_test_loader()
train_loader = data_gen.instance_a_train_loader()
n_iter = 0
stopped = False
for i in range(configuration.num_epochs):
    
    
    net.train()
    for batch_id, data in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        n_iter += 1

        item_ids = Variable(data[0]).to(device=device_type)
        targets = Variable(data[1]).to(device=device_type)
        past_interactions = Variable(data[2]).to(device=device_type)
        
        past_interaction_masks = (data[3])

        # past_interaction_masks = past_interaction_masks[:,:, np.newaxis]
        # past_interaction_masks = [past_interaction_masks] * configuration.categorical_emb_dim
        # past_interaction_masks = np.concatenate(past_interaction_masks, axis=2)
        # past_interaction_masks = torch.FloatTensor(past_interaction_masks)
        # past_interaction_masks = Variable(past_interaction_masks).to(device=device_type)
        
        price_rank = Variable(data[4]).to(device=device_type)
        city = Variable(data[5]).to(device=device_type)
        last_item = Variable(data[6]).to(device=device_type)
        impression_index = Variable(data[7]).to(device=device_type)
        continuous_features = Variable(data[8]).to(device=device_type)
        star = Variable(data[9]).to(device=device_type)
        
        past_interactions_sess = Variable(data[10]).to(device=device_type)
        past_actions_sess = Variable(data[11]).to(device=device_type)
        
        # other_item_impressions = Variable(data[13]).to(device=device_type)
        last_click_item = Variable(data[12]).to(device=device_type)
        last_click_impression = Variable(data[13]).to(device=device_type)
        last_interact_index = Variable(data[14]).to(device=device_type)
        neighbor_prices = Variable(data[15]).to(device=device_type)
        other_item_ids = Variable(data[16]).to(device=device_type)
        city_platform = Variable(data[17]).to(device=device_type)
        prediction = net(item_ids, past_interactions, past_interaction_masks, price_rank, city, last_item, impression_index, continuous_features, star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, neighbor_prices, other_item_ids, city_platform)
        
        loss = crit(prediction,targets)
        loss.backward()
        optim.step()
        
    mrr, mrr_group, val_loss = evaluate_valid(val_loader, valid_data, net)
    if mrr > best_mrr:
        print(f"improve from {best_mrr} to {mrr}")
        best_mrr = mrr
        not_improve_round = 0
        torch.save(net.state_dict(), weight_path)
    else:
        print(f"didn't improve from {best_mrr} to {mrr}")
        not_improve_round += 1
    if not_improve_round >= early_stopping:
        break


net.load_state_dict(torch.load(weight_path))    

# item_embeddings = net.emb_dict['item_id'].weight.detach().cpu().numpy()
# item_ids = data_gen.cat_encoders['item_id'].reverse_transform(np.arange(len(item_embeddings)))

# item_embedding_df = pd.DataFrame(item_embeddings, columns=[ f'nn{i}' for i in range(item_embeddings.shape[1])])
# item_embedding_df['item_id'] = item_ids

# with open(f'../input/{model_name}_ie.p','wb') as f:
#     pickle.dump(item_embedding_df, f)
print("BEST mrr", best_mrr)


# dot = make_dot(prediction, params=dict(net.named_parameters()))
# dot.format='png'
# dot.render('graph')

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
                "text": f"*Features* :\n```{data_gen.get_features()}```"
            }
        ]
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*Model Summary* :\n{net}"
            }
        ]
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*best mrr & loss* :\n {best_mrr}, {val_loss}"
            }
        ]
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*Mrr Group* :\n{mrr_group}"
            }
        ]
    }
  ]) 
    # client.files_upload(file=open('graph.png', 'rb'), filename='graph.png', channels='#recsys2019')


if configuration.debug:
    exit(0)



        
test_df = data_gen.test_data
test_df['score'], _ = get_prediction(test_loader, net)



with open(f'../output/{model_name}_test_score.p', 'wb') as f:
    pickle.dump( test_df.loc[:,['score', 'session_id', 'step']],f, protocol=4)
    
grouped_test = test_df.groupby('session_id')
predictions = []
session_ids = []
for session_id, group in grouped_test:
    
    scores = group['score']
    sorted_arg = np.flip(np.argsort(scores))
    sorted_item_ids = group['item_id'].values[sorted_arg]
    sorted_item_ids = data_gen.cat_encoders['item_id'].reverse_transform(sorted_item_ids)
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