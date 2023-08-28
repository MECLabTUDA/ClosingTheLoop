import numpy as np
import pickle
import os
import cv2
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from config import seresnext50_128_2class_config as conf

# https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed
class Attention(nn.Module):
    
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PEDataset(Dataset):
    
    def __init__(self,
                 feature_array,
                 image_to_feature,
                 series_dict,
                 image_dict,
                 series_list,
                 seq_len):
        self.feature_array=feature_array
        self.image_to_feature=image_to_feature
        self.series_dict=series_dict
        self.image_dict=image_dict
        self.series_list=series_list
        self.seq_len=seq_len
        
    def __len__(self):
        return len(self.series_list)
    
    def __getitem__(self,index):
        """ Loads the earlier extracted features, labels and mask

        Args:
            index (int): index to be loaded 

        Returns:
            _type_: the extracted features, the labels, the mask, the information about
            negative_exam_for_pe
            and the information contained as list
        """
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
        if len(image_list)>self.seq_len:
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            y_pe = np.zeros((len(image_list), 1), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
            x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
            y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation = cv2.INTER_LINEAR))
        else:
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            y_pe = np.zeros((self.seq_len,), dtype=np.float32)
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  
                y_pe[i] = self.image_dict[image_list[i]]['pe_present_on_image']
        x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        y_pe = torch.tensor(y_pe, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        y_npe = self.series_dict[self.series_list[index]]['negative_exam_for_pe']
        return x, y_pe, mask, y_npe, self.series_list[index]

# prepare input
with open(conf['series_list_train'], 'rb') as f:
    series_list_train = pickle.load(f)
with open(conf['series_list_valid'], 'rb') as f:
    series_list_valid = pickle.load(f) 
with open(conf['image_list_train'], 'rb') as f:
    image_list_train = pickle.load(f)
with open(conf['image_list_valid'], 'rb') as f:
    image_list_valid = pickle.load(f) 
with open(conf['image_dict'], 'rb') as f:
    image_dict = pickle.load(f) 
with open(conf['series_dict'], 'rb') as f:
    series_dict = pickle.load(f)
feature_train = np.load(conf['feature_train'])
feature_valid = np.load(conf['feature_valid'])
print(feature_train.shape, feature_valid.shape, len(series_list_train), len(series_list_valid), len(image_list_train), len(image_list_valid), len(image_dict), len(series_dict))

image_to_feature_train = {}
image_to_feature_valid = {}
for i in range(len(feature_train)):
    image_to_feature_train[image_list_train[i]] = i
for i in range(len(feature_valid)):
    image_to_feature_valid[image_list_valid[i]] = i

loss_weight_dict = {
                     'negative_exam_for_pe': 0.5,
                     'pe_present_on_image': 0.5,
                   }


seed = 2001
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# hyperparameters
seq_len =conf['seq_len']
feature_size = conf['feature_size']
lstm_size =conf['lstm_size']
learning_rate = conf['learning_rate']
batch_size = conf['batch_size']
num_epoch = conf['num_epoch']

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class PENet(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear_pe = nn.Linear(lstm_size*2, 1)
        self.last_linear_npe = nn.Linear(lstm_size*4, 1)
        self.attention = Attention(lstm_size*2, seq_len)
        
    def forward(self, x, mask):
        #x = SpatialDropout(0.5)(x)
        h_lstm1, _ = self.lstm1(x)
        #avg_pool = torch.mean(h_lstm2, 1)
        logits_pe = self.last_linear_pe(h_lstm1)
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)  
        logits_npe = self.last_linear_npe(conc)
        return logits_pe, logits_npe

model = PENet(input_len=feature_size, lstm_size=lstm_size)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
criterion1 = nn.BCEWithLogitsLoss().cuda()


# training

# iterator for training
train_datagen = PEDataset(feature_array=feature_train,
                          image_to_feature=image_to_feature_train,
                          series_dict=series_dict,
                          image_dict=image_dict,
                          series_list=series_list_train,
                          seq_len=seq_len)
train_generator = DataLoader(dataset=train_datagen,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True)
valid_datagen = PEDataset(feature_array=feature_valid,
                          image_to_feature=image_to_feature_valid,
                          series_dict=series_dict,
                          image_dict=image_dict,
                          series_list=series_list_valid,
                          seq_len=seq_len)
valid_generator = DataLoader(dataset=valid_datagen,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)


for ep in range(num_epoch):

    # train
    losses_pe = AverageMeter()
    losses_npe = AverageMeter()
    model.train()
    for j, (x, y_pe, mask, y_npe, series_list) in enumerate(train_generator):

        loss_weights_pe = np.zeros((x.size(0), x.size(1)), dtype=np.float32)
        for n in range(len(series_list)):
            image_list = series_dict[series_list[n]]['sorted_image_list']
            num_positive = 0
            for m in range(len(image_list)):
                num_positive += image_dict[image_list[m]]['pe_present_on_image']
            positive_ratio = num_positive / len(image_list)
            adjustment = 0
            if len(image_list)>seq_len:
                adjustment = len(image_list)/seq_len
            else:
                adjustment = 1.
            loss_weights_pe[n,:] = loss_weight_dict['pe_present_on_image']*positive_ratio*adjustment
        loss_weights_pe = torch.tensor(loss_weights_pe, dtype=torch.float32).cuda()

        x = x.cuda()
        y_pe = y_pe.float().cuda()
        mask = mask.cuda()
        y_npe = y_npe.float().cuda()
        logits_pe, logits_npe = model(x, mask)
        loss_pe = criterion(logits_pe.squeeze(),y_pe)
        loss_pe = loss_pe*mask*loss_weights_pe
        loss_pe = loss_pe.sum()/mask.sum()
        loss_npe = criterion1(logits_npe.view(-1),y_npe)*loss_weight_dict['negative_exam_for_pe']
        losses_pe.update(loss_pe.item(), mask.sum().item())
        losses_npe.update(loss_npe.item(), x.size(0))
        loss = loss_pe + loss_npe

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    scheduler.step()

    print()
    print('epoch: {}, train_loss_pe: {}, train_loss_npe: {}'.format(ep, losses_pe.avg, losses_npe.avg), flush=True)

    # valid
    pred_prob_list = []
    gt_list = []
    loss_weight_list = []
    series_len_list = []
    id_list = []

    losses_pe = AverageMeter()
    losses_npe = AverageMeter()
    model.eval()
    for j, (x, y_pe, mask, y_npe, series_list) in enumerate(valid_generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(valid_generator)-1:
                end = len(valid_generator.dataset)

            for n in range(len(series_list)):
                gt_list.append(series_dict[series_list[n]]['negative_exam_for_pe'])
                loss_weight_list.append(loss_weight_dict['negative_exam_for_pe'])
                id_list.append(series_list[n].split('_')[0]+'_negative_exam_for_pe')
                image_list = series_dict[series_list[n]]['sorted_image_list']
                num_positive = 0
                for m in range(len(image_list)):
                    num_positive += image_dict[image_list[m]]['pe_present_on_image']
                positive_ratio = num_positive / len(image_list)
                for m in range(len(image_list)):
                    gt_list.append(image_dict[image_list[m]]['pe_present_on_image'])
                    loss_weight_list.append(loss_weight_dict['pe_present_on_image']*positive_ratio)
                series_len_list.append(len(gt_list))
                id_list += list(series_dict[series_list[n]]['sorted_image_list'])

            loss_weights_pe = np.zeros((x.size(0), x.size(1)), dtype=np.float32)
            for n in range(len(series_list)):
                image_list = series_dict[series_list[n]]['sorted_image_list']
                num_positive = 0
                for m in range(len(image_list)):
                    num_positive += image_dict[image_list[m]]['pe_present_on_image']
                positive_ratio = num_positive / len(image_list)
                adjustment = 0
                if len(image_list)>seq_len:
                    adjustment = len(image_list)/seq_len
                else:
                    adjustment = 1.
                loss_weights_pe[n,:] = loss_weight_dict['pe_present_on_image']*positive_ratio*adjustment
            loss_weights_pe = torch.tensor(loss_weights_pe, dtype=torch.float32).cuda()

            x = x.cuda()
            y_pe = y_pe.float().cuda()
            mask = mask.cuda()
            y_npe = y_npe.float().cuda()
            logits_pe, logits_npe = model(x, mask)
            loss_pe = criterion(logits_pe.squeeze(),y_pe)
            loss_pe = loss_pe*mask*loss_weights_pe
            loss_pe = loss_pe.sum()/mask.sum()
            loss_npe = criterion1(logits_npe.view(-1),y_npe)*loss_weight_dict['negative_exam_for_pe']
            losses_pe.update(loss_pe.item(), mask.sum().item())
            losses_npe.update(loss_npe.item(), x.size(0))

            pred_prob_pe = np.squeeze(logits_pe.sigmoid().cpu().data.numpy())
            pred_prob_npe = np.squeeze(logits_npe.sigmoid().cpu().data.numpy())
            for n in range(len(series_list)):
                pred_prob_list.append(pred_prob_npe[n])
                num_image = len(series_dict[series_list[n]]['sorted_image_list'])
                if num_image>seq_len:
                    pred_prob_list += list(np.squeeze(cv2.resize(pred_prob_pe[n, :], (1, num_image), interpolation = cv2.INTER_LINEAR)))
                else:
                    pred_prob_list += list(pred_prob_pe[n, :num_image])

    pred_prob_list = torch.tensor(pred_prob_list, dtype=torch.float32)
    gt_list = torch.tensor(gt_list, dtype=torch.float32)
    loss_weight_list = torch.tensor(loss_weight_list, dtype=torch.float32)
    print(len(pred_prob_list), len(series_len_list))
    print(series_len_list[:5])
    print(series_len_list[-5:])
    kaggle_loss = torch.nn.BCELoss(reduction='none')(pred_prob_list, gt_list)
    kaggle_loss = (kaggle_loss*loss_weight_list).sum() / loss_weight_list.sum()

    print()
    print('epoch: {}, valid_loss_pe: {}, valid_loss_npe: {}, kaggle_loss: {}'.format(ep, losses_pe.avg, losses_npe.avg, kaggle_loss), flush=True)
    print()


out_dir = conf['result_out_dir']
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.save(out_dir+'pred_prob_list_seresnext50_128', np.array(pred_prob_list))
np.save(out_dir+'gt_list_seresnext50_128', np.array(gt_list))
np.save(out_dir+'loss_weight_list_seresnext50_128', np.array(loss_weight_list))
np.save(out_dir+'series_len_list', np.array(series_len_list))
np.save(out_dir+'id_list', np.array(id_list))

out_dir = conf['model_out_dir']
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
torch.save(model.state_dict(), out_dir+'seresnext50_128_2class')
