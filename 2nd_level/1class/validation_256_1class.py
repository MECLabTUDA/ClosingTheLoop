from datetime import datetime
import numpy as np
import pickle
import cv2
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score
from config import validation_256_1class_config as conf

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


class PEDataset(Dataset):
    """Transforms image embeddings of a series into Mx2048 series embeddings by resizing if N > M or zero-padding if N < M. The final series embedding encodes the current image embedding and the difference between the current and 2 direct neighboring image embeddings, resulting in a Mx6144 series embedding."""
    
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
        """ Loads the image, labels and mask

            Args:
                index (int): index to be loaded 

            Returns:
                _type_: the image, the mask, the information about
                negative_exam_for_pe            
                and the information contained as list
        """
        image_list = self.series_dict[self.series_list[index]]['sorted_image_list'] 
        # Handle case where number of series images exceeds sequence length
        if len(image_list)>self.seq_len:
            x = np.zeros((len(image_list), self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.ones((self.seq_len,), dtype=np.float32)
            # Resize image features to Mx2048
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]] 
            x = cv2.resize(x, (self.feature_array.shape[1]*3, self.seq_len), interpolation = cv2.INTER_LINEAR)
        else:
            x = np.zeros((self.seq_len, self.feature_array.shape[1]*3), dtype=np.float32)
            mask = np.zeros((self.seq_len,), dtype=np.float32)
            # Use zero-padding for N < M
            for i in range(len(image_list)):      
                x[i,:self.feature_array.shape[1]] = self.feature_array[self.image_to_feature[image_list[i]]]
                mask[i] = 1.  

        # Calculate difference embeddings between current image and 2 direct neighbors and concatenate with current features
        x[1:,self.feature_array.shape[1]:self.feature_array.shape[1]*2] = x[1:,:self.feature_array.shape[1]] - x[:-1,:self.feature_array.shape[1]]
        x[:-1,self.feature_array.shape[1]*2:] = x[:-1,:self.feature_array.shape[1]] - x[1:,:self.feature_array.shape[1]]

        x = torch.tensor(x, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Count indeterminate cases as negative
        y_npe = max(self.series_dict[self.series_list[index]]['negative_exam_for_pe'], self.series_dict[self.series_list[index]]['indeterminate'])
        return x, mask, y_npe, self.series_list[index]


class PENet(nn.Module):
    
    def __init__(self, input_len, lstm_size, seq_len):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear_npe = nn.Linear(lstm_size*4, 1)
        self.attention = Attention(lstm_size*2, seq_len)
        
    def forward(self, x, mask):
        h_lstm1, _ = self.lstm1(x)
        max_pool, _ = torch.max(h_lstm1, 1)
        att_pool = self.attention(h_lstm1, mask)
        conc = torch.cat((max_pool, att_pool), 1)  
        logits_npe = self.last_linear_npe(conc)
        return logits_npe


def main():

    # checkpoint list
    checkpoint_list = ['seresnext50_256_1class',]

    # prepare input
    with open(conf['series_list_valid'], 'rb') as f:
        series_list_valid = pickle.load(f) 
    with open(conf['image_list_valid'], 'rb') as f:
        image_list_valid = pickle.load(f) 
    with open(conf['image_dict'], 'rb') as f:
        image_dict = pickle.load(f) 
    with open(conf['series_dict'], 'rb') as f:
        series_dict = pickle.load(f)
    feature_valid = np.load(conf['feature_valid'])

    image_to_feature_valid = {}
    for i in range(len(feature_valid)):
        image_to_feature_valid[image_list_valid[i]] = i

    # set seeds
    seed = 2001
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # hyperparameters   
    seq_len = conf['seq_len'] 
    feature_size = conf['feature_size'] 
    lstm_size = conf['lstm_size'] 
    learning_rate = conf['learning_rate'] 
    batch_size = conf['batch_size'] 

    # start validation
    for ckp in checkpoint_list:

        # build model
        model = PENet(input_len=feature_size, lstm_size=lstm_size, seq_len=seq_len)
        model.load_state_dict(torch.load(conf['model_state']+ckp))
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


        # iterator for validation
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


        # Volume-wise
        pe_volume_pred = []
        pe_volume_gt = []
        
        model.eval()
        for i, (x, mask, y_npe, series_list) in enumerate(valid_generator):
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(valid_generator)-1:
                    end = len(valid_generator.dataset) 

            x = x.cuda()
            mask = mask.cuda()
            y_npe = y_npe.float().cuda()
            logits_npe = model(x, mask)
            pred_prob_npe = np.squeeze(logits_npe.sigmoid().cpu().data.numpy())
            
            for n in range(len(series_list)):
                pe_volume_gt.append(y_npe[n])
                pe_volume_pred.append(pred_prob_npe[n]) 
                

        pe_volume_pred_list = torch.tensor(pe_volume_pred, dtype=torch.float32)
        pe_volume_gt_list = torch.tensor(pe_volume_gt, dtype=torch.float32)
        print(len(pe_volume_pred), len(pe_volume_gt))


        # Get best image-level decision boundary
        precision, recall, thresholds = precision_recall_curve(pe_volume_gt_list, pe_volume_pred_list)
        f1_score = (2 * precision * recall) / (precision + recall)
        best_thres_ix = np.argmax(f1_score)

        print("Best threshold: ", thresholds[best_thres_ix])

        # Threshold image-level predicitions
        thresholded = np.copy(pe_volume_pred_list)

        thresholded[thresholded < thresholds[best_thres_ix]] = 0
        thresholded[thresholded >= thresholds[best_thres_ix]] = 1


        # Calculate volume-wise metrics for PE negative
        auc = roc_auc_score(pe_volume_gt_list, pe_volume_pred_list)
        accuracy = accuracy_score(pe_volume_gt_list, thresholded)
        confusion = confusion_matrix(pe_volume_gt_list, thresholded)
        tn, fp, fn, tp = confusion.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("checkpoint {} ...".format(ckp))
        now = datetime.now().isoformat(" ", "seconds")
        print("Time: {}".format(now))
        print("Results for single class validation (not PE)")
        print('ROC-AUC: {}'.format(auc, flush=True))
        print('accuracy: {} \n precision: {} \n recall: {} \n confusion matrix: {}'.format(accuracy, precision[best_thres_ix], recall[best_thres_ix], confusion), flush=True)
        print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp), flush=True)
        print('sensitivity: {}, specificity: {}'.format(sensitivity, specificity), flush=True)
        print()


if __name__ == "__main__":
    main()