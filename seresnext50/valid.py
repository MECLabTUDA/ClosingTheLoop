import numpy as np
import os
import cv2
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d
from sklearn.metrics import roc_auc_score
import pydicom
from config import valid_config as conf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

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

def window(img:np.ndarray, WL:int=50, WW:int=350) -> np.ndarray:
    r'''Clips the given image into a [0;255] int value-scale after normalizing it.'''
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

class PEDataset(Dataset):
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,index:int):
        r"""Loads the data for the given index and creates 3 channel image. 
        Returns x as the image within the loaded Bounding Box and y as the corresponding label"""
        
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]

        data1 = pydicom.dcmread(conf['data_base_path']+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_minus1']+'.dcm')

        data2 = pydicom.dcmread(conf['data_base_path']+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        
        data3 = pydicom.dcmread(conf['data_base_path']+study_id+'/'+series_id+'/'+self.image_dict[self.image_list[index]]['image_plus1']+'.dcm')
        
        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array
        x1 = x1*data1.RescaleSlope+data1.RescaleIntercept
        x2 = x2*data2.RescaleSlope+data2.RescaleIntercept
        x3 = x3*data3.RescaleSlope+data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]['series_id']]
        x = x[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(x)
        y = self.image_dict[self.image_list[index]]['pe_present_on_image']
        return x, y

class seresnext50(nn.Module):
    
    def __init__(self ):
        super().__init__()
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

def main():

    # checkpoint list
    checkpoint_list = ['epoch0',]

    # prepare input
    import pickle
    with open(conf['image_list_valid'], 'rb') as f:
        image_list_valid = pickle.load(f) 
    with open(conf['image_dict_valid'], 'rb') as f:
        image_dict = pickle.load(f) 
    with open(conf['bbox_dict_valid'], 'rb') as f:
        bbox_dict_valid = pickle.load(f)
    print(len(image_list_valid), len(image_dict), len(bbox_dict_valid))

    # hyperparameters
    batch_size = conf['batch_size']
    image_size = conf['image_size']
    criterion = nn.BCEWithLogitsLoss().cuda()

    # start validation
    for ckp in checkpoint_list:

        # build model
        model = seresnext50()
        model.load_state_dict(torch.load(conf['model_state_path']+ckp)['model_state_dict'])
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.eval()

        pred_prob = np.zeros((len(image_list_valid),),dtype=np.float32)

        # iterator for validation
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_valid, image_list=image_list_valid, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        losses = AverageMeter()
        for i, (images, labels) in enumerate(generator):
            with torch.no_grad():
                start = i*batch_size
                end = start+batch_size
                if i == len(generator)-1:
                    end = len(generator.dataset)
                images = images.cuda()
                labels = labels.float().cuda()
                logits = model(images)
                loss = criterion(logits.view(-1),labels)
                losses.update(loss.item(), images.size(0))
                pred_prob[start:end] = np.squeeze(logits.sigmoid().cpu().data.numpy())

        label = np.zeros((len(image_list_valid),),dtype=int)        
        for i in range(len(image_list_valid)):
            label[i] = image_dict[image_list_valid[i]]['pe_present_on_image']
        auc = roc_auc_score(label, pred_prob)

        print("checkpoint {} ...".format(ckp))
        print('loss:{}, auc:{}'.format(losses.avg, auc), flush=True)
        print()

if __name__ == "__main__":
    main()
