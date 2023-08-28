import numpy as np
import pandas as pd
import os
import cv2
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from efficientnet_pytorch import EfficientNet
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
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

class PEDataset(Dataset):
    
    def __init__(self, image_dict, bbox_dict, image_list, target_size):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,index):
        """ Loads and returns the given image and the labeled bounding box"""
        study_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[0]
        series_id = self.image_dict[self.image_list[index]]['series_id'].split('_')[1]
        data = pydicom.dcmread(conf['data_base_path']+study_id+'/'+series_id+'/'+self.image_list[index]+'.dcm')
        x = data.pixel_array.astype(np.float32)
        x = x*data.RescaleSlope+data.RescaleIntercept
        x1 = window(x, WL=100, WW=700)
        x = np.zeros((x1.shape[0], x1.shape[1], 3), dtype=np.float32)
        x[:,:,0] = x1
        x[:,:,1] = x1
        x[:,:,2] = x1
        x = cv2.resize(x, (self.target_size,self.target_size))
        x = x.transpose(2, 0, 1)

        # Create average bbox for handling non-annotated volumes
        mean_bbox = np.array(list(self.bbox_dict.values())).mean(axis=0)
        bboxes = [self.bbox_dict[self.image_list[index]] if self.image_list[index] in self.bbox_dict else mean_bbox]

        y = torch.from_numpy(np.array(bboxes))
        return x, y

class efficientnet(nn.Module):
    
    def __init__(self ):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.net._fc.in_features
        self.last_linear = nn.Linear(in_features, 4)
        
    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

def main():

    # checkpoint list
    checkpoint_list = [
                       'epoch0',
                       'epoch1',
                       'epoch2',
                       'epoch3',
                       'epoch4',
                       'epoch5',
                       'epoch6',
                       'epoch7',
                       'epoch8',
                       'epoch9',
                       'epoch10',
                       'epoch11',
                       'epoch12',
                       'epoch13',
                       'epoch14',
                       'epoch15',
                       'epoch16',
                       'epoch17',
                       'epoch18',
                       'epoch19',
                      ]

    # prepare input
    import pickle
    with open(conf['series_list_valid'], 'rb') as f:
        series_list_valid = pickle.load(f) 
    with open(conf['series_dict'], 'rb') as f:
        series_dict = pickle.load(f) 
    with open(conf['image_dict_valid'], 'rb') as f:
        image_dict = pickle.load(f)
    #df = pd.read_csv('../lung_bbox.csv')
    df = pd.read_csv(conf['csv_path'])
    bbox_image_id_list = df['Image'].values
    bbox_Xmin_list = df['Xmin'].values
    bbox_Ymin_list = df['Ymin'].values
    bbox_Xmax_list = df['Xmax'].values
    bbox_Ymax_list = df['Ymax'].values
    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        bbox_dict[bbox_image_id_list[i]] = [max(0.0, bbox_Xmin_list[i]), max(0.0, bbox_Ymin_list[i]), min(1.0, bbox_Xmax_list[i]), min(1.0, bbox_Ymax_list[i])]
    image_list_valid = []
    for series_id in series_list_valid:
        sorted_image_list = series_dict[series_id]['sorted_image_list']
        num_image = len(sorted_image_list)
        selected_idx = [int(0.2*num_image), int(0.3*num_image), int(0.4*num_image), int(0.5*num_image)]
        image_list_valid.append(sorted_image_list[selected_idx[0]])
        image_list_valid.append(sorted_image_list[selected_idx[1]])
        image_list_valid.append(sorted_image_list[selected_idx[2]])
        image_list_valid.append(sorted_image_list[selected_idx[3]])

    # hyperparameters
    batch_size = conf['batch_size']
    image_size = conf['image_size']
    criterion = nn.L1Loss().cuda()

    # start validation
    for ckp in checkpoint_list:

        # build model
        model = efficientnet()
        model.load_state_dict(torch.load(conf['model_state_path']+ckp+'_polyak')['model_state_dict'])
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.eval()

        pred_bbox = np.zeros((len(image_list_valid),4),dtype=np.float32)

        # iterator for validation
        datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict, image_list=image_list_valid, target_size=image_size)
        generator = DataLoader(dataset=datagen, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

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
                loss = criterion(logits,labels)
                losses.update(loss.item(), images.size(0))
                pred_bbox[start:end] = np.squeeze(logits.cpu().data.numpy())

        print("checkpoint {} ...".format(ckp))
        print('loss:{}'.format(losses.avg), flush=True)
        print()

if __name__ == "__main__":
    main()
