import argparse
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import albumentations
import pydicom
import copy
from config import train_config as conf

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
    
    def __init__(self, image_dict, bbox_dict, image_list, target_size, transform):
        self.image_dict=image_dict
        self.bbox_dict=bbox_dict
        self.image_list=image_list
        self.target_size=target_size
        self.transform=transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,index:int):
        """ Loads and returns the given image and bounding boxes"""
        
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

        # Create average bbox for handling non-annotated volumes
        mean_bbox = np.array(list(self.bbox_dict.values())).mean(axis=0)
        bboxes = [self.bbox_dict[self.image_list[index]] if self.image_list[index] in self.bbox_dict else mean_bbox]

        class_labels = ['lung']
        transformed = self.transform(image=x, bboxes=bboxes, class_labels=class_labels)
        x = transformed['image']
        x = x.transpose(2, 0, 1)
        y = transformed['bboxes'][0]
        y = torch.from_numpy(np.array(y))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 2001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    import pickle
    with open(conf['series_list_train'], 'rb') as f:
        series_list_train = pickle.load(f) 
    with open(conf['series_dict_train'], 'rb') as f:
        series_dict = pickle.load(f) 
    with open(conf['image_dict_train'], 'rb') as f:
        image_dict = pickle.load(f)
    df = pd.read_csv(conf['csv_path'])
    bbox_image_id_list = df['Image'].values
    bbox_Xmin_list = df['Xmin'].values
    bbox_Ymin_list = df['Ymin'].values
    bbox_Xmax_list = df['Xmax'].values
    bbox_Ymax_list = df['Ymax'].values
    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        bbox_dict[bbox_image_id_list[i]] = [max(0.0, bbox_Xmin_list[i]), max(0.0, bbox_Ymin_list[i]), min(1.0, bbox_Xmax_list[i]), min(1.0, bbox_Ymax_list[i])]
    image_list_train = []
    for series_id in series_list_train:
        sorted_image_list = series_dict[series_id]['sorted_image_list']
        num_image = len(sorted_image_list)
        selected_idx = [int(0.2*num_image), int(0.3*num_image), int(0.4*num_image), int(0.5*num_image)]
        image_list_train.append(sorted_image_list[selected_idx[0]])
        image_list_train.append(sorted_image_list[selected_idx[1]])
        image_list_train.append(sorted_image_list[selected_idx[2]])
        image_list_train.append(sorted_image_list[selected_idx[3]])
    print(len(image_list_train))

    # hyperparameters
    learning_rate = conf['learning_rate']
    batch_size = conf['batch_size']
    image_size = conf['image_size'] 
    num_polyak = conf['num_polyak'] 
    num_epoch = conf['num_epoch'] 

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = efficientnet()
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.local_rank == 0 and conf['load_model']:
        model.load_state_dict(torch.load(conf['model_state_path'])['model_state_dict'])

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    criterion = nn.L1Loss().to(args.device)

    # training
    train_transform = albumentations.Compose([
        albumentations.Cutout(num_holes=1, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
    ], bbox_params=albumentations.BboxParams(format='albumentations', label_fields=['class_labels']))

    # iterator for training
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict, image_list=image_list_train, target_size=image_size, transform=train_transform)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=5, pin_memory=True)

    for ep in range(conf['start_epoch'], num_epoch):
        losses = AverageMeter()
        model.train()
        for j,(images,labels) in enumerate(generator):
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = model(images)
            loss = criterion(logits,labels)
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if args.local_rank == 0:
                if j==len(generator)-num_polyak:
                    averaged_model = copy.deepcopy(model)
                if j>len(generator)-num_polyak:
                    for k in averaged_model.module.state_dict().keys():
                        averaged_model.module.state_dict()[k].data += model.module.state_dict()[k].data
                if j==len(generator)-1:
                    for k in averaged_model.module.state_dict().keys():
                        averaged_model.module.state_dict()[k].data /= num_polyak

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)

        if args.local_rank == 0:
            out_dir = conf['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save({
                'epoch': ep,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'loss': losses
            }, out_dir+'epoch{}'.format(ep))

            torch.save({
                'epoch': ep,
                'model_state_dict': averaged_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'loss': losses
            }, out_dir+'epoch{}_polyak'.format(ep))

if __name__ == "__main__":
    main()
