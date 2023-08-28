import argparse
import numpy as np
import os
import cv2
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import albumentations
import pydicom
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
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
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

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
        x = self.transform(image=x)['image']
        x = x.transpose(2, 0, 1)
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
    with open(conf['image_list_train'], 'rb') as f:
        image_list_train = pickle.load(f) 
    with open(conf['image_dict_train'], 'rb') as f:
        image_dict = pickle.load(f) 
    with open(conf['bbox_dict_train'], 'rb') as f:
        bbox_dict_train = pickle.load(f) 
    print(len(image_list_train), len(image_dict), len(bbox_dict_train))

    # hyperparameters
    batch_size = conf['batch_size']
    image_size = conf['image_size']
    num_epoch = conf['num_epoch']
    learning_rate = conf['learning_rate']
    
    # Resume training
    resume_training = conf['resume_training']
    checkpoint_path = conf['checkpoint_path']

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = seresnext50()
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    num_train_steps = int(len(image_list_train)/(batch_size*2)*num_epoch)   # 2 GPUs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    # training
    train_transform = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2, p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0)
    ])

    # iterator for training
    datagen = PEDataset(image_dict=image_dict, bbox_dict=bbox_dict_train, image_list=image_list_train, target_size=image_size, transform=train_transform)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, sampler=sampler, batch_size=batch_size, num_workers=5, pin_memory=True)


    # Load model
    if resume_training:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        amp.load_state_dict(checkpoint['amp'])
        start_epoch = checkpoint['epoch'] + 1
        loaded_losses = checkpoint['loss']
        start_step = 0
        print("Resumed training in epoch {}, step {}. Checkpoint was {}".format(start_epoch, start_step, checkpoint_path))
    else:
        start_epoch = 0
        start_step = 0

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j,(images,labels) in enumerate(generator):

            if args.local_rank == 0:
                out_dir = conf['model_out_dir']
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                    
            images = images.to(args.device)
            labels = labels.float().to(args.device)

            logits = model(images)
            loss = criterion(logits.view(-1),labels)
            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()

            if((j+1)% 1000 == 0):
                # current date and time
                now = datetime.now().isoformat(" ", "seconds")
                print("{} | Made {}/{} steps in epoch {} | train_loss: {}".format(now, (j+1), len(generator), ep, losses.avg))
                if args.local_rank == 0:
                    out_dir = conf['out_dir'] 
                    torch.save({
                        'epoch': ep,
                        'step': j,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': losses
                    }, out_dir+'epoch{}_step{}'.format(ep, j))

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
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses
            }, out_dir+'epoch{}'.format(ep))

if __name__ == "__main__":
    main()
