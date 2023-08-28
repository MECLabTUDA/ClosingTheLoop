import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from efficientnet_pytorch import EfficientNet
import pydicom
import glob
from config import save_bbox_train_config as conf
# need to import class definition, otherwise loading checkpoints will

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

def window(img:np.ndarray, WL:int=50, WW:int=350) -> np.ndarray:
    r'''Clips the given image into a [0;255] int value-scale after normalizing it.'''    
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return x

class BboxDataset(Dataset):
    
    def __init__(self, series_list):
        self.series_list = series_list
        
    def __len__(self):
        return len(self.series_list)
    
    def __getitem__(self,index):
        return index

class BboxCollator(object):
    
    def __init__(self, series_list):
        self.series_list = series_list
        
    def _load_dicom_array(self, f:str): 
        """Load dicom files

        Args:
            f (str): dir path containing .dcm files

        Returns:
            loaded and preprocessed dicoms, a list of all dicom files, a list of all selected dicom files
        """
        dicom_files = glob.glob(os.path.join(f, '*.dcm'))
        dicoms = [pydicom.dcmread(d) for d in dicom_files]
        M = np.float32(dicoms[0].RescaleSlope)
        B = np.float32(dicoms[0].RescaleIntercept)
        z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
        sorted_idx = np.argsort(z_pos)
        dicom_files = np.asarray(dicom_files)[sorted_idx]
        dicoms = np.asarray(dicoms)[sorted_idx]
        selected_idx = [int(0.2*len(dicom_files)), int(0.3*len(dicom_files)), int(0.4*len(dicom_files)), int(0.5*len(dicom_files))]
        selected_dicom_files = dicom_files[selected_idx]
        selected_dicoms = dicoms[selected_idx]
        dicoms = np.asarray([d.pixel_array.astype(np.float32) for d in selected_dicoms])
        dicoms = dicoms * M
        dicoms = dicoms + B
        dicoms = window(dicoms, WL=100, WW=700)
        return dicoms, dicom_files, selected_dicom_files
    
    def __call__(self, batch_idx):
        study_id = self.series_list[batch_idx[0]].split('_')[0]
        series_id = self.series_list[batch_idx[0]].split('_')[1]

        series_dir = conf['series_base_path'] + study_id + '/'+ series_id

        dicoms, dicom_files, selected_dicom_files = self._load_dicom_array(series_dir)
        image_list = []
        for i in range(len(dicom_files)):
            name = dicom_files[i][-16:-4]
            image_list.append(name)
        selected_image_list = []
        for i in range(len(selected_dicom_files)):
            name = selected_dicom_files[i][-16:-4]
            selected_image_list.append(name)
        x = np.zeros((4, 3, dicoms.shape[1], dicoms.shape[2]), dtype=np.float32)
        for i in range(4):
            x[i,0] = dicoms[i]
            x[i,1] = dicoms[i]
            x[i,2] = dicoms[i]
        return torch.from_numpy(x), image_list, selected_image_list, self.series_list[batch_idx[0]]

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

    # prepare input
    import pickle
    with open(conf['series_list_train'], 'rb') as f:
        series_list = pickle.load(f) 
    df = pd.read_csv(conf['csv_path'])
    bbox_image_id_list = df['Image'].values
    bbox_Xmin_list = df['Xmin'].values
    bbox_Ymin_list = df['Ymin'].values
    bbox_Xmax_list = df['Xmax'].values
    bbox_Ymax_list = df['Ymax'].values
    bbox_dict = {}
    for i in range(len(bbox_image_id_list)):
        bbox_dict[bbox_image_id_list[i]] = [max(0.0, bbox_Xmin_list[i]), max(0.0, bbox_Ymin_list[i]), min(1.0, bbox_Xmax_list[i]), min(1.0, bbox_Ymax_list[i])]

    # build model
    model = efficientnet()
    model.load_state_dict(torch.load(conf['model_state_path'])['model_state_dict'])
    model = model.cuda()
    model.eval()

    pred_bbox = np.zeros((len(series_list)*4,4),dtype=np.float32)
    bbox_dict_train = {}
    selected_image_list_train = []

    # iterator for validation
    datagen = BboxDataset(series_list=series_list)
    collate_fn = BboxCollator(series_list=series_list)
    generator = DataLoader(dataset=datagen, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
    total_steps = len(generator)
    for i, (images, image_list, selected_image_list, series_id) in tqdm(enumerate(generator), total=total_steps):
        with torch.no_grad():
            start = i*4
            end = start+4
            if i == len(generator)-1:
                end = len(generator.dataset)*4
            images = images.cuda()
            logits = model(images)
            bbox = np.squeeze(logits.cpu().data.numpy())
            pred_bbox[start:end] = bbox
            selected_image_list_train += list(selected_image_list)
            xmin = np.round(min([bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]])*512)
            ymin = np.round(min([bbox[0,1], bbox[1,1], bbox[2,1], bbox[3,1]])*512)
            xmax = np.round(max([bbox[0,2], bbox[1,2], bbox[2,2], bbox[3,2]])*512)
            ymax = np.round(max([bbox[0,3], bbox[1,3], bbox[2,3], bbox[3,3]])*512)
            bbox_dict_train[series_id] = [int(max(0, xmin)), int(max(0, ymin)), int(min(512, xmax)), int(min(512, ymax))]

    # Create average bbox for handling non-annotated volumes
    mean_bbox = np.array(list(bbox_dict.values())).mean(axis=0)

    not_annotated_count = 0
    annotated_count = 0

    total_loss = 0
    for i in range(len(series_list)*4):
        for j in range(4):
            # Check if bbox was annotated for volume
            if selected_image_list_train[i] in bbox_dict.keys():
                total_loss += abs(pred_bbox[i,j]-bbox_dict[selected_image_list_train[i]][j])
                annotated_count = annotated_count + 1
            else:
                total_loss += abs(pred_bbox[i,j]-mean_bbox[j])
                not_annotated_count = not_annotated_count + 1
                print("Not annotated: ", selected_image_list_train[i])
    total_loss = total_loss / len(series_list) / 4 / 4
    print("Total elements not annotated: ", not_annotated_count)
    print("Total elements annotated: ", annotated_count)
    print("total loss: ", total_loss)

    with open(conf['bbox_out_path'], 'wb') as f:
        pickle.dump(bbox_dict_train, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
