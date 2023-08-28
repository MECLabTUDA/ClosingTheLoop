import numpy as np
import pandas as pd
import pydicom
import os
from tqdm import tqdm
import glob
import pickle
from config import process_input_split2_config as conf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"


def get_dicom_array(f):
    """Get dicom array from input path

    Args:
        f (str): The path where the dcm files are located

    Returns:
        ndarray: Returns 4 ndarrays containing the dicom files, the z-position, the exposure and thickness ordered by z-position of the image respectively
    """
    dicom_files = glob.glob(os.path.join(f, '*.dcm'))
    dicoms = [pydicom.dcmread(d) for d in dicom_files]
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    sorted_idx = np.argsort(z_pos)
    exposure = [float(d.Exposure) for d in dicoms]
    thickness = [float(d.SliceThickness) for d in dicoms]
    return np.asarray(dicom_files)[sorted_idx], np.asarray(z_pos)[sorted_idx], np.asarray(exposure)[sorted_idx], np.asarray(thickness)[sorted_idx]

df = pd.read_csv(conf['csv_path'])


study_id_list = df['StudyInstanceUID'].values
series_id_list = df['SeriesInstanceUID'].values
image_id_list = df['SOPInstanceUID'].values
pe_present_on_image_list = df['pe_present_on_image'].values
negative_exam_for_pe_list = df['negative_exam_for_pe'].values
qa_motion_list = df['qa_motion'].values
qa_contrast_list = df['qa_contrast'].values
flow_artifact_list = df['flow_artifact'].values
rv_lv_ratio_gte_1_list = df['rv_lv_ratio_gte_1'].values
rv_lv_ratio_lt_1_list = df['rv_lv_ratio_lt_1'].values
leftsided_pe_list = df['leftsided_pe'].values
chronic_pe_list = df['chronic_pe'].values
true_filling_defect_not_pe_list = df['true_filling_defect_not_pe'].values
rightsided_pe_list = df['rightsided_pe'].values
acute_and_chronic_pe_list = df['acute_and_chronic_pe'].values
central_pe_list = df['central_pe'].values
indeterminate_list = df['indeterminate'].values


series_dict = {}
image_dict = {}
series_list = []
for i in tqdm(range(len(series_id_list))):
    series_id = study_id_list[i]+'_'+series_id_list[i]
    image_id = image_id_list[i]
    series_dict[series_id] = {
                              'negative_exam_for_pe': negative_exam_for_pe_list[i],
                              'qa_motion': qa_motion_list[i],
                              'qa_contrast': qa_contrast_list[i],
                              'flow_artifact': flow_artifact_list[i],
                              'rv_lv_ratio_gte_1': rv_lv_ratio_gte_1_list[i],
                              'rv_lv_ratio_lt_1': rv_lv_ratio_lt_1_list[i],
                              'leftsided_pe': leftsided_pe_list[i],
                              'chronic_pe': chronic_pe_list[i],
                              'true_filling_defect_not_pe': true_filling_defect_not_pe_list[i],
                              'rightsided_pe': rightsided_pe_list[i],
                              'acute_and_chronic_pe': acute_and_chronic_pe_list[i],
                              'central_pe': central_pe_list[i],
                              'indeterminate': indeterminate_list[i],
                              'sorted_image_list': [],
                             }
    
    image_dict[image_id] = {
                            'pe_present_on_image': pe_present_on_image_list[i],
                            'series_id': series_id,
                            'z_pos': series_id,
                            'exposure': series_id,
                            'thickness': series_id,
                            'image_minus1': '',
                            'image_plus1': '',
                           }
    series_list.append(series_id)
    
series_list = sorted(list(set(series_list)))
print(len(series_list), len(series_dict), len(image_dict))
for series_id in tqdm(series_dict, total=len(series_dict)):
    series_dir = conf['series_base_path'] + series_id.split('_')[0] + '/'+ series_id.split('_')[1]
    file_list, z_pos_list, exposure_list, thickness_list = get_dicom_array(series_dir)
    image_list = []
    for i in range(len(file_list)):
        name = file_list[i][-16:-4]
        image_list.append(name)
        if i==0:
            image_dict[name]['image_minus1'] = name
            image_dict[name]['image_plus1'] = file_list[i+1][-16:-4]
        elif i==len(file_list)-1:
            image_dict[name]['image_minus1'] = file_list[i-1][-16:-4]
            image_dict[name]['image_plus1'] = name
        else:
            image_dict[name]['image_minus1'] = file_list[i-1][-16:-4]
            image_dict[name]['image_plus1'] = file_list[i+1][-16:-4]
        image_dict[name]['z_pos'] = z_pos_list[i]
        image_dict[name]['exposure'] = exposure_list[i]
        image_dict[name]['thickness'] = thickness_list[i]
    series_dict[series_id]['sorted_image_list'] = image_list   
print(series_dict[series_list[0]])
print(image_dict[image_list[0]])


np.random.seed(100)
np.random.shuffle(series_list)

# Our splits - each split has unique series, they do not seem to overlap with other folds
# Use Fold 0 for validation
valid_df = df.query('outer == 0')

# Use only Fold 3 for training
train_df = df.query('outer == 3')

series_list_train = list(pd.unique(train_df['StudyInstanceUID'] + '_' + train_df['SeriesInstanceUID']))
series_list_valid = list(pd.unique(valid_df['StudyInstanceUID'] + '_' + valid_df['SeriesInstanceUID']))



print(len(series_list_train), len(series_list_valid))
print(series_list_train[:3])
print(series_list_valid[:3])

image_list_train = []
image_list_valid = []
for series_id in series_list_train:
    image_list_train += list(series_dict[series_id]['sorted_image_list'])
for series_id in series_list_valid:
    image_list_valid += list(series_dict[series_id]['sorted_image_list'])
print(len(image_list_train), len(image_list_valid))
print(image_list_train[:3])
print(image_list_valid[:3])

out_dir = conf['out_dir']

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(out_dir+'series_dict.pickle', 'wb') as f:
    pickle.dump(series_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'image_dict.pickle', 'wb') as f:
    pickle.dump(image_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'series_list_train.pickle', 'wb') as f:
    pickle.dump(series_list_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'series_list_valid.pickle', 'wb') as f:
    pickle.dump(series_list_valid, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'image_list_train.pickle', 'wb') as f:
    pickle.dump(image_list_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'image_list_valid.pickle', 'wb') as f:
    pickle.dump(image_list_valid, f, protocol=pickle.HIGHEST_PROTOCOL)