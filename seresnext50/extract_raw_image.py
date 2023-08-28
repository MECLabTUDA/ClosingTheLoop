from __future__ import print_function
from datetime import datetime
import numpy as np
import pandas as pd
import os
import cv2
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from pretrainedmodels.senet import se_resnext50_32x4d
import pydicom
import re
from config import extract_raw_image_config as conf

# Saleciency maps
from medcam import medcam

# dicom writing
import SimpleITK as sitk
import time, os, glob


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


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
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X * 255.0).astype("uint8")
    return X


def mask_image_multiply(mask, image):
    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if components_per_pixel == 1:
        return mask * image
    else:
        return sitk.Compose([mask * sitk.VectorIndexSelectionCast(image, channel) for channel in range(components_per_pixel)])


def alpha_blend(image1, image2, alpha=0.5, mask1=None, mask2=None):
    """
    Alpha blend two images, pixels can be scalars or vectors.
    The region that is alpha blended is controled by the given masks.
    """

    if not mask1:
        mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
        mask1.CopyInformation(image1)
    else:
        mask1 = sitk.Cast(mask1, sitk.sitkFloat32)
    if not mask2:
        mask2 = sitk.Image(image2.GetSize(), sitk.sitkFloat32) + 1
        mask2.CopyInformation(image2)
    else:
        mask2 = sitk.Cast(mask2, sitk.sitkFloat32)

    components_per_pixel = image1.GetNumberOfComponentsPerPixel()
    if components_per_pixel > 1:
        img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
        img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
    else:
        img1 = sitk.Cast(image1, sitk.sitkFloat32)
        img2 = sitk.Cast(image2, sitk.sitkFloat32)

    intersection_mask = mask1 * mask2

    intersection_image = mask_image_multiply(
        alpha * intersection_mask, img1
    ) + mask_image_multiply((1 - alpha) * intersection_mask, img2)
    return (
        intersection_image
        + mask_image_multiply(mask2 - intersection_mask, img2)
        + mask_image_multiply(mask1 - intersection_mask, img1)
    )


def write_dicom(
    input_directory_with_DICOM_series,
    output_directory,
    content_raw,
    pred_scores=None,
    box=None,
    mode="map",
    threshold=None,
):

    # Read the original series. First obtain the series file names using the
    # image series reader.
    data_directory = input_directory_with_DICOM_series[0]
    print(data_directory)
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(data_directory))
    print(series_IDs, os.path.basename(os.path.normpath(data_directory)))
    if not series_IDs:
        print(
            'ERROR: given directory "'
            + data_directory
            + '" does not contain a DICOM series.'
        )
        series_IDs = (str(os.path.basename(os.path.normpath(data_directory))),)
        print("fixed", series_IDs)

    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        data_directory, series_IDs[0]
    )

    series_file_names = glob.glob(data_directory + "*.dcm")
    print("Read Image",sitk.ReadImage(series_file_names[0],imageIO="GDCMImageIO"))
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.SetImageIO("GDCMImageIO")

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()
    # Print avialable Dicom Tags:
    # for key in series_reader.GetMetaDataKeys(0):
    #    print("\"{0}\":\"{1}\"".format(key, series_reader.GetMetaData(0,key)))

    # Modify the image (blurring)
    # filtered_image = sitk.DiscreteGaussian(image3D)

    if mode == "map":
        overlay = content_raw
    elif mode in ["full", "outer"]:
        overlay = np.zeros_like(content_raw)
        if threshold == None:
            overlay[:] = pred_scores.reshape(-1, 1, 1)
        else:
            overlay[pred_scores > threshold, :, :] = 1.0
        if mode == "outer":
            overlay[
                :,
                box[1] : box[3],
                box[0] : box[2],
            ] = 0
        overlay[0, 0, 0] = 2.0
    elif mode == "dot":
        overlay = np.zeros_like(content_raw)
        if threshold == None:
            overlay[:, 20:110, -110:-20] = pred_scores.reshape(-1, 1, 1)
        else:
            overlay[pred_scores > threshold, 20:110, -110:-20] = 1.0
        overlay[0, 0, 0] = 2.0
    elif "bar":
        overlay = np.zeros_like(content_raw)
        for i in range(overlay.shape[0]):
            until = int((overlay.shape[2] - 60) * pred_scores[i])
            overlay[i, 30:100, 30 : 30 + until] = 1
        overlay[:, 20:110, 28:32] = 1
        overlay[:, 20:110, -32:-28] = 1
        overlay[0, 0, 0] = 2.0

    filtered_image = sitk.GetImageFromArray(overlay)
    filtered_image.CopyInformation(image3D)
    filtered_image = sitk.ScalarToRGBColormap(
        filtered_image,
        sitk.ScalarToRGBColormapImageFilter.Jet,
        #sitk.ScalarToRGBColormapImageFilter.Red,
        # useInputImageExtremaForScaling=False,
    )
    
    # Combine the color overlay volume with the spatial structure volume using alpha blending
    # and cast to a three component vector 8 bit unsigned int. We can readily incorporate a
    # mask into the blend (pun intended). By adding mask2=roi we can limit the overlay to
    # the region of interest.
    mask2 = np.zeros_like(content_raw)
    if mode == "map":
        mask2[content_raw > 0.05] = 1
    else:
        mask2[:] = 1
    mask2 = sitk.GetImageFromArray(mask2)
    mask2.CopyInformation(image3D)
    img_255 = sitk.Cast(
        sitk.IntensityWindowing(
            image3D,
            windowMinimum=-600,
            windowMaximum=1500,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )
    combined_volume = sitk.Cast(
            sitk.Compose(img_255, img_255, img_255),# filtered_image, mask2=mask2
        sitk.sitkVectorUInt8,
    )
    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are also
    # accessible).
    tags_to_copy = [
        "0010|0010",  # Patient Name
        "0010|0020",  # Patient ID
        "0010|0030",  # Patient Birth Date
        "0020|000D",  # Study Instance UID, for machine consumption
        "0020|0010",  # Study ID, for human consumption
        "0020|0013",  # Instance Number
        "0008|0020",  # Study Date
        "0008|0030",  # Study Time
        "0008|0050",  # Accession Number
        "0008|0060",  # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:

    direction = combined_volume.GetDirection()
    series_tag_values = [
        (k, series_reader.GetMetaData(0, k))
        for k in tags_to_copy
        if series_reader.HasMetaDataKey(0, k)
    ] + [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        # ("0020|000e", image3D.get), # Series Instance UID
        (
            "0020|000e",
            series_reader.GetMetaData(0, "0020|000e") + "",
        ),  # Series Instance UID
        (
            "0020|000d",
            series_reader.GetMetaData(0, "0020|000d") + "",
        ),  # Series Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],  # Image Orientation (Patient)
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
    ]  # Series Description
    # ("0028|2000",)("0008|103e", series_reader.GetMetaData(0,"0008|103e") + " Processed-SimpleITK")] # Series Description
    print(output_directory)
    for i in range(combined_volume.GetDepth()):
        image_slice = combined_volume[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # for key in series_reader.GetMetaDataKeys(i):
        #    image_slice.SetMetaData(key, series_reader.GetMetaData(i,key))
        # Slice specific tags.
        image_slice.SetMetaData(
            "0008|0012", time.strftime("%Y%m%d")
        )  # Instance Creation Date
        image_slice.SetMetaData(
            "0008|0013", time.strftime("%H%M%S")
        )  # Instance Creation Time
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(
                map(str, combined_volume.TransformIndexToPhysicalPoint((0, 0, i)))
            ),
        )  # Image Position (Patient)
        image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.

        # writer.SetFileName(os.path.join(output_directory, str(i) + ".dcm"))
        # writer.Execute(image_slice)
    return 0


class PEDataset(Dataset):
    def __init__(
        self,
        image_dict,
        bbox_dict,
        image_list,
        target_size,
        base_path= conf['base_path'],
    ):
        self.image_dict = image_dict
        self.bbox_dict = bbox_dict
        self.image_list = image_list
        self.target_size = target_size
        self.base_path = base_path


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):        
        r"""Loads the data for the given index and creates 3 channel image. 
        Returns x as the image within the loaded Bounding Box, the study id, the data path and the Bounding Box itself"""
        study_id = self.image_dict[self.image_list[index]]["series_id"].split("_")[0]
        series_id = self.image_dict[self.image_list[index]]["series_id"].split("_")[1]

        data1 = pydicom.dcmread(
            self.base_path
            + study_id
            + "/"
            + series_id
            + "/"
            + self.image_dict[self.image_list[index]]["image_minus1"]
            + ".dcm"
        )

        data2 = pydicom.dcmread(
            self.base_path
            + study_id
            + "/"
            + series_id
            + "/"
            + self.image_list[index]
            + ".dcm"
        )

        data3 = pydicom.dcmread(
            self.base_path
            + study_id
            + "/"
            + series_id
            + "/"
            + self.image_dict[self.image_list[index]]["image_plus1"]
            + ".dcm"
        )

        x1 = data1.pixel_array
        x2 = data2.pixel_array
        x3 = data3.pixel_array

        x1 = x1 * data1.RescaleSlope + data1.RescaleIntercept
        x2 = x2 * data2.RescaleSlope + data2.RescaleIntercept
        x3 = x3 * data3.RescaleSlope + data3.RescaleIntercept
        x1 = np.expand_dims(window(x1, WL=100, WW=700), axis=2)
        x2 = np.expand_dims(window(x2, WL=100, WW=700), axis=2)
        x3 = np.expand_dims(window(x3, WL=100, WW=700), axis=2)
        x = np.concatenate([x1, x2, x3], axis=2)
        bbox = self.bbox_dict[self.image_dict[self.image_list[index]]["series_id"]]
        x = x[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        x = cv2.resize(x, (self.target_size, self.target_size))
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(
            x
        )
        # Only return image
        return x, study_id, self.base_path + study_id + "/" + series_id + "/", bbox


class seresnext50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.net.features(x)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return x


def main():
    current_id = None
    current_bbox = None
    current_filepath = None
    current_layer = "net.layer3.5.conv3"
    base_path = conf['base_path']
    attention_map = torch.zeros(0)
    pred_scores = []
    image_vol = torch.zeros(0)

    now = datetime.now().isoformat(" ", "seconds")
    print("Started at {}".format(now))

    # checkpoint list
    checkpoint_list = [
        "epoch0",
    ]

    mode = "bar"  # from [full, outer, bar,dot, map]
    threshold = 0.0  # float or None
    # prepare input
    import pickle

    with open(
        conf['image_list_test'],
        "rb",
    ) as f:
        image_list_test = pickle.load(f)
    with open(
        conf['image_dict_test'],
        "rb",
    ) as f:
        image_dict = pickle.load(f)
    with open(
        conf['bbox_dict_test'],
        "rb",
    ) as f:
        bbox_dict_test = pickle.load(f)

    # r = re.compile('[0-3][a-f0-1][a-f0-9][A-Za-z0-9]+')
    # image_list_test = list(filter(r.match, image_list_test))
    # r = re.compile('[0-3][a-f0-1][acf02359][acf567][A-Za-z0-9]+')
    r = re.compile('01f6c315aabb')
    # r = re.compile('0d5f3216ec1a')
    tmp = []
    for i in image_list_test:
        study_id = image_dict[i]["series_id"].split("_")[0]
        if r.match(study_id) is not None:
            tmp.append(i)
    image_list_test = tmp
    print(len(image_list_test), len(image_dict), len(bbox_dict_test))
    uuids = [x[1] for x in os.walk(base_path)][0]
    mask_ids = [
        True if image_dict[x]["series_id"].split("_")[0] in uuids else False
        for x in image_list_test
    ]
    print(len(list(np.array(image_list_test)[mask_ids])))
    image_list_test = list(np.array(image_list_test)[mask_ids])
    # hyperparameters
    batch_size = conf['hyperparam_batch_size']
    image_size = conf['hyperparameter_image_size']
    criterion = nn.BCEWithLogitsLoss().cuda()
    activation = nn.Sigmoid()
    # start extraction
    for ckp in checkpoint_list:

        # build model
        model = seresnext50()
        model.load_state_dict(
            torch.load(
                conf['model_state_path']
                + ckp
            )["model_state_dict"]
        )
        # Inject model with M3d-CAM
        model = medcam.inject(
            model,
            backend="gcampp",
            output_dir=conf['out_dir'],
            save_maps=False,
            label=lambda x: 0.5 > x,
            layer=current_layer,
            return_attention=True,
        )

        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        # iterator for feature extraction
        datagen = PEDataset(
            image_dict=image_dict,
            bbox_dict=bbox_dict_test,
            image_list=image_list_test,
            target_size=image_size,
            base_path=base_path,
        )
        generator = DataLoader(
            dataset=datagen, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        pred_score = []
        pred_scores = []
        for i, (images, study_id, input_directory, bbox) in enumerate(generator):
            if current_id != study_id[0]:
                if current_id != None:
                    attention_map_final = np.zeros((attention_map.shape[0], 512, 512))
                    attention_map_final[
                        :,
                        current_bbox[1] : current_bbox[3],
                        current_bbox[0] : current_bbox[2],
                    ] = attention_map
                    write_dicom(
                        current_filepath,
                        model.medcam_dict["output_dir"]
                        + f"/overlays/{mode}_{threshold if threshold != None else 'score'}/"
                        + model.medcam_dict["current_layer"]
                        + "/"
                        + str(current_id),
                        attention_map_final,
                        np.array(pred_score),
                        box=bbox,
                        mode=mode,
                        threshold=threshold,
                    )
                    pred_sl = sum([p > 0.5 for p in pred_score])
                    pred = "yes" if pred_sl>1 else "no"
                    pred_scores.append([str(current_id), pred_sl, str(pred)])
                    attention_map = torch.zeros(0)
                    pred_score = []
                if images is None and input_directory is None and bbox is None:
                    continue
                current_id = study_id[0]
                current_bbox = bbox
                current_filepath = input_directory
                model.medcam_dict["current_layer"] = (
                    current_layer + "/" + str(current_id)
                )
                folder = (
                    model.medcam_dict["output_dir"]
                    + f"/overlays/{mode}_{threshold if threshold != None else 'score'}/"
                    + model.medcam_dict["current_layer"]
                )
                os.makedirs(folder, exist_ok=True)
            if images is None and input_directory is None and bbox is None:
                continue
            start = i * batch_size
            end = start + batch_size
            if i == len(generator) - 1:
                end = len(generator.dataset)
            images = images.cuda()
            output, a = model(images)
            pred_score.append(activation(output).detach().cpu().item())
            a = torch.from_numpy(cv2.resize(a.squeeze().detach().cpu().numpy(),((current_bbox[3] - current_bbox[1]).item(),(current_bbox[2] - current_bbox[0]).item(),),)).T.unsqueeze(0)
            if attention_map.size()[0] == 0:
                attention_map = a
            else:
                attention_map = torch.cat([attention_map,a ], dim=0)

            if i == len(image_list_test) - 1:
                    attention_map_final = np.zeros((attention_map.shape[0],512,512))
                    attention_map_final[:,current_bbox[1]:current_bbox[3],current_bbox[0]:current_bbox[2]] = attention_map
                    write_dicom(
                        current_filepath,
                        model.medcam_dict["output_dir"]
                        + f"/overlays/{mode}_{threshold if threshold != None else 'score'}/"
                        + model.medcam_dict["current_layer"]
                        + "/"
                        + str(current_id),
                        attention_map_final,
                        np.array(pred_score),
                        box=bbox,
                        mode=mode,
                        threshold=threshold,
                    )
                    pred_sl = sum([p > 0.5 for p in pred_score])
                    pred = "yes" if pred_sl>1 else "no"
                    pred_scores.append([str(current_id), pred_sl, str(pred)])

            optimizer.zero_grad()
    print(pred_scores)
    out_excel = pd.DataFrame(pred_scores, columns=["StudyInstanceUID", "positively evaluated slices", "postive prediction"])
    print(out_excel)
    out_excel.to_excel(conf['result_path'])


if __name__ == "__main__":
    main()