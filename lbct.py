# %%
    # Load the pretrained weights
# import torch 
# pretrained_weights = torch.load('/home/nada.saadi/CTPET/hecktor2022_cropped/4centers-ctonly/4centers-ctonly.pth')

# %%
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
# import wandb

import monai
from monai.losses import DiceCELoss, DiceFocalLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai import transforms

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    ScaleIntensityd,
    #AddChanneld,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ConcatItemsd,
    AdjustContrastd, 
    Rand3DElasticd,
    HistogramNormalized,
    NormalizeIntensityd,
    Invertd,
    SaveImage,

)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNETR, SegResNet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai import data


from monai.utils import first, set_determinism
from sklearn.model_selection import train_test_split
import json


from monai.transforms import apply_transform
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import DataLoader, Dataset, decollate_batch
from tqdm import tqdm
import json
import torch

# %%


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch 
pretrained_weights = torch.load('/home/nada.saadi/CTPET/hecktor2022_cropped/4centers-ctonly/4centers-ctonly.pth')
model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072, 
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# %%
data_dir='/home/nada.saadi/MIS-FM/hecktor2022_cropped'
#json_dir= '/home/nada.saadi/CTPET/hecktor2022_cropped/test_ct_only_mda.json'
json_dir='/home/nada.saadi/CTPET/hecktor2022_cropped/simplifiedct_json_file.json'

# %%
import json

json_file_path = "/home/nada.saadi/CTPET/hecktor2022_cropped/simplifiedct_json_file.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)

    num_elements = len(data)

    print("Number of elements in the JSON file:", num_elements)
first_5_elements = data[:5] 
print(first_5_elements)


# %%
def datafold_read(datalist, basedir, fold=0, key=""):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

# %%
# train_files, validation_files = datafold_read(datalist=json_dir, basedir=data_dir)
# len(train_files), len(validation_files)

# %%
class ClipCT(MapTransform):
    """
    Convert labels to multi channels based on hecktor classes:
    label 1 is the tumor
    label 2 is the lymph node

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "ct":
                d[key] = torch.clip(d[key], min=-200, max=200)
            # elif key == "pt":
            #     d[key] = torch.clip(d[key], d[key].min(), 5)
        return d

class MulPTFM(MapTransform):
    """
    Mult PT and FM 

    """

    def __call__(self, data):
        d = dict(data)

        fm = d["ct"] > 0
        d["pt"] = d["pt"] * fm
        return d

class SelectClass(MapTransform):
    """
    Select the class for which you want to fine tune the model 

    """
    # def __init__(self, keys, cls=1):
    #     super(self).__init__(keys)
    #     self.cls = cls

    def __call__(self, data):
        d = dict(data)
        d["seg"][d["seg"] == 1] = 0
        # d["seg"][d["seg"] == 2] = 1
        
        return d

# %%
def datafold_read(datalist, basedir, fold=0):
    with open(datalist) as f:
        json_data = json.load(f)

    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold and "key" not in d:
            for k in d:
                if isinstance(d[k], list):
                    d[k] = [os.path.join(basedir, iv) for iv in d[k]]
                elif isinstance(d[k], str):
                    d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
            val.append(d)

    return val

# %%
val_transforms = Compose(
    [
        LoadImaged(keys=["ct",  "seg"], ensure_channel_first = False),
        SpatialPadd(keys=["ct",  "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct",  "seg"], axcodes="PLS"),
        #NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct",], minv=0, maxv=1),
        #MulPTFM(keys=["ct","pt"]),
        ConcatItemsd(keys=[ "ct"], name="ct"),
    ]
)
def create_dataloader(data, transforms, batch_size=2, shuffle=True):
    # Create CacheDataset with the reformatted data
    dataset = Dataset(data=data, transform=transforms)

    # Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
val_loader = create_dataloader(json_dir, val_transforms, shuffle=False)

# %%
torch.backends.cudnn.benchmark = True
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# %%
test_loader = DataLoader(
    dataset=Dataset(data=json_dir, transform=val_transforms),
    batch_size=1,  # Batch size for testing can be 1 since no backpropagation is required
    shuffle=False,
    num_workers=4
)

oss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

# Dice metric for evaluation
dice_metric_fn = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch_fn = DiceMetric(include_background=False, reduction="mean_batch")

def testing():
    model.eval()
    dice_metric_fn.reset()
    dice_metric_batch_fn.reset()
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            test_inputs, test_labels = batch_data["ct"].cuda(), batch_data["seg"].cuda()
            
            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), 4, model)
            
            # Convert outputs and labels to one-hot format for DiceMetric calculation
            test_outputs = [AsDiscrete(argmax=True, to_onehot=3)(i) for i in decollate_batch(test_outputs)]
            test_labels = [AsDiscrete(to_onehot=3)(i) for i in decollate_batch(test_labels)]

            dice_metric_fn(y_pred=test_outputs, y=test_labels)
            dice_metric_batch_fn(y_pred=test_outputs, y=test_labels)

    mean_dice_test = dice_metric_fn.aggregate().item()
    metric_batch_test = dice_metric_batch_fn.aggregate()
    metric_tumor = metric_batch_test[0].item()
    metric_lymph = metric_batch_test[1].item()

    print(f"Testing - Avg Dice: {mean_dice_test:.4f}, Tumor Dice: {metric_tumor:.4f}, Lymph Dice: {metric_lymph:.4f}")
    return mean_dice_test, metric_tumor, metric_lymph
    # Save the weights
save_path = '/home/nada.saadi/CTPET/hecktor2022_cropped/lowerbound_ct_saved_weights.pth'
torch.save(model.state_dict(), save_path)
print(f"Model weights saved at: {save_path}")


testing()

# %%


# %%
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from tqdm import tqdm
# Assuming `Dataset` and `DiceCELoss` are defined elsewhere, as well as `model`, `json_dir`, and `val_transforms`

# Initialize test_loader with the dataset of the fifth center
test_loader = DataLoader(
    dataset=Dataset(data=json_dir, transform=val_transforms),  # Make sure `json_dir` points to the fifth center's data
    batch_size=1,
    shuffle=False,
    num_workers=4
)

# Define loss function, optimizer, and post-processing transforms
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

# Initialize Dice metrics for evaluation
dice_metric_fn = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch_fn = DiceMetric(include_background=False, reduction="mean_batch")

def load_and_test_model(weights_path):
    # Load model weights
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode

    # Reset metrics
    dice_metric_fn.reset()
    dice_metric_batch_fn.reset()

    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            test_inputs, test_labels = batch_data["ct"].cuda(), batch_data["seg"].cuda()
            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), 4, model)

            # Convert outputs and labels to one-hot format
            test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]

            # Update metrics
            dice_metric_fn(y_pred=test_outputs, y=test_labels)
            dice_metric_batch_fn(y_pred=test_outputs, y=test_labels)

    # Aggregate and print results
    mean_dice_test = dice_metric_fn.aggregate().item()
    metric_batch_test = dice_metric_batch_fn.aggregate()
    metric_tumor = metric_batch_test[0].item()
    metric_lymph = metric_batch_test[1].item()
    print(f"Testing with weights: {weights_path} - Avg Dice: {mean_dice_test:.4f}, Tumor Dice: {metric_tumor:.4f}, Lymph Dice: {metric_lymph:.4f}")

# Paths to the saved weights of the 4 centers
weights_paths = '/home/nada.saadi/CTPET/hecktor2022_cropped/4centers-ctonly/4centers-ctonly.pth'

# Loop through each set of weights and test
# for path in weights_paths:
#     load_and_test_model(path)
load_and_test_model(weights_paths)

# %%



