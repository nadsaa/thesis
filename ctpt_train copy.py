import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
torch.cuda.device_count()

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


import torch

set_determinism(seed=1024)
import os
import json
import random


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
    num_samples = 4


# def generate_paths(patient_id):
#     base_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped'
#     center, _ = patient_id.split('-')
#     return {
#         'id': os.path.join(base_dir, center, patient_id),
#         'ct': os.path.join(base_dir, center, patient_id, f"{patient_id}_ct.nii.gz"),
#         'pt': os.path.join(base_dir, center, patient_id, f"{patient_id}_pt.nii.gz"),
#         'seg': os.path.join(base_dir, center, patient_id, f"{patient_id}_gt.nii.gz")
#     }

# # Assuming that all patient IDs are stored in a list
# patient_ids = ["CHUM-002", "CHUP-001", "CHUS-003", "CHUV-004", "HGJ-005", "HMR-006", "MDA-007"]  # Example list
# # Extending the patient_ids list with the specified ranges for each center

# patient_ids = []

# # Adding CHUM patient IDs (from CHUM-001 to CHUM-065)
# exceptions = [3,4,5,9,20,25,28,31,52] 
# patient_ids.extend([f"CHUM-{i:03}" for i in range(1, 66)])

# # Adding CHUP patient IDs (from CHUP-000 to CHUP-075)
# exceptions = [21,31,45]
# patient_ids.extend([f"CHUP-{i:03}" for i in range(0, 76)])

# # Adding CHUS patient IDs (from CHUS-003 to CHUS-101)
# exceptions = [11,12,14,17,18,23,24,25,29,32,34,37,44,54,62,63,70,71,72,75,79,82,84,92,93,99]
# patient_ids.extend([f"CHUS-{i:03}" for i in range(3, 102)])

# # Adding HMR patient IDs (from CHUV-001 to CHUV-053)
# exceptions = [2,3,6,7,8,9,10,14,15,17,18,19,22,26,27,31,32,33,35,36,37,38,39]   # Add the numbers that don't exist as exceptions
# patient_ids.extend([f"HMR-{i:03}" for i in range(1, 41) if i not in exceptions])

# # Adding HGJ patient IDs (from HGJ-007 to HGJ-092)
# exceptions = [9,11,12,14,19,20,21,22,23,24,27,30,33,40,41,42,44,4547,49,51,54,56,59,60,61,63,64,68,75,79,84]
# patient_ids.extend([f"HGJ-{i:03}" for i in range(7, 93)])

# # Adding MDA patient IDs (from MDA-001 to MDA-200)
# exceptions = [2,8,9,200] 
# patient_ids.extend([f"MDA-{i:03}" for i in range(1, 201)])

#  #Adding CHUV patient IDs (from CHUV-001 to CHUV-053)

# patient_ids.extend([f"CHUV-{i:03}" for i in range(1, 54)])

# Assign each data entry to a random fold
# all_data = []
# num_folds = 4
# for patient_id in patient_ids:
#     entry = generate_paths(patient_id)
#     entry['fold'] = random.randint(1, num_folds)
#     all_data.append(entry)

# # Compile data into a JSON structure
# data_json = {"training": all_data}



# random.shuffle(patient_ids)

# # Split into training and validation sets (80% training, 20% validation)
# # Total files = 419 (training) + 105 (validation)
# num_training = 419
# num_validation = 105
# train_patient_ids = patient_ids[:num_training]
# val_patient_ids = patient_ids[num_training:num_training + num_validation]

# # Generate dictionaries for training and validation
# train_dicts = []
# val_dicts = []

# for pid in train_patient_ids:
#     train_dicts.append({
#         'id': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}",
#         'fold': random.randint(0, 3),  # Random fold assignment
#         'ct': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_ct.nii.gz",
#         'pt': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_pt.nii.gz",
#         'seg': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_gt.nii.gz"
#     })

# for pid in val_patient_ids:
#     val_dicts.append({
#         'id': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}",
#         'fold': random.randint(0, 3),  # Random fold assignment
#         'ct': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_ct.nii.gz",
#         'pt': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_pt.nii.gz",
#         'seg': f"/home/nada.saadi/MIS-FM/hecktor2022_cropped/{pid}/{pid}_gt.nii.gz"
#     })

# # Creating JSON data structure
# json_data = {
#     "training": train_dicts,
#     "validation": val_dicts
# }

# # Save as JSON file
# json_file_path = "/home/nada.saadi/MIS-FM/hecktor2022_cropped/train_val_json_new.json"
# with open(json_file_path, 'w') as f:
#     json.dump(json_data, f, indent=4)

# # Outputting the path of the JSON file
# json_file_path

data_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped'
json_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped/train_val_json_new.json'

def datafold_read(datalist, basedir, fold=0, key="training"):
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
train_files, validation_files = datafold_read(datalist=json_dir, basedir=data_dir, fold=0)
print(f'Training Files: {len(train_files)}, Validation Files: {len(validation_files)}')


train_transforms = Compose(
    [
        LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True),
        SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
        NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        #MulPTFM(keys=["ct","pt"]),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
        #NormalizeIntensityd(keys=["ctpt"], channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["ctpt", "seg"],
            label_key="seg",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="ctpt",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["ctpt", "seg"],
            spatial_axis=[0],
            prob=0.20,
        ),
        RandFlipd(
            keys=["ctpt", "seg"],
            spatial_axis=[1],
            prob=0.20,
        ),
        RandFlipd(
            keys=["ctpt", "seg"],
            spatial_axis=[2],
            prob=0.20,
        ),
        RandRotate90d(
            keys=["ctpt", "seg"],
            prob=0.20,
            max_k=3,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True),
        SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
        NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        #MulPTFM(keys=["ct","pt"]),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
    ] 
)
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=6,
    shuffle=True,
    num_workers=8,
    pin_memory=torch.cuda.is_available(),
    )
breakpoint()
val_ds = monai.data.Dataset(data=validation_files, transform=val_transforms)

val_loader = DataLoader(
    val_ds, 
    batch_size=2, 
    num_workers=8, 
    shuffle= False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(img_size= (96, 96, 96),
                  in_channels=2,
                  out_channels=3,
                  depths = (2, 2, 2, 2),
                  num_heads = (3, 6, 12, 24),
                  feature_size = 24,
                  norm_name = "instance",
                  drop_rate = 0.0,
                attn_drop_rate = 0.0,
                dropout_path_rate = 0.0,
                normalize = True,
                use_checkpoint = False,
                spatial_dims = 3,
                downsample="merging").to(device)

torch.backends.cudnn.benchmark = True
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

model_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped'

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["ctpt"].cuda(), batch["seg"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        metric_batch_val = dice_metric_batch.aggregate()

        metric_tumor = metric_batch_val[0].item()
        metric_lymph = metric_batch_val[1].item()

        dice_metric.reset()
        dice_metric_batch.reset()
    return mean_dice_val, metric_tumor, metric_lymph


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["ctpt"].cuda(), batch["seg"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val, metric_tumor, metric_lymph = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            metric_values_tumor.append(metric_tumor)
            metric_values_lymph.append(metric_lymph)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(model_dir, "best_metric_segresnet.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Avg. tumor Dice: {} Current Avg. lymph Dice: {}".format(
                        dice_val_best, dice_val, metric_tumor, metric_lymph
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Avg. tumor Dice: {} Current Avg. lymph Dice: {}".format(
                        dice_val_best, dice_val,  metric_tumor, metric_lymph
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 18000
eval_num = 100

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

epoch = 0
max_num_epochs = 530

global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
metric_values_tumor = []
metric_values_lymph = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
    # wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})
    # wandb.log({'Best Dice': dice_val_best})
    epoch += 1
    #optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_num_epochs, 0.005676 , 0.9)
# model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_luck_UNETr_prompt.pth")))
val_transforms = Compose(
    [
        LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True),
        SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
        ClipCT(keys=["ct"]),
        MulPTFM(keys=["ct","pt"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        #MulPTFM(keys=["ct","pt"]),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
        NormalizeIntensityd(keys=["ctpt"], channel_wise=True),
        RandFlipd(
            keys=["ctpt"],
            spatial_axis=[2],
            prob=1,
        ),
    ]
)

files_ds = monai.data.Dataset(data=validation_files, transform=val_transforms)
print("Total validation cases:", len(files_ds))

val_loader = DataLoader(files_ds, batch_size=1, num_workers=4)


# post_label = AsDiscrete(to_onehot=3)
post_label = AsDiscrete(to_onehot=3)
post_pred = Compose([
    Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=val_transforms,
            orig_keys="ctpt",  # get the previously applied pre_transforms information on the `img` data field,
                              # then invert `pred` based on this information. we can use same info
                              # for multiple fields, also support different orig_keys for different fields
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
    # AsDiscreted(keys="pred", argmax=True, to_onehot=3)
    ])

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")


# model = SegResNet(in_channels=2, 
#                   out_channels=3, 
#                   init_filters=16).to(device)

check_path = "/home/nada.saadi/MIS-FM/hecktor2022_cropped/best_metric_swinmodel_s111_in176_DiceCE_AdamW_lr_1e-4_b4.pth"
output_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped/APredictions-unetr'
model.load_state_dict(torch.load(check_path))
model.eval()

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        
        pat_id = batch['id'][0].split('/')[-1]
        roi_size = (96, 96, 96)
        sw_batch_size = 4

        val_inputs = (batch["ctpt"].cuda())
        batch["pred"] = sliding_window_inference(
            val_inputs, roi_size, sw_batch_size, overlap=0.25,  predictor=model
        )

        val_outputs_list = decollate_batch(batch)
        val_output_convert = [
            post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]


        preds = torch.argmax(val_output_convert[0]['pred'] , dim=0)
        SaveImage(output_dir=output_dir, resample=False, separate_folder=False, output_postfix='seg')(preds)