import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections.abc import Sequence

from unetr import CustomedUNETR

import json
from tqdm.autonotebook import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

print(torch.cuda.device_count())
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from monai.transforms import EnsureTyped
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, ConcatItemsd
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceFocalLoss, FocalLoss
from monai.inferers import sliding_window_inference


from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


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

# %%
data_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped'
json_dir = '/home/nada.saadi/MIS-FM/hecktor2022_cropped/MDA_CTPT_TRAIN.json'

# %%
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


train_data, valid_data = datafold_read(datalist=json_dir, basedir=data_dir, fold=0)
print(len(train_data), len(valid_data))

# %%
num_samples = 4

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

train_transforms = Compose(
    [
        LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first=True),
        SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
        NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),  # Concatenate CT and PET
        RandCropByPosNegLabeld(
            keys=["ctpt", "seg"],
            label_key="seg",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="ctpt",
            image_threshold=0,
        ),
        RandFlipd(keys=["ctpt", "seg"], spatial_axis=[0], prob=0.20),
        RandFlipd(keys=["ctpt", "seg"], spatial_axis=[1], prob=0.20),
        RandFlipd(keys=["ctpt", "seg"], spatial_axis=[2], prob=0.20),
        RandRotate90d(keys=["ctpt", "seg"], prob=0.20, max_k=3),
        EnsureTyped(keys=["ctpt", "seg"]),
    
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first=True),
        SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
        NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),# Concatenate CT and PET
        EnsureTyped(keys=["ctpt", "seg"]),
    ]
)


##----------

def create_dataloader(data, transforms, batch_size=2, shuffle=True):
    # Create CacheDataset with the reformatted data
    dataset = Dataset(data=data, transform=transforms)

    # Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)



train_loader = create_dataloader(train_data, train_transforms, shuffle=True)
val_loader = create_dataloader(valid_data, val_transforms, shuffle=False)

# %%



model = CustomedUNETR(
    in_channels=1,  # Number of input channels
    out_channels=3,  # Number of output channels
    img_size=(96, 96, 96),  # Size of the input image
    feature_size=48,  # Size of the feature maps
    hidden_size=768,
    num_heads = 12,# Size of the hidden layers in the transformer
    mlp_dim=3072,  # Dimension of the MLP in the transformer
    pos_embed="perceptron",  # Type of positional embedding
    norm_name="instance",  # Type of normalization
    res_block=True,  # Whether to use residual blocks
    dropout_rate=0.0,
    proj_type= "conv",
    r=4,
    lora_layer=None,
    
).to(device) 


# %%
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

model_dir='/home/nada.saadi/CTPET/hecktor2022_cropped/module2'

max_iterations = 18000
eval_num = 100

## running stats
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
metric_values_tumor = []
metric_values_lymph = []


loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")


##-----------------------------------------------------------------------------

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
            logit_map = model(x, mode="mix")
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
                        model.state_dict(), os.path.join(model_dir, "m2.pth")
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


# %%
epoch = 0 # used for LR scheduler
max_num_epochs = 530 # used for LR scheduler


while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
    epoch += 1
    #optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_num_epochs, 0.005676 , 0.9)
# model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_luck_UNETr_prompt.pth")))


