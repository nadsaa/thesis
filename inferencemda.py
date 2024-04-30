# %%
test_ct_only= '/home/nada.saadi/CTPET/hecktor2022_cropped/test_ct_only_mda.json'
test_pet_only= '/home/nada.saadi/CTPET/hecktor2022_cropped/test_pet_only_mda.json'
test_CTPT= '/home/nada.saadi/CTPET/hecktor2022_cropped/test_CTPT_mda.json'

# %%
Encoder_frozen_PT_weights='/home/nada.saadi/CTPET/hecktor2022_cropped/module2-with-pet-skipconnection-firsthalf/m2-withpet-intheskipconnection-firsthalf.pth'
Encoder_decoder_frozen_PT_weights= '/home/nada.saadi/CTPET/hecktor2022_cropped/module2-with-pet-skipconnection-firsthalf-decoderfrozen/m2-withpet-intheskipconnection-firsthalf-decoderfrozen.pth'

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections.abc import Sequence

from unetr import CustomedUNETR

import json
from tqdm.autonotebook import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

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
model = CustomedUNETR(
    in_channels=1,
    out_channels=3,
    img_size=(96, 96, 96),
    feature_size=48,
    hidden_size=768,
    num_heads=12,
    mlp_dim=3072,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
    proj_type="conv",
    r=4,
    lora_layer=None,
).to(device)

# %%
torch.backends.cudnn.benchmark = True
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

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
val_transforms = Compose(
    [
        LoadImaged(keys=["ct", "seg"], ensure_channel_first = True),
        SpatialPadd(keys=["ct", "seg"], spatial_size=(200, 200, 310), method='end'),
        Orientationd(keys=["ct",  "seg"], axcodes="PLS"),
        #NormalizeIntensityd(keys=["pt"]),
        ClipCT(keys=["ct"]),
        ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
        #MulPTFM(keys=["ct","pt"]),
        #ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
    ]
)

# %%
import torch
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from tqdm import tqdm


# Load the pre-trained weights
pretrained_weights_path = '/home/nada.saadi/CTPET/hecktor2022_cropped/module2-with-pet-skipconnection-firsthalf/m2-withpet-intheskipconnection-firsthalf.pth'
model.load_state_dict(torch.load(pretrained_weights_path), strict=True)

# Assume `valid_data` is your validation dataset loaded from a JSON or other sources
# For demonstration, let's say `valid_data` is available as a list of dictionaries
# Each dictionary contains paths to the 'ct', 'pt', and 'seg' for each validation sample

# Assume `val_transforms` is already defined and includes all necessary preprocessing steps

val_loader = DataLoader(
    dataset=Dataset(data=test_ct_only, transform=val_transforms),
    batch_size=1,  # Batch size for validation can be 1 since no backpropagation is required
    shuffle=False,
    num_workers=4
)

# Dice metric for evaluation
dice_metric_fn = DiceMetric(include_background=False, reduction="mean")
dice_metric_batch_fn = DiceMetric(include_background=False, reduction="mean_batch")

def validation():
    model.eval()
    dice_metric_fn.reset()
    dice_metric_batch_fn.reset()
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            val_inputs, val_labels = batch_data["ct"].cuda(), batch_data["seg"].cuda()
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            
            # Convert outputs and labels to one-hot format for DiceMetric calculation
            val_outputs = [AsDiscrete(argmax=True, to_onehot=3)(i) for i in decollate_batch(val_outputs)]
            val_labels = [AsDiscrete(to_onehot=3)(i) for i in decollate_batch(val_labels)]

            dice_metric_fn(y_pred=val_outputs, y=val_labels)
            dice_metric_batch_fn(y_pred=val_outputs, y=val_labels)

    mean_dice_val = dice_metric_fn.aggregate().item()
    metric_batch_val = dice_metric_batch_fn.aggregate()
    metric_tumor = metric_batch_val[0].item()
    metric_lymph = metric_batch_val[1].item()

    print(f"Validation - Avg Dice: {mean_dice_val:.4f}, Tumor Dice: {metric_tumor:.4f}, Lymph Dice: {metric_lymph:.4f}")

# Run validation
validation()


# %%


# %%
import json

test_ct_only = '/home/nada.saadi/CTPET/hecktor2022_cropped/test_ct_only_mda.json'

with open(test_ct_only) as file:
    data = json.load(file)

first_5_elements = data[:5]
print(first_5_elements)


# %%
from monai.data import Dataset
from tqdm import tqdm

# Assuming test_ct_only is a list of dictionaries as shown in your JSON snippet
dataset = Dataset(data=test_ct_only, transform=val_transforms)

# Manually iterate over the dataset and apply transforms
for i in tqdm(range(len(dataset))):
    sample = dataset[i]  # This should not raise an error if everything is correct
    print(sample.keys())  # Should print dict keys including 'ct' and 'seg'


# %%
sample = dataset[0]
print(sample)


# %%
simple_transforms = Compose([LoadImaged(keys=["ct", "seg"], ensure_channel_first=True)])
simple_dataset = Dataset(data=test_ct_only, transform=simple_transforms)
sample = simple_dataset[0]
print(sample)



