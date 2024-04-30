from __future__ import annotations
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import deprecated_arg
from collections.abc import Sequence
from torch.nn.parameter import Parameter
from safetensors.torch import save_file
from safetensors import safe_open
from torch import Tensor
import math




__all__ = ["ViT"]


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        
        x = self.w(x) + self.w_b(self.w_a(x)) #self.SCALE# self.w_b(self.w_a(x))
        return x

class LoRA_ViT(nn.Module):

    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        pretrained_weights = '/home/nada.saadi/CTPET/hecktor2022_cropped/Module1-4centers-CT-only-tokens/-module1-4centers-ctonly-tokens.pth'
        state_dict = torch.load(pretrained_weights)

        # Adjust state_dict for patch_embedding layer if needed
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vit."):
                new_key = k[len("vit."):]
                if "patch_embeddings.1" in new_key:
                    new_key = new_key.replace("patch_embeddings.1", "patch_embeddings")
                adjusted_state_dict[new_key] = v

        # Load the adjusted state dictionary
        vit_model.load_state_dict(adjusted_state_dict, strict=True)


# Load the adjusted state dictionary
        #vit_model.load_state_dict(adjusted_state_dict)

        

        assert r > 0
        base_vit_dim = vit_model.blocks[0].attn.proj_q.in_features
        print("base_vit_dim", base_vit_dim)
        dim = base_vit_dim
        if lora_layer: 
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        
        #
        # for param in vit_model.fc.parameters():
        #         param.requires_grad = False
        
        # for param in vit_model.blocks.parameters():
        #     param.requires_grad = False
        
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        save_file(fc_tensors, filename)
    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.fc.in_features
            _out = self.lora_vit.fc.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")
                
                
        #Xavier normal Initiliazation 
    # def reset_parameters(self) -> None:
    #          for w_A in self.w_As:
    #             nn.init.xavier_normal_(w_A.weight)
    #          for w_B in self.w_Bs:
    #             nn.init.xavier_normal_(w_B.weight)
                
        #Xavier uniform Initiliazation
        #def reset_parameters(self) -> None: 
         #    for w_A in self.w_As:
          #       nn.init.xavier_uniform_(w_A.weight)
           #  for w_B in self.w_Bs:
            #     nn.init.xavier_uniform_(w_B.weight)

        #Kaiming normal Initiliazation for A and zero initiliazation for B
    def reset_parameters(self) -> None:
       for w_A in self.w_As:
           nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
       for w_B in self.w_Bs:
           nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


 

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()
        
        patch_in_channels = 1
        
        assert patch_in_channels==1, "Patch in channels must be 1 for now."
        
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=patch_in_channels,
            img_size=img_size,
            patch_size=patch_size,
            num_heads=num_heads,
            hidden_size=hidden_size,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
        )
        TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias, 
                    save_attn=save_attn,
                    
                )
                for _ in range(num_layers)
            ]
        ) 
        self.norm = nn.LayerNorm(hidden_size)
        if classification:
            self.classification_head = nn.Linear(hidden_size, num_classes)

    # def forward(self, x):
    #     ct = x[:, 0, :, :, :].unsqueeze(1)  # Extract CT scan
    #     pet = x[:, 1, :, :, :].unsqueeze(1) # Extract PET scan
        
    #     #print("CT SHAPE", ct.shape)
        
    #     x_ct = self.patch_embedding(ct)  # Convert CT scan to patch embeddings
    #     x_pet = self.patch_embedding(pet)  # Convert PET scan to patch embeddings
    #     #print("/////////////////////////////")
    #     x = torch.cat((x_ct, x_pet), dim=1)  # Concatenate CT and PET patch embeddings
    #     #print("printing The SHAPE of X s", x.shape)
    #     if hasattr(self, "cls_token"):
    #         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    #         x = torch.cat((cls_token, x), dim=1)

    #     hidden_states_out = [] 
    #     for blk in self.blocks:
    #         x = blk(x)
    #         hidden_states_out.append(x)
    #     x = self.norm(x)
    #     if hasattr(self, "classification_head"):
    #         x = self.classification_head(x[:, 0])
            
    #     return x, hidden_states_out

    def forward(self, x):
        # Determine the number of modalities provided based on the input shape
        num_modalities = x.shape[1]
        
        # Process CT scan if available
        if num_modalities > 1:
            ct = x[:, 0, :, :, :].unsqueeze(1)  # Extract CT scan
            x_ct = self.patch_embedding(ct)  # Convert CT scan to patch embeddings
        else:
            x_ct = None
        
        # Always process PET scan (assuming it's always present)
        pet = x[:, -1, :, :, :].unsqueeze(1)  # Extract PET scan, assuming it's the last modality
        x_pet = self.patch_embedding(pet)  # Convert PET scan to patch embeddings
        
        # Concatenate embeddings conditionally based on what's available
        if x_ct is not None:
            x = torch.cat((x_ct, x_pet), dim=1)  # Concatenate CT and PET patch embeddings if both are present
        else:
            x = x_pet  # Use only PET patch embeddings if CT is not available
        
        # Adding class token if exists
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
            
        return x, hidden_states_out

