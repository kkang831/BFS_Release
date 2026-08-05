import copy
from typing import Type, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from peft.tuners.lora.layer import Linear, BaseTunerLayer

def set_patch_lora_mask(model: torch.nn.Module, lora_name, lora_mask, kv_lora_mask = None):
    """ Set mask for LoRA layers"""

    model = model.transformer if hasattr(model, "transformer") else model
    qo_lora_mask = torch.tensor(lora_mask, dtype=torch.bool)
    kv_lora_mask = qo_lora_mask if kv_lora_mask is None else torch.tensor(kv_lora_mask, dtype=torch.bool)

    for name, module in model.named_modules():
        if isinstance(module, Linear) and lora_name in module.active_adapters:
            if not hasattr(module, "lora_mask"):
                module.lora_mask = dict()
            if "attn1n.to_k" in name or "attn1n.to_v" in name:
                module.lora_mask[lora_name] = kv_lora_mask
            else:
                module.lora_mask[lora_name] = qo_lora_mask
    return model


def set_unicon_config(model: torch.nn.Module, k, v):
    """ Update joint cross attention configurations in patched modules """

    model = model.transformer if hasattr(model, "transformer") else model
    model.unicon_config[k] = v
    return model
