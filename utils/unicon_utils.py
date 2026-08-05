import torch
from patch import patch_utils as patch

def set_unicon_config(transformer, input_len, device="cuda", dtype=torch.float16, debug=False):

    xy_lora = "xy_lora"
    yx_lora = "yx_lora"
    xlen = ylen = input_len // 2
    true_mask = [True] * xlen
    false_mask = [False] * xlen
    xy_lora_qo_mask = yx_lora_kv_mask = true_mask + false_mask
    xy_lora_kv_mask = yx_lora_qo_mask = false_mask + true_mask
    patch.set_patch_lora_mask(transformer, xy_lora, xy_lora_qo_mask, kv_lora_mask = xy_lora_kv_mask)
    patch.set_patch_lora_mask(transformer, yx_lora, yx_lora_qo_mask, kv_lora_mask = yx_lora_kv_mask)

    if debug:
        print("Set", xy_lora, xy_lora_qo_mask, xy_lora_kv_mask)
        print("Set", yx_lora, yx_lora_qo_mask, yx_lora_kv_mask)
        
    x_ids = torch.arange(xlen).to(device)
    y_ids = torch.arange(xlen, input_len).to(device)
    x_weights = torch.ones([1,1,1]).to(device).to(dtype)
    y_weights = torch.ones([1,1,1]).to(device).to(dtype)

    attn_config = x_ids, y_ids, x_weights, y_weights
    patch.set_unicon_config(transformer, "attn_config", attn_config)
    if debug:
        print("Set attn_config", attn_config)
    
    return transformer

# def set_unicon_config_inference(unet, input_pairs, use_cfg = True, input_len = None, device = "cuda", debug=False):
#     """ Set config for unicon inference.
#         It does two things:
#         1. Tell the model how to pair the inputs for joint cross attention.
#             The input_pairs shoule be:
#             [
#                 (x_0, y_0, wx_0, wy_0, model_name_0),
#                 (x_1, y_1, wx_1, wy_1, model_name_1),
#                 ...
#             ]
#             A simple example is [(0,1,1.0,1.0,"depth")], which means the model will pair your 1st and 2nd input (count in batch dimension) for the joint cross attention of depth model. And the attention output will be scaled by 1.0 for both inputs.
#         2. Set active LoRA adapters and their masks.
#             According to input pairs, we can determine which LoRA adapters to use. Then set their masks so that the adapters can selectively apply to the inputs.
#             In above simple example [(0,1,1.0,1.0,"depth")], we need masks for depth_y_lora, depth_xy_lora, depth_yx_lora.
#             As x has index 0 and y has index 1, depth_y_lora mask is [False, True].
#             In the joint cross attention, the input is:
#                 [x,y] -> Q
#                 [y,x] -> K,V
#             So depth_xy_lora mask is (Q: [True, False], K,V: [False, True]) and depth_yx_lora mask is (Q: [False, True], K,V: [True, False]), so that depth_xy_lora apply to x and depth_yx_lora apply to y.
#     """

#     attn_config = list(zip(*input_pairs)) 
#     for i in range(len(attn_config) - 1):
#         attn_config[i] = torch.tensor(attn_config[i])
#     x_ids, y_ids, x_weights, y_weights, model_names = attn_config
#     x_weights = x_weights.view(-1, 1, 1)
#     y_weights = y_weights.view(-1, 1, 1)

#     if use_cfg:
#         x_ids = torch.cat([x_ids, x_ids + input_len])
#         y_ids = torch.cat([y_ids, y_ids + input_len])
#         x_weights, y_weights = torch.cat([x_weights] * 2), torch.cat([y_weights] * 2)
#         model_names = model_names * 2
#         # batch_size *= 2
    
#     cond_masks = dict()
#     false_mask = [False] * len(model_names)

#     input_len = input_len * 2 if use_cfg else input_len
    
#     active_adapters = get_adapter_names(unet, set(model_names))
#     set_unicon_infer_adapters(unet, active_adapters)

    
#     for cur_cond in set(model_names):
#         cond_masks[cur_cond] = [model_name == cur_cond for model_name in model_names]
#         xy_lora = f"{cur_cond}_xy_lora"
#         yx_lora = f"{cur_cond}_yx_lora"
#         xy_lora_qo_mask = yx_lora_kv_mask = cond_masks[cur_cond] + false_mask
#         xy_lora_kv_mask = yx_lora_qo_mask = false_mask + cond_masks[cur_cond]
#         patch.set_patch_lora_mask(unet, xy_lora, xy_lora_qo_mask, kv_lora_mask = xy_lora_kv_mask)
#         patch.set_patch_lora_mask(unet, yx_lora, yx_lora_qo_mask, kv_lora_mask = yx_lora_kv_mask)
        
#         if debug:
#             print("Set", xy_lora, xy_lora_qo_mask, xy_lora_kv_mask)
#             print("Set", yx_lora, yx_lora_qo_mask, yx_lora_kv_mask)
#         y_lora = f"{cur_cond}_y_lora"
#         if y_lora in active_adapters:
#             cur_y_ids = y_ids[cond_masks[cur_cond]]
#             y_lora_mask = [True if i in cur_y_ids else False for i in range(input_len)]
#             # y_lora_mask = torch.zeros(input_len).to(torch.bool)
#             # for y_id in y_ids[cond_masks[cur_cond]]:
#             #     y_lora_mask[y_id] = True
#             patch.set_patch_lora_mask(unet, y_lora, y_lora_mask)
#             if debug:
#                 print("Set", y_lora, y_lora_mask)
            

#     attn_config = x_ids.to(device), y_ids.to(device), x_weights.to(device), y_weights.to(device)
#     unet.unicon_config["attn_config"] = attn_config
#     unet.unicon_config["cond_masks"] = cond_masks
#     if debug:
#         print("Set attn_config", attn_config)
#         print("Set cond masks", cond_masks)
    
#     return unet
