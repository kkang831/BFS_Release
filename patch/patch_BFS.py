import copy
from typing import Type, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from peft.tuners.lora.layer import Linear, BaseTunerLayer

logger = logging.get_logger(__name__) 


def apply_patch(
        model: torch.nn.Module,
        train = False,
        name_skip = None,
        skip_all = False):

    """
    Patches a diffusion model from diffusers with UniCon.

    Args:
     - model: A top level Stable Diffusion module to patch in place.
     - train: Whether to train the model.
     - name_skip: name for module you do not want to patch UniCon, e.g., name_skip="down_blocks" will skip the UNet encoder.

    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(
        model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    assert is_diffusers, "Only support diffusers model currently."

    diffusion_model = model.transformer if hasattr(model, "transformer") else model

    diffusion_model.unicon_config = {
        "train": train,
    }
    
    if skip_all:
        return model

    for name, module in diffusion_model.named_modules():
        if name_skip is not None and name_skip in name:
            continue
        if isinstance_str(module, "FluxSingleTransformerBlock"):
            make_unicon_block_fn = make_flux_single_unicon_block
            module.__class__ = make_unicon_block_fn(module.__class__)
            module.unicon_config = diffusion_model.unicon_config
            
        if isinstance_str(module, "FluxTransformerBlock"):
            make_unicon_block_fn = make_flux_unicon_block
            module.__class__ = make_unicon_block_fn(module.__class__)
            module.unicon_config = diffusion_model.unicon_config

    return model

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    model = model.transformer if hasattr(model, "transformer") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.__class__ = module._parent
    return model

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


class FluxAttnProcessor2_0_HACK_for_single_block: # original
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape 
        # if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states) #####
        value = attn.to_v(encoder_hidden_states) #####

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # if encoder_hidden_states is not None:
        #     # `context` projections.
        #     encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        #     encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        #     encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        #     encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        #         batch_size, -1, attn.heads, head_dim
        #     ).transpose(1, 2)
        #     encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        #         batch_size, -1, attn.heads, head_dim
        #     ).transpose(1, 2)
        #     encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        #         batch_size, -1, attn.heads, head_dim
        #     ).transpose(1, 2)

        #     if attn.norm_added_q is not None:
        #         encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        #     if attn.norm_added_k is not None:
        #         encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        #     # attention
        #     query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        #     key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        #     value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # from .embeddings import apply_rotary_emb
            from diffusers.models.embeddings import apply_rotary_emb #####
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        return hidden_states

        # if encoder_hidden_states is not None:
        #     encoder_hidden_states, hidden_states = (
        #         hidden_states[:, : encoder_hidden_states.shape[1]],
        #         hidden_states[:, encoder_hidden_states.shape[1] :],
        #     )

        #     # linear proj
        #     hidden_states = attn.to_out[0](hidden_states)
        #     # dropout
        #     hidden_states = attn.to_out[1](hidden_states)

        #     encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        #     return hidden_states, encoder_hidden_states
        # else:
        #     return hidden_states


class FluxAttnProcessor2_0_HACK_for_dual_block: # original
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape 
        # if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states) #####
        value = attn.to_v(encoder_hidden_states) #####

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            # from .embeddings import apply_rotary_emb
            from diffusers.models.embeddings import apply_rotary_emb #####
            
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

        # if encoder_hidden_states is not None:
        #     encoder_hidden_states, hidden_states = (
        #         hidden_states[:, : encoder_hidden_states.shape[1]],
        #         hidden_states[:, encoder_hidden_states.shape[1] :],
        #     )

        #     # linear proj
        #     hidden_states = attn.to_out[0](hidden_states)
        #     # dropout
        #     hidden_states = attn.to_out[1](hidden_states)

        #     encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        #     return hidden_states, encoder_hidden_states
        # else:
        #     return hidden_states


def make_flux_unicon_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a class of UniCon blocks.
    It adds joint cross attention to the forward function and enables related functions for initialzation and training.
    """
    
    class UniConBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        
        def set_joint_layer_requires_grad(self, adapter_names, requires_grad):
            
            for module in self.attn1n.modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                if isinstance(adapter_names, str):
                    adapter_names = [adapter_names]

                # Deactivate grads on the inactive adapter and activate grads on the active adapter
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in adapter_names:
                            # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                            # happen if a completely different adapter layer is being activated.
                            layer.requires_grad_(requires_grad)

            self.conv1n.requires_grad_(requires_grad)

        @property
        def post_joint(self):
            if self.post == "scale":
                return self.scale1n
            elif self.post == "conv" or self.post == "conv_fuse":
                return self.conv1n

        def add_post_joint(self, name, post = "conv", add_bias = False):
            if not hasattr(self, "post1n"):
                self.post1n = nn.ModuleDict({})
                self.post_type = dict()

            if name in self.post1n:
                return

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)
            post_joint = zero_module(conv1n)
        
            self.post1n[name] = post_joint
            self.post_type[name] = post
            
        def initialize_joint_layers(self, post = "conv", add_bias = False):
            self.attn1n = copy.deepcopy(self.attn) ###############이게 말이되나 더 클텐데 원래 text까지 embedding하니깐 NOTE

            # remove additional qkv projections
            if hasattr(self.attn1n, "add_q_proj"):
                del self.attn1n.add_q_proj
                del self.attn1n.add_k_proj
                del self.attn1n.add_v_proj
                del self.attn1n.to_add_out
                del self.attn1n.norm_added_q
                del self.attn1n.norm_added_k
            
            if hasattr(self.attn1n, "set_processor"):
                self.attn1n.set_processor(FluxAttnProcessor2_0_HACK_for_dual_block())
            else:
                raise ValueError(
                    "The attention processor is not set in the attention module. "
                    "Please check if the attention module is compatible with FluxAttnProcessor2_0_HACK_for_dual_block."
                )

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)

            self.conv1n = zero_module(conv1n)
            self.post = post

            self.joint_scale = 1.0
            self.enable_joint_attention = True

        def set_joint_attention(self, enable = True):
            self.enable_joint_attention = enable

        def set_joint_scale(self, joint_scale = 1.0):
            self.joint_scale = joint_scale
        
        def post_proj(self, x_out, y_out, post_op, post_type):
            if post_type == "conv":
                xy_out = torch.cat([x_out, y_out], dim = 0)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = 0) 
            elif post_type == "conv_fuse":
                xy_out = torch.cat([x_out, y_out], dim = -1)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = -1)
            
            return x_post_out, y_post_out
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            image_only_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # added
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            joint_attention_kwargs = joint_attention_kwargs or {}
            
            if self.enable_joint_attention:
                # Attention.
                attention_outputs = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                
                joint_norm_hidden_states = norm_hidden_states
                batch_size = joint_norm_hidden_states.shape[0]
                
                # Get paired joint cross attention inputs
                x_ids, y_ids, x_weights, y_weights = self.unicon_config["attn_config"]
                input_ids = torch.cat([x_ids, y_ids])
                joint_ids = torch.cat([y_ids, x_ids])
                input_norm_hidden_states = joint_norm_hidden_states[input_ids].clone().detach()
                joint_norm_hidden_states = joint_norm_hidden_states[joint_ids].clone().detach()
                
                # Attention
                attn_output1n = self.attn1n(
                    hidden_states=input_norm_hidden_states,
                    encoder_hidden_states=joint_norm_hidden_states,
                    image_rotary_emb=image_only_rotary_emb,
                    **joint_attention_kwargs
                )

                # Post projection
                output1n = torch.zeros_like(attn_output1n)

                attn_output1n_x, attn_output1n_y = attn_output1n.chunk(2, dim = 0)
                is_train = self.unicon_config["train"]
                if is_train:
                    attn_output1n_x, attn_output1n_y = self.post_proj(attn_output1n_x, attn_output1n_y, self.conv1n, self.post)
                else:
                    cond_masks = self.unicon_config["cond_masks"]
                    for cur_cond, cond_mask in cond_masks.items():
                        # cond_mask = [model_name == cur_cond for model_name in model_names]
                        x_post_out, y_post_out = self.post_proj(attn_output1n_x[cond_mask], attn_output1n_y[cond_mask], self.post1n[cur_cond], self.post_type[cur_cond])
                        attn_output1n_x[cond_mask], attn_output1n_y[cond_mask] = x_post_out, y_post_out
                
                # Aggregate output
                attn_output1n_x = attn_output1n_x * x_weights.to(attn_output1n_x)
                attn_output1n_y = attn_output1n_y * y_weights.to(attn_output1n_y)
                B, N, C = attn_output1n_x.shape
                x_indexes = x_ids.view(-1,1,1).expand(-1, N, C)
                y_indexes = y_ids.view(-1,1,1).expand(-1, N, C)

                output1n = torch.scatter_reduce(output1n, dim = 0, index=x_indexes, src=attn_output1n_x, reduce = "sum")
                output1n = torch.scatter_reduce(output1n, dim = 0, index=y_indexes, src=attn_output1n_y, reduce = "sum")
                # output1n = output1n.clone()
                # output1n[x_ids] = output1n[x_ids] * 0
                attention_outputs = (attention_outputs[0] + output1n * self.joint_scale, attention_outputs[1])
                
            else:
                # Attention.
                attention_outputs = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )

            if len(attention_outputs) == 2:
                attn_output, context_attn_output = attention_outputs
            elif len(attention_outputs) == 3:
                attn_output, context_attn_output, ip_attn_output = attention_outputs

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output
            if len(attention_outputs) == 3:
                hidden_states = hidden_states + ip_attn_output

            # Process attention outputs for the `encoder_hidden_states`.
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states
    
    return UniConBlock
            
def make_flux_single_unicon_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a class of UniCon blocks.
    It adds joint cross attention to the forward function and enables related functions for initialzation and training.
    """

    class UniConBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def set_joint_layer_requires_grad(self, adapter_names, requires_grad):
            
            for module in self.attn1n.modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                if isinstance(adapter_names, str):
                    adapter_names = [adapter_names]

                # Deactivate grads on the inactive adapter and activate grads on the active adapter
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in adapter_names:
                            # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                            # happen if a completely different adapter layer is being activated.
                            layer.requires_grad_(requires_grad)

            self.conv1n.requires_grad_(requires_grad)
        
        @property
        def post_joint(self):
            if self.post == "scale":
                return self.scale1n
            elif self.post == "conv" or self.post == "conv_fuse":
                return self.conv1n

        def add_post_joint(self, name, post = "conv", add_bias = False):
            if not hasattr(self, "post1n"):
                self.post1n = nn.ModuleDict({})
                self.post_type = dict()

            if name in self.post1n:
                return

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)
            post_joint = zero_module(conv1n)
        
            self.post1n[name] = post_joint
            self.post_type[name] = post

        def initialize_joint_layers(self, post = "conv", add_bias = False):
            self.attn1n = copy.deepcopy(self.attn)
            
            # self.attn1n = Attention(
            #     query_dim=self.attn.query_dim,
            #     heads=self.attn.heads,
            #     dim_head=64, ###
            #     bias=True,
            #     cross_attention_dim=self.attn.cross_attention_dim,
            #     upcast_attention=False,
            #     out_bias=False, ###
            # )
            
            if hasattr(self.attn1n, "set_processor"):
                self.attn1n.set_processor(FluxAttnProcessor2_0_HACK_for_single_block())
            else:
                raise ValueError(
                    "The attention processor is not set in the attention module. "
                    "Please check if the attention module is compatible with FluxAttnProcessor2_0_HACK_for_single_block."
                )

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)

            self.conv1n = zero_module(conv1n)
            self.post = post

            self.joint_scale = 1.0
            self.enable_joint_attention = True

        def set_joint_attention(self, enable = True):
            self.enable_joint_attention = enable

        def set_joint_scale(self, joint_scale = 1.0):
            self.joint_scale = joint_scale
        
        def post_proj(self, x_out, y_out, post_op, post_type):
            if post_type == "conv":
                xy_out = torch.cat([x_out, y_out], dim = 0)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = 0) 
            elif post_type == "conv_fuse":
                xy_out = torch.cat([x_out, y_out], dim = -1)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = -1)
            
            return x_post_out, y_post_out
            
        def forward(
            self,
            hidden_states: torch.Tensor,
            temb: torch.Tensor,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        ) -> torch.Tensor:
            
            # Reference:
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L261
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
            
            # Based on diffusers 0.33.1
            
            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            joint_attention_kwargs = joint_attention_kwargs or {}
            
            if self.enable_joint_attention:
            
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )

                joint_norm_hidden_states = norm_hidden_states
                batch_size = joint_norm_hidden_states.shape[0]
                
                # Get paired joint cross attention inputs
                x_ids, y_ids, x_weights, y_weights = self.unicon_config["attn_config"]
                input_ids = torch.cat([x_ids, y_ids])
                joint_ids = torch.cat([y_ids, x_ids])
                input_norm_hidden_states = joint_norm_hidden_states[input_ids].clone().detach()
                joint_norm_hidden_states = joint_norm_hidden_states[joint_ids].clone().detach()
                
                # Attention
                attn_output1n = self.attn1n(
                    hidden_states=input_norm_hidden_states,
                    encoder_hidden_states=joint_norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs
                )

                # Post projection
                output1n = torch.zeros_like(attn_output)

                attn_output1n_x, attn_output1n_y = attn_output1n.chunk(2, dim = 0)
                is_train = self.unicon_config["train"]
                if is_train:
                    attn_output1n_x, attn_output1n_y = self.post_proj(attn_output1n_x, attn_output1n_y, self.conv1n, self.post)
                else:
                    cond_masks = self.unicon_config["cond_masks"]
                    for cur_cond, cond_mask in cond_masks.items():
                        # cond_mask = [model_name == cur_cond for model_name in model_names]
                        x_post_out, y_post_out = self.post_proj(attn_output1n_x[cond_mask], attn_output1n_y[cond_mask], self.post1n[cur_cond], self.post_type[cur_cond])
                        attn_output1n_x[cond_mask], attn_output1n_y[cond_mask] = x_post_out, y_post_out
                
                # Aggregate output
                attn_output1n_x = attn_output1n_x * x_weights.to(attn_output1n_x)
                attn_output1n_y = attn_output1n_y * y_weights.to(attn_output1n_y)
                B, N, C = attn_output1n_x.shape
                x_indexes = x_ids.view(-1,1,1).expand(-1, N, C)
                y_indexes = y_ids.view(-1,1,1).expand(-1, N, C)

                output1n = torch.scatter_reduce(output1n, dim = 0, index=x_indexes, src=attn_output1n_x, reduce = "sum")
                output1n = torch.scatter_reduce(output1n, dim = 0, index=y_indexes, src=attn_output1n_y, reduce = "sum")
                # output1n = output1n.clone()
                # output1n[x_ids] = output1n[x_ids] * 0 # updirection
                attn_output = attn_output + output1n * self.joint_scale
                
            else:
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )

            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states

    return UniConBlock

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def initialize_joint_layers(model: torch.nn.Module, post = "conv"):
    """ Initialize all joint cross attentions """

    model = model.transformer if hasattr(model, "transformer") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.initialize_joint_layers(post = post)
    return model

def hack_lora_forward(model: torch.nn.Module):
    """ Replace the forward function of LoRA layers """

    model = model.transformer if hasattr(model, "transformer") else model
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            # if "x_embedder" in name:
            # # if "transformer_block" in name:
            # # if "single_transformer_blocks" in name:
            #     print(name)
            #     print(module)
            #     print(module.disable_adapters)
            #     import pdb; pdb.set_trace()
            module.forward = lora_forward_hack(module)
    return model

def lora_forward_hack(self):
    """
    Hack forward function of LoRA layers.
    Let each adapter selectively applies to inputs specified by a mask.
    """
    def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                lora_mask = self.lora_mask[active_adapter]
                lora_mask = lora_mask.repeat_interleave(x.shape[0] // len(lora_mask), dim=0)
                masked_x = x[lora_mask]

                if not self.use_dora[active_adapter]:
                    result_lora = lora_B(lora_A(dropout(masked_x))) * scaling
                else:
                    masked_x = dropout(masked_x)
                    result_lora = self._apply_dora(masked_x, lora_A, lora_B, scaling, active_adapter)

                result[lora_mask] += result_lora

            result = result.to(torch_result_dtype)
        return result
    return forward

def set_joint_layer_requires_grad(model: torch.nn.Module, adapter_names, requires_grad):
    """ Set requires_grad for all unicon parameters """

    model = model.transformer if hasattr(model, "transformer") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.set_joint_layer_requires_grad(adapter_names, requires_grad)
    return model

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

def set_joint_attention(model: torch.nn.Module, enable = True, name_filter = None):
    """ Set whether to enable the joint cross attention """

    model = model.transformer if hasattr(model, "transformer") else model
    for name, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            if name_filter is None or name_filter in name:
                module.set_joint_attention(enable = enable)
    return model

def set_joint_scale(model: torch.nn.Module, scale = 1.0):
    """ Set the scale of joint cross attention """

    model = model.transformer if hasattr(model, "transformer") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.set_joint_scale(scale = scale)
    return model

def add_post_joint(model: torch.nn.Module, name, post = "conv", add_bias = False, **kwargs):
    """ Add a post projection """

    model = model.transformer if hasattr(model, "transformer") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.add_post_joint(name, post, add_bias)
    return model
    
def set_unicon_config(model: torch.nn.Module, k, v):
    """ Update joint cross attention configurations in patched modules """

    model = model.transformer if hasattr(model, "transformer") else model
    model.unicon_config[k] = v
    return model

