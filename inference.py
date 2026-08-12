import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch

import diffusers, transformers
from diffusers import FluxFillPipeline, AutoencoderKL
from peft import LoraConfig

from utils.image_utils import resize_by_short_side
from utils.unicon_utils import set_unicon_config
from pipelines.pipeline import FluxControlPipeline
from patch import patch_BFS as patch

def generate_argparser():
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_trans_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained VAE.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LoRA.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    # BFS specific
    parser.add_argument(
        "--rank_joint",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument(
        "--post_joint",
        type=str,
        default="conv",
        help=("The post joint module type. Choose between 'conv' or 'conv_fuse'."),
    )
    
    # data loading
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='Input image or directory of images.')
    parser.add_argument('-m', '--mask_path', type=str, required=True,
                        help='Input mask image or directory of masks.')
    parser.add_argument('-c', '--foreground_caption_path', type=str, required=True,
                        help='Caption text file or directory of text files.')
    parser.add_argument('-o', '--output_path', type=str, default='./results',
                        help='Output directory (default: ./results).')
    parser.add_argument("--inference_resize_short", type=int, default=512)
    parser.add_argument("--return_recomposite", action="store_true",
                        help="Whether to return the recomposited image using the predicted RGBA foreground and the original background.")
    
    # inference
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible inference.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=30,
        help="the guidance scale used for transformer.",
    )
    return parser

def load_model(args, weight_dtype, device):
    pipe = FluxFillPipeline.from_pretrained(args.pretrained_model_name_or_path, 
                                            variant=args.variant,
                                            revision=args.revision,
                                            torch_dtype=weight_dtype)
    
    vae = pipe.vae
    flux_transformer = pipe.transformer
    
    import types
    from patch.patch_flux import my_forward
    flux_transformer.forward = types.MethodType(my_forward, flux_transformer)
    
    trans_vae = AutoencoderKL.from_pretrained(args.pretrained_trans_vae_path).to(dtype=pipe.vae.dtype)
    trans_vae.eval()
    
    vae.requires_grad_(False)
    trans_vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    
    # let's not move the VAE to the GPU yet. 
    vae.to(device=device, dtype=torch.float32)
    trans_vae.to(device=device, dtype=torch.float32)
    flux_transformer.to(device=device, dtype=weight_dtype)
    
    # The released checkpoint uses the training-layout branch of the BFS patch.
    # This flag selects module layout only; gradients remain disabled below.
    patch.apply_patch(flux_transformer, name_skip=None, train=True)
    patch.initialize_joint_layers(flux_transformer, post=args.post_joint)

    all_loras = []
    joint_lora_config = LoraConfig(
        r=args.rank_joint,
        lora_alpha=args.rank_joint,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=["attn1n.to_k", "attn1n.to_q", "attn1n.to_v", "attn1n.to_out.0"],
    )
    flux_transformer.add_adapter(joint_lora_config, adapter_name = "xy_lora")
    flux_transformer.add_adapter(joint_lora_config, adapter_name = "yx_lora")
    all_loras += ["xy_lora", "yx_lora"]
    
    patch.hack_lora_forward(flux_transformer)
    
    lora_state_dict = resolve_bfs_checkpoint(args.pretrained_lora_path)
    print(f"Loading BFS weights from {lora_state_dict}")
    flux_transformer.load_state_dict(
        torch.load(lora_state_dict, map_location="cpu", weights_only=True),
        strict=False,
    )
        
    # ------------------------------------------------------------------------
    flux_transformer.set_adapter(all_loras)
    patch.set_joint_attention(flux_transformer, enable = True)
    return flux_transformer, vae, trans_vae


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_paths(path, suffixes):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"Unsupported file type: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        raise ValueError(f"No supported files found in {path}")
    return files


def build_samples(input_path, mask_path, caption_path):
    roots = [Path(input_path), Path(mask_path), Path(caption_path)]
    root_types = [path.is_file() for path in roots]
    if any(root_types) and not all(root_types):
        raise ValueError("Input, mask, and caption paths must all be files or all be directories.")

    images = collect_paths(input_path, IMAGE_SUFFIXES)
    masks = collect_paths(mask_path, IMAGE_SUFFIXES)
    captions = collect_paths(caption_path, {".txt"})

    if all(root_types):
        return [(images[0], masks[0], captions[0])]

    mask_by_stem = {path.stem: path for path in masks}
    caption_by_stem = {path.stem: path for path in captions}
    if len(mask_by_stem) != len(masks) or len(caption_by_stem) != len(captions):
        raise ValueError("Mask and caption directories must not contain duplicate filename stems.")
    missing_masks = [path.stem for path in images if path.stem not in mask_by_stem]
    missing_captions = [path.stem for path in images if path.stem not in caption_by_stem]
    if missing_masks:
        raise ValueError("Missing same-named mask for: " + ", ".join(missing_masks))
    if missing_captions:
        raise ValueError("Missing same-named caption for: " + ", ".join(missing_captions))
    return [(path, mask_by_stem[path.stem], caption_by_stem[path.stem]) for path in images]


def validate_samples(samples):
    """Fail fast on inexpensive input checks before loading GPU models."""
    for img_path, mask_path, caption_path in samples:
        if not caption_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Caption is empty: {caption_path}")
        with Image.open(img_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Image and mask sizes differ for {img_path.name}: "
                    f"{image.size} vs {mask.size}"
                )


def resolve_bfs_checkpoint(path):
    checkpoint = Path(path)
    weights = checkpoint if checkpoint.is_file() else checkpoint / "trained_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"BFS weights not found: {weights}")
    return weights


def main(args):
    diffusers.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity_warning()
    
    samples = build_samples(args.input_path, args.mask_path, args.foreground_caption_path)
    validate_samples(samples)
    resolve_bfs_checkpoint(args.pretrained_lora_path)
    
    # ------------------ set up pipeline -------------------
    if not torch.cuda.is_available():
        raise RuntimeError("BFS inference requires a CUDA-capable GPU.")
    device = torch.device('cuda')
    weight_dtype = torch.bfloat16
    flux_transformer, _, trans_vae = load_model(args, weight_dtype, device)
    result_root = Path(args.output_path)
    result_root.mkdir(parents=True, exist_ok=True)
    
    bsz = 2
    set_unicon_config(flux_transformer, bsz, device=device, dtype=weight_dtype)
    
    pipeline = FluxControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=flux_transformer,
        torch_dtype=weight_dtype,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # -------------------- start processing ---------------------
    for img_path, mask_path, caption_path in tqdm(samples, desc="BFS inference"):
        foreground_caption = caption_path.read_text(encoding="utf-8").strip()

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").convert("RGB")
        image_or = image.copy()
        
        image = resize_by_short_side(image, args.inference_resize_short, resample=Image.BICUBIC)
        mask = resize_by_short_side(mask, args.inference_resize_short, resample=Image.NEAREST)
        
        w, h = image.size
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=weight_dtype):
            result = pipeline(
                prompt=[foreground_caption],
                control_image=image,
                control_mask=mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=512,
                height=h,
                width=w,
                trans_vae=trans_vae,
            ).images
        result_fg_rgba_pil = result[1].resize(image_or.size, Image.BICUBIC)
        
        # save results
        save_path = result_root / f'{img_path.stem}.png'
        result_fg_rgba_pil.save(save_path)
        
        # save recomposited image
        if args.return_recomposite:
            result_fg_rgba_np = np.array(result_fg_rgba_pil).astype(np.float32) / 255.0
            alpha = result_fg_rgba_np[:, :, 3:4]
            image_np = np.array(image_or).astype(np.float32) / 255.0
            result_re_comp_np = result_fg_rgba_np[:, :, :3] * alpha + image_np * (1 - alpha)
            result_re_comp_pil = Image.fromarray((result_re_comp_np * 255.0).astype(np.uint8))
            save_path_comp = result_root / f'{img_path.stem}_recomp.png'
            result_re_comp_pil.save(save_path_comp)
    print(f'\nAll results are saved in {result_root}')


if __name__ == '__main__':
    # argument parse
    parser = generate_argparser()
    args = parser.parse_args()
    
    main(args)
    print('Inference Done!')
