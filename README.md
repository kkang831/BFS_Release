# [SIGGRAPH 2026] Official PyTorch implementation of "BFS: Back-to-Front Layered Image Synthesis via Knowledge Transfer"

Kyoungkook Kang, Gyujin Sim, and Sunghyun Cho

## Installation

Python 3.10 or newer and a CUDA-capable GPU are required. We recommend using a
clean virtual environment.

```bash
git clone https://github.com/kkang831/BFS_Release.git
cd BFS_Release
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The base model `black-forest-labs/FLUX.1-Fill-dev` is gated. Accept its license
on Hugging Face and authenticate with `hf auth login` before inference.

## Model weights

BFS inference uses three sets of weights:

- FLUX.1-Fill-dev: a local path or Hugging Face model ID
- Transparency VAE: a local path or Hugging Face model ID
- BFS checkpoint: either `trained_weights.pt` or a directory containing it

The BFS checkpoint must match the default release architecture (`rank_joint=4`,
`post_joint=conv`). Use the corresponding command-line options only if your
checkpoint was trained with different settings.

### Download the base model

`FLUX.1-Fill-dev` is a gated Hugging Face model. First request access and accept
the license on its model page, then authenticate from the virtual environment:

```bash
source .venv/bin/activate
hf auth login
```

Download the model into the standard Hugging Face cache:

```bash
hf download black-forest-labs/FLUX.1-Fill-dev
```

The default cache location is
`~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-Fill-dev/`.
Inference can then use the model ID directly:

```text
--pretrained_model_name_or_path black-forest-labs/FLUX.1-Fill-dev
```

To keep a standalone model copy at a specific location instead, use:

```bash
hf download black-forest-labs/FLUX.1-Fill-dev \
  --local-dir ./checkpoints/FLUX.1-Fill-dev
```

In that case, pass `./checkpoints/FLUX.1-Fill-dev` to
`--pretrained_model_name_or_path`.

### Download the BFS checkpoints

The release checkpoints will be shared through Google Drive. Replace the
placeholders below with the public download links when they are available:

- Transparency VAE: [Link](https://drive.google.com/file/d/1u4ZIz_MRvVDeJ9Qv4E2zLTPxMldy_TGP/view?usp=sharing)
- BFS checkpoint: [Link](https://drive.google.com/file/d/1AgxztNBgi2vYW4FKA3VTXESNxGUKWf4-/view?usp=sharing)

After downloading, arrange the files as follows:

```text
checkpoints/
├── transparency_vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── bfs/
    └── trained_weights.pt
```

The current local test setup follows this layout. The `checkpoints/` directory
is excluded from Git because these weight files are large. Use the following
arguments with this layout:

```text
--pretrained_trans_vae_path ./checkpoints/transparency_vae
--pretrained_lora_path ./checkpoints/bfs/trained_weights.pt
```

## Input format

For one sample, pass an image, mask, and caption file directly:

```text
example/
├── image.png
├── mask.png
└── caption.txt
```

For batch inference, pass three directories. Each image must have a mask and a
caption with the same filename stem. Supported image formats are PNG, JPG, and
JPEG.

```text
inputs/
├── images/       000.png, 001.jpg, ...
├── masks/        000.png, 001.png, ...
└── captions/     000.txt, 001.txt, ...
```

White mask pixels indicate the target foreground region. Each caption text file
should describe that foreground object.

## Inference

```bash
python3 inference.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-Fill-dev \
  --pretrained_trans_vae_path ./checkpoints/transparency_vae \
  --pretrained_lora_path ./checkpoints/bfs/trained_weights.pt \
  --input_path ./inputs/images \
  --mask_path ./inputs/masks \
  --foreground_caption_path ./inputs/captions \
  --output_path ./results \
  --return_recomposite
```

The main output is an RGBA PNG. With `--return_recomposite`, the command also
writes `<name>_recomp.png`, which composites the predicted foreground over the
original image.

Run `python3 inference.py --help` for all options. `run_inference.sh` provides a
short equivalent example using environment variables.

## Citation

```bibtex
@article{kang2026bfs,
  title={BFS: Back-to-Front Layered Image Synthesis via Knowledge Transfer},
  author={Kang, Kyoungkook and Sim, Gyujin and Cho, Sunghyun},
  journal={arXiv preprint arXiv:2605.24894},
  year={2026}
}
```

## Acknowledgements

This project builds on FLUX, Diffusers, UniCon, ObjectClear, and OmniErase.
