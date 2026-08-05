# BFS: Back-to-Front Layered Image Synthesis via Knowledge Transfer

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
on Hugging Face and authenticate with `huggingface-cli login` before inference.

## Model weights

BFS inference uses three sets of weights:

- FLUX.1-Fill-dev: a local path or Hugging Face model ID
- Transparency VAE: a local path or Hugging Face model ID
- BFS checkpoint: either `trained_weights.pt` or a directory containing it

The BFS checkpoint must match the default release architecture (`rank_joint=4`,
`post_joint=conv`). Use the corresponding command-line options only if your
checkpoint was trained with different settings.

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
  --pretrained_trans_vae_path /path/to/transparency-vae \
  --pretrained_lora_path /path/to/bfs-checkpoint \
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

## Acknowledgements

This project builds on FLUX, Diffusers, UniCon, ObjectClear, and OmniErase.
