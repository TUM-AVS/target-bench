# Checkpoint Setup for DA3 and VGGT

Target-Bench uses VGGT-Omega as the default evaluation backbone. This page is only needed if you want to evaluate with [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) or [VGGT](https://github.com/facebookresearch/vggt).

All checkpoint files and model caches should stay under `checkpoints/`, which is ignored by git.

## DA3

DA3 uses Hugging Face model IDs and caches model files under `checkpoints/da3` by default. The default Target-Bench DA3 model is `DA3NESTED-GIANT-LARGE-1.1`.

```bash
mkdir -p checkpoints/da3
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --cache-dir checkpoints/da3
```

Run DA3 evaluation with:

```bash
python evaluation/evaluate.py --backbone da3
```

To use `DA3-GIANT-1.1` instead:

```bash
hf download depth-anything/DA3-GIANT-1.1 \
  --cache-dir checkpoints/da3

python evaluation/evaluate.py --backbone da3 --model_id DA3-GIANT-1.1
```

## VGGT

VGGT expects `model.pt` under `checkpoints/vggt` by default.

```bash
mkdir -p checkpoints/vggt
hf download facebook/VGGT-1B \
  model.pt \
  --local-dir checkpoints/vggt
```

Run VGGT evaluation with:

```bash
python evaluation/evaluate.py --backbone vggt
```

You can also use an explicit checkpoint path:

```bash
python evaluation/evaluate.py --backbone vggt \
  --checkpoint_path /path/to/model.pt
```
