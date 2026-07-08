# Environment Setup and Evaluation

Target-Bench evaluation now uses one shared environment for the supported
backbones:

- VGGT
- Depth Anything 3
- VGGT-Omega

The repository keeps only these three evaluation backbones wired into code.

## Install

```bash
bash set_env.sh install
```

The installer creates `target-bench-eval` by default and installs:

- Python dependencies from `requirements.txt`
- editable `models/vggt`
- editable `models/vggt-omega`
- vendored Depth Anything 3 source from `${DA3_REPO_DIR:-models/da3}`

Depth Anything 3 is included under `models/da3`. If you want to use a different
local checkout, set:

```bash
export DA3_REPO_DIR=/path/to/Depth-Anything-3
bash set_env.sh install
```

## Verify

```bash
bash set_env.sh verify
```

The verify step imports `vggt`, `vggt_omega`, `torch`, `cv2`, and the
Target-Bench DA3 eval adapter.

## Run Evaluation

Use the unified batch entry point:

```bash
conda activate target-bench-eval
NUM_SAMPLES=3 bash evaluation/evaluate.sh
```

Useful environment variables:

- `TARGET_BENCH_ENV_NAME`: conda env name, default `target-bench-eval`
- `NUM_SAMPLES`: optional sample limit passed to each evaluator
- `GT_TRAJECTORY_SCOPE`: default `current_to_end`
- `DA3_MODEL_ID`: default `DA3NESTED-GIANT-LARGE-1.1`
- `TARGET_BENCH_CHECKPOINT_DIR`: shared local checkpoint/cache directory, default `checkpoints`
- `VGGT_CHECKPOINT`: optional local VGGT checkpoint path; otherwise downloaded from Hugging Face
- `VGGT_OMEGA_CHECKPOINT`: optional local VGGT-Omega checkpoint path; otherwise downloaded from Hugging Face after `hf auth login`

By default checkpoints and Hugging Face model files are kept under:

```text
checkpoints/
├── da3/
├── vggt/
└── vggt_omega/
```

This directory is git-ignored.

Run an individual backbone through the unified evaluator:

```bash
python evaluation/evaluate.py --backbone vggt
python evaluation/evaluate.py --backbone da3 --model_id DA3NESTED-GIANT-LARGE-1.1
python evaluation/evaluate.py --backbone vggt_omega
```
