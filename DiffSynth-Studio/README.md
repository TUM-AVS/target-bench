# DiffSynth-Studio for Target-Bench

This directory keeps a minimal vendored DiffSynth-Studio subset for Target-Bench
world-model fine-tuning and inference.

Kept files:

- `diffsynth/`: source code required by the Wan2.2-TI2V pipeline and trainer
- `examples/wanvideo/model_training/train.py`: Wan video training entry
- `examples/wanvideo/model_training/full/accelerate_config_14B.yaml`: accelerate config used by Target-Bench scripts
- `Wan2.2-TI2V-5B_*.sh`: Target-Bench fine-tuning launch scripts
- `run_inference_*_batch.py`: Target-Bench inference scripts
- `models/train/`: local checkpoint and training-output location

Local checkpoints and generated artifacts under `models/train/ckpts/` are ignored
by git.

For the original upstream project, see:
https://github.com/modelscope/DiffSynth-Studio
