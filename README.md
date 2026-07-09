<div align="center">
<img src="assets/imgs/icon.png" width="90"/>
</div>

<h1 align="center">Target-Bench</h1>
<h3 align="center">Can Video World Models Achieve Mapless Path Planning with Semantic Targets?</h3>

<div align="center">

If you find our work useful, please give us a star ⭐! Your support drives us to keep improving.

[Project Page](https://target-bench.github.io/) | [Paper](https://target-bench.github.io/resources/TargetBench.pdf) | [Dataset (Hugging Face)](https://huggingface.co/datasets/target-bench/dataset) | [Dataset (ModelScope)](https://modelscope.cn/datasets/parkring/target-bench)

![teaser](assets/imgs/teaser.gif)

</div>

**TL;DR:** Target-Bench is a benchmark and dataset for evaluating video world models on mapless robotic path planning with semantic targets.

## TODO 📋

- [x] Fine-tuned checkpoints release
- [x] Fine-tune code release
- [x] Benchmark code release
- [x] Dataset release
- [x] Paper release
- [x] Website launch

## News

- **[2026.07]** 🚀 Target-Bench is using [VGGT-Omega](https://vggt-omega.github.io/) as the default evaluation backbone.
- **[2026.06]** 🎉 Target-Bench is accepted by ECCV 2026 🥳! See you September in Malmö 🇸🇪
- **[2026.01]** 🔥 We release the training code for fine-tuning world models.
- **[2025.11]** We release the paper, dataset, benchmark code, and project website.

## Contents

- [Installation](#installation)
- [Data Setup](#data-setup)
  - [Option A: Hugging Face](#option-a-hugging-face) <img src="assets/imgs/huggingface.svg" width="16" alt="Hugging Face" style="vertical-align: -0.25em;" />
  - [Option B: ModelScope](#option-b-modelscope) <img src="assets/imgs/modelscope.svg" width="16" alt="ModelScope" style="vertical-align: -0.25em;" />
- [Checkpoint Setup](#checkpoint-setup)
- [Evaluation](#evaluation)
- [Fine-tune](#fine-tune)
- [Citation](#citation)

## Installation

Clone the repository:

```bash
git clone https://github.com/TUM-AVS/target-bench.git
cd target-bench
```

Ensure you have [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) installed, then create the evaluation environment:

```bash
bash set_env.sh install
conda activate target-bench-eval
```

This creates the `target-bench-eval` conda environment, installs `requirements.txt`, and registers the vendored VGGT/VGGT-Omega packages. The `target-bench-eval` environment supports all three trajectory backbones. See [docs/env.md](docs/env.md) for environment variables, checkpoint paths, and verification.

We use [VGGT-Omega](https://vggt-omega.github.io/) as the default trajectory backbone because it gives stronger reconstruction quality than [VGGT](https://github.com/facebookresearch/vggt) and [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) in our evaluation framework.

## Data Setup

Target-Bench evaluation needs two archives:

- `benchmark_data.tar`: benchmark scenarios, metadata, and trajectories
- `wm_videos.tar`: generated world-model videos

After extraction, the expected layout is:

```text
target-bench/
└── dataset/
    ├── Benchmark/
    └── wm_videos/
```

<a id="option-a-hugging-face"></a>

### Option A: Hugging Face <img src="assets/imgs/huggingface.svg" width="22" alt="Hugging Face" style="vertical-align: -0.25em;" />

The Hugging Face dataset contains only the two required archives: `benchmark_data.tar` and `wm_videos.tar`.

```bash
mkdir -p dataset

hf download target-bench/dataset \
  benchmark_data.tar wm_videos.tar \
  --repo-type dataset \
  --local-dir dataset

tar -xf dataset/benchmark_data.tar -C dataset
tar -xf dataset/wm_videos.tar -C dataset
```

<a id="option-b-modelscope"></a>

### Option B: ModelScope <img src="assets/imgs/modelscope.svg" width="22" alt="ModelScope" style="vertical-align: -0.25em;" />

ModelScope is sometimes faster. The ModelScope dataset also contains `preprocessed_dataset.tar`, but it is not required for benchmark evaluation. Download only `benchmark_data.tar` and `wm_videos.tar`:

```bash
mkdir -p dataset

modelscope download \
  --dataset parkring/target-bench \
  benchmark_data.tar wm_videos.tar \
  --local_dir dataset

tar -xf dataset/benchmark_data.tar -C dataset
tar -xf dataset/wm_videos.tar -C dataset
```

## Checkpoint Setup

Evaluation uses VGGT-Omega by default. Download the default checkpoint into `checkpoints/` before running evaluation:

```bash
mkdir -p checkpoints/vggt_omega
hf download facebook/vggt-omega \
  vggt_omega_1b_512.pt \
  --local-dir checkpoints/vggt_omega
```

VGGT-Omega is a gated model, so run `hf auth login` first if needed. To set up optional [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) and [VGGT](https://github.com/facebookresearch/vggt) checkpoints, see [docs/checkpoints.md](docs/checkpoints.md).

## Evaluation

VGGT-Omega is the default evaluation backbone:

```bash
conda activate target-bench-eval
python evaluation/evaluate.py
```

For a quick smoke test:

```bash
python evaluation/evaluate.py -n 3
```

The same entry point can run the other supported backbones:

```bash
python evaluation/evaluate.py --backbone vggt  # da3 or vggt_omega
```

To evaluate only `gt_videos` or specific video world model:

```bash
python evaluation/evaluate.py --models gt_videos
python evaluation/evaluate.py --models sora-2
```

To run all supported backbones:

```bash
bash evaluation/evaluate.sh
```

Results are written to `evaluation_results/`. The main files are:

- `model_summary_*.csv`
- `segment_model_results_*.csv`
- `evaluation_results_*.json`

## Fine-tune

The fine-tuning code is kept under `DiffSynth-Studio/`.

### 1. Fine-tune Environment

```bash
conda deactivate
conda create -n target-finetune python=3.10 -y
conda activate target-finetune
python -m pip install -r DiffSynth-Studio/requirements.txt
```

### 2. Checkpoints

```bash
mkdir -p DiffSynth-Studio/models/train/ckpts
hf download target-bench/ckpts \
  --repo-type model \
  --local-dir DiffSynth-Studio/models/train/ckpts
```

### 3. Fine-tune Dataset

```bash
hf download target-bench/finetune_dataset \
  --repo-type dataset \
  --local-dir dataset

unzip dataset/data_four_segments_121_frames.zip -d dataset
unzip dataset/data_single_segment_121_frames.zip -d dataset
unzip dataset/data_inference.zip -d dataset
```

### 4. Inference

Run inference with the released fine-tuned checkpoint:

```bash
(cd DiffSynth-Studio && python run_inference_four_segments_epoch-49_batch.py)
```

### 5. LoRA Fine-tuning

Fine-tune Wan2.2-TI2V-5B on Target-Bench scenarios:

```bash
(cd DiffSynth-Studio && bash Wan2.2-TI2V-5B_four_segments.sh)
```

## Citation

```bibtex
@inproceedings{wang2025target,
  title={Target-Bench: Can Video World Models Achieve Mapless Path Planning with Semantic Targets?},
  author={Wang, Dingrui and Ye, Hongyuan and Liang, Zhihao and Sun, Zhexiao and Lu, Zhaowei and Zhang, Yuchen and Zhao, Yuyu and Gao, Yuan and Seegert, Marvin and Sch{\"a}fer, Finn and others},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

## Credits

This project builds on the following open-source works:

- [VGGT](https://github.com/facebookresearch/vggt)
- [VGGT-Omega](https://github.com/facebookresearch/vggt-omega)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [ViPE](https://github.com/nv-tlabs/vipe)
- [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

Please refer to their respective repositories and licenses for details.

## License

Target-Bench original code is released under the MIT License. Third-party components vendored in this repository are governed by their own licenses. In particular, VGGT-Omega is released under the FAIR Noncommercial Research License and may restrict commercial use.
