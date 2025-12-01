# Environment Setup and Evaluation

This guide explains how to set up the environments and run the evaluation pipeline for Target-Bench.

## 1. Environment Installation

Target-Bench uses multiple environments to manage dependencies for different models. You can install them all at once using the provided script.

### Install All Environments
To install support for VGGT, ViPE, and SpaTracker2:

```bash
bash set_env.sh all
```

### Individual Environment Installation
If you only need a specific environment, you can install them individually:

```bash
# Install only VGGT environment
bash set_env.sh vggt

# Install only ViPE environment
bash set_env.sh vipe

# Install only SpaTracker environment
bash set_env.sh spatracker
```

## 2. Evaluation Pipeline

The main evaluation entry point is `evaluation/evaluate.sh`. This script automatically handles environment switching and executes the benchmark for all supported models.

### Run Full Evaluation

```bash
bash evaluation/evaluate.sh
```

This script will sequentially:
1.  **SpaTracker Evaluation**: Activates `SpaTrack2` environment and runs evaluation.
2.  **VIPE Evaluation**: Activates `vipe` environment and runs evaluation.
3.  **VGGT Evaluation**: Activates `vggt` environment and runs evaluation.
