#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-}"
GT_SCOPE="${GT_TRAJECTORY_SCOPE:-current_to_end}"
DA3_MODEL_ID="${DA3_MODEL_ID:-DA3NESTED-GIANT-LARGE-1.1}"

sample_args=()
if [[ -n "${NUM_SAMPLES}" ]]; then
    sample_args=(-n "${NUM_SAMPLES}")
fi

echo "Running VGGT evaluation..."
python evaluation/evaluate.py \
    --backbone vggt \
    "${sample_args[@]}" \
    --eval_config_path evaluation/config/vggt_config.yaml \
    --gt_trajectory_scope "${GT_SCOPE}"

echo "Running DA3 evaluation..."
python evaluation/evaluate.py \
    --backbone da3 \
    "${sample_args[@]}" \
    --model_id "${DA3_MODEL_ID}" \
    --eval_config_path evaluation/config/da3_config.yaml \
    --gt_trajectory_scope "${GT_SCOPE}"

echo "Running VGGT-Omega evaluation..."
omega_args=()
if [[ -n "${VGGT_OMEGA_CHECKPOINT:-}" ]]; then
    omega_args=(--checkpoint_path "${VGGT_OMEGA_CHECKPOINT}")
fi
python evaluation/evaluate.py \
    --backbone vggt_omega \
    "${sample_args[@]}" \
    "${omega_args[@]}" \
    --eval_config_path evaluation/config/vggt_omega_config.yaml \
    --gt_trajectory_scope "${GT_SCOPE}"

echo "All supported evaluations completed successfully."
