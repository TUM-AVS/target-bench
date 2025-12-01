
eval "$(conda shell.bash hook)"
conda deactivate || true

echo "Running SpaTrack2 evaluation..."
conda activate SpaTrack2
python evaluation/target_eval_spa.py -n 1
conda deactivate

echo "Running VIPE evaluation..."
conda activate vipe
python evaluation/target_eval_vipe.py
python evaluation/compare_overall.py
conda deactivate

echo "Running VGGt evaluation..."
conda activate vggt
python evaluation/target_eval_vggt.py
python evaluation/target_eval_vggt_implicit.py
conda deactivate

echo "All evaluations completed successfully!"
