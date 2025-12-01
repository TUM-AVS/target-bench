#!/bin/bash

# Enable conda commands
eval "$(conda shell.bash hook)"

install_vggt() {
    echo "========================================"
    echo "Installing VGGT Environment..."
    echo "========================================"
    
    conda create -n vggt python=3.10 -y
    conda activate vggt
    
    # Install dependencies from requirements.txt
    pip install -r models/vggt/requirements.txt
    pip install opencv-python huggingface_hub==0.34.4

    conda deactivate
    echo "VGGT installation complete."
}

install_vipe() {
    echo "========================================"
    echo "Installing VIPE Environment..."
    echo "========================================"
    
    conda env create -f models/vipe/envs/base.yml
    conda activate vipe
    
    # Install python dependencies
    pip install -r models/vipe/envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

    # Install package in editable mode
    # Store current directory
    cd models/vipe
    pip install --no-build-isolation -e .
    cd ../../
    conda deactivate
    echo "VIPE installation complete."
}

install_spatracker() {
    echo "========================================"
    echo "Installing SpaTracker Environment..."
    echo "========================================"
    
    conda create -n SpaTrack2 python=3.11 -y
    conda activate SpaTrack2
    
    # Install torch specific version as per README
    python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    
    # Install dependencies
    pip install -r models/spatracker/requirements.txt
    
    conda deactivate
    echo "SpaTracker installation complete."
}

# Main execution
case "$1" in
    vggt)
        install_vggt
        ;;
    vipe)
        install_vipe
        ;;
    spatracker)
        install_spatracker
        ;;
    all)
        install_vggt
        install_vipe
        install_spatracker
        ;;
    *)
        echo "Usage: $0 {vggt|vipe|spatracker|all}"
        exit 1
        ;;
esac

