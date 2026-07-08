#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TARGET_BENCH_ENV_NAME:-target-bench-eval}"
PYTHON_VERSION="${TARGET_BENCH_PYTHON:-3.10}"
DA3_REPO_DIR="${DA3_REPO_DIR:-models/da3}"

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        return
    fi

    for conda_sh in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "${conda_sh}" ]]; then
            # shellcheck disable=SC1090
            source "${conda_sh}"
            return
        fi
    done

    echo "conda not found. Please install Miniconda or initialize conda first." >&2
    exit 1
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

check_da3() {
    if [[ ! -d "${DA3_REPO_DIR}/src/depth_anything_3" ]]; then
        echo "Depth-Anything-3 source not found: ${DA3_REPO_DIR}/src/depth_anything_3" >&2
        echo "Expected checkout: git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git models/da3" >&2
        exit 1
    fi
}

install_env() {
    load_conda

    if ! env_exists; then
        conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
    fi

    conda activate "${ENV_NAME}"
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e models/vggt -e models/vggt-omega

    check_da3
    verify_env
}

verify_env() {
    load_conda
    conda activate "${ENV_NAME}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-target-bench}"
    mkdir -p "${MPLCONFIGDIR}"

    python - <<'PY'
import importlib

for module in [
    "torch",
    "cv2",
    "modelscope",
    "vggt.models.vggt",
    "vggt_omega.models",
]:
    importlib.import_module(module)
    print(f"OK: {module}")

da3_adapter = importlib.import_module("pipelines.da3.scale_factor.compute_scale_factor")
da3_config = da3_adapter.load_config("pipelines/da3/configs/da3_pcd_configs.yml")
da3_adapter._add_da3_source_paths(da3_config)
da3_adapter._install_da3_eval_import_stubs()
importlib.import_module("depth_anything_3.api")
print("OK: depth_anything_3.api")
PY
}

case "${1:-install}" in
    install)
        install_env
        ;;
    verify)
        verify_env
        ;;
    *)
        echo "Usage: $0 {install|verify}" >&2
        exit 1
        ;;
esac
