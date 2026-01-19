#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
BASE_DIR = Path("../dataset/data_inference")
METADATA_PATH = Path("../dataset/data_inference/metadata.csv")
OUTPUT_DIR = None  # Set to Path("custom/output/dir") to override the automatic folder
LORA_PATH = Path("models/train/ckpts/four_segments_epoch-49.safetensors")  # Set to Path("path/to/lora.safetensors") when using LoRA
LORA_ALPHA = 1.0
HEIGHT = 704
WIDTH = 1280
NUM_FRAMES = 121
FPS = 15
QUALITY = 5
SEED = 1
NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT


def sanitize(name: str) -> str:
    """Return a filesystem-safe tag derived from the LoRA filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def infer_output_dir(explicit: Path | None, lora_path: Path | None, num_frames: int) -> Path:
    """Decide the output folder based on whether LoRA is used."""
    if explicit:
        return explicit
    base = Path("outputs")
    if lora_path:
        tag = sanitize(lora_path.stem)
        return base / f"wan2.2-ti2v-5b_inference_lora_{tag}_{num_frames}_frames"
    return base / f"wan2.2-ti2v-5b_inference_base_{num_frames}_frames"


def build_pipeline(lora_path: Path | None, lora_alpha: float):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    if lora_path:
        pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_alpha)
        print(f"[info] LoRA loaded: {lora_path} (alpha={lora_alpha})")
    else:
        print("[info] No LoRA path provided; using base model only.")
    pipe.enable_vram_management()
    return pipe


def main():
    output_dir = infer_output_dir(OUTPUT_DIR, LORA_PATH, NUM_FRAMES)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving outputs to: {output_dir}")

    pipe = build_pipeline(LORA_PATH, LORA_ALPHA)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    for idx, row in enumerate(tqdm(entries, desc="Generating")):
        image_path = BASE_DIR / row["image"]
        if not image_path.exists():
            print(f"[warn] Missing {image_path}, skipping.")
            continue

        input_image = Image.open(image_path).resize((WIDTH, HEIGHT))
        seed = SEED + idx
        output_path = output_dir / f"{Path(row['image']).stem}.mp4"

        video = pipe(
            prompt=row["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            input_image=input_image,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            seed=seed,
            tiled=True,
        )
        save_video(video, str(output_path), fps=FPS, quality=QUALITY)


if __name__ == "__main__":
    main()

