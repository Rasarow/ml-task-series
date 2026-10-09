import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache" / "keras"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
SUMMARY_MODEL = "google-t5/t5-small"
SUMMARY_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
ASR_MODEL = "openai/whisper-tiny.en"
ASR_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
LLM_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
LLM_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_SHA256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
LLAMA_TAG = "b11510"
LLAMA_ARCHIVE = "llama-b11510-bin-win-cpu-x64.zip"
LLAMA_SHA256 = "e56a5c6713751e5b5943271558a8abe46e83e0602232ba32a71d2dea789d8fe9"


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1"


def torch_cpu():
    import torch

    torch.set_num_threads(4)
    return torch
