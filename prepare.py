import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from ml_tasks.config import (
    ROOT,
    LLM_FILE,
    LLM_REPO,
    LLM_REVISION,
    LLM_SHA256,
    LLAMA_TAG,
    LLAMA_ARCHIVE,
    LLAMA_SHA256,
)

SAMPLE_URLS = {
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "cats.png": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data/vtest.avi",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, expected_hash=None):
    destination = Path(destination)
    if destination.exists() and (not expected_hash or sha256(destination) == expected_hash):
        print(f"Already available: {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ml-task-series/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as output:
        length = int(response.headers.get("Content-Length", 0))
        received = 0
        next_message = 256 * 1024 * 1024
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received >= next_message:
                print(
                    f"  {received / 1024 ** 2:.0f} MiB / {length / 1024 ** 2:.0f} MiB", flush=True
                )
                next_message += 256 * 1024 * 1024
    if expected_hash and sha256(temporary) != expected_hash:
        raise RuntimeError(f"SHA256 verification failed: {destination.name}")
    os.replace(temporary, destination)


def prepare_samples():
    target = ROOT / "data" / "downloads"
    for name, url in SAMPLE_URLS.items():
        download(url, target / name)
    if not all(((target / f"speech{i}.wav").is_file() for i in (1, 2))):
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "make_audio.ps1"),
                "-OutputDirectory",
                str(target),
            ],
            check=True,
        )
    manifest = {
        name: {"source": url, "sha256": sha256(target / name)} for name, url in SAMPLE_URLS.items()
    }
    manifest["speech1.wav"] = {
        "source": "Locally synthesized Windows English voice",
        "sha256": sha256(target / "speech1.wav"),
    }
    manifest["speech2.wav"] = {
        "source": "Locally synthesized Windows English voice",
        "sha256": sha256(target / "speech2.wav"),
    }
    (ROOT / "data" / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def prepare_llm():
    archive = ROOT / "runtime" / LLAMA_ARCHIVE
    download(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_ARCHIVE}",
        archive,
        LLAMA_SHA256,
    )
    target = (ROOT / "runtime" / "llama.cpp").resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            resolved = (target / info.filename).resolve()
            if not resolved.is_relative_to(target):
                raise RuntimeError("Archive contains a path outside the runtime directory.")
        source.extractall(target)
    download(
        f"https://huggingface.co/{LLM_REPO}/resolve/{LLM_REVISION}/{LLM_FILE}?download=true",
        ROOT / "models" / LLM_FILE,
        LLM_SHA256,
    )


def prepare_models():
    from ml_tasks.summarization import load_model as summary
    from ml_tasks.audio import load_model as audio
    from ml_tasks.image import load_model as image
    from ml_tasks.video import load_model as video
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    import numpy as np

    for name, loader in [
        ("T5", summary),
        ("Whisper", audio),
        ("MobileNetV2", image),
        ("SSDLite", video),
    ]:
        print(f"Preparing {name} ...", flush=True)
        loader()
    decode_predictions(np.zeros((1, 1000), dtype=np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("Select --samples, --models, and/or --llm.")
    if args.samples:
        prepare_samples()
    if args.models:
        prepare_models()
    if args.llm:
        prepare_llm()
    print("Preparation completed.", flush=True)
