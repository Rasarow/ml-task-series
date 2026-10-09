import math
import time
from functools import lru_cache
from .config import ASR_MODEL, ASR_REVISION, offline, torch_cpu
from .common import Measurement, existing_file


@lru_cache(maxsize=1)
def load_model():
    torch_cpu()
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    options = dict(revision=ASR_REVISION, local_files_only=offline())
    processor = AutoProcessor.from_pretrained(ASR_MODEL, **options)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL, use_safetensors=True, **options
    ).eval()
    return (processor, model)


def read_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(existing_file(path), dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Audio file is empty.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid sample values.")
    mono = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor).astype("float32")
    return (mono, 16000)


def transcribe(path):
    existing_file(path)
    with Measurement() as measurement:
        audio, rate = read_audio(path)
        start = time.perf_counter()
        processor, model = load_model()
        load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        torch = torch_cpu()
        transcripts = []
        for offset in range(0, len(audio), 30 * rate):
            segment = audio[offset : offset + 30 * rate]
            features = processor(
                segment, sampling_rate=rate, return_tensors="pt", return_attention_mask=True
            )
            with torch.inference_mode():
                generated = model.generate(
                    input_features=features.input_features,
                    attention_mask=features.attention_mask,
                    max_new_tokens=224,
                    do_sample=False,
                )
            transcripts.append(
                processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            )
        inference_seconds = time.perf_counter() - start
    duration = len(audio) / rate
    return {
        "task": "speech_to_text",
        "model": ASR_MODEL,
        "revision": ASR_REVISION,
        "framework": "PyTorch + Transformers",
        "input_file": str(path),
        "transcript": " ".join(transcripts),
        "duration_seconds": round(duration, 4),
        "chunks": len(transcripts),
        "sample_rate": rate,
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "real_time_factor": round(inference_seconds / duration, 4),
        **measurement.result(),
    }
