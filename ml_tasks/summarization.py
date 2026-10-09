import time
from functools import lru_cache
from .config import SUMMARY_MODEL, SUMMARY_REVISION, offline, torch_cpu
from .common import Measurement


@lru_cache(maxsize=1)
def load_model():
    torch_cpu()
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    options = dict(revision=SUMMARY_REVISION, local_files_only=offline())
    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL, **options)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL, use_safetensors=True, **options
    ).eval()
    return (tokenizer, model)


def summarize(text, max_new_tokens=64):
    if not text.strip():
        raise ValueError("Summary input must contain text.")
    if not 16 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 16 and 256.")
    with Measurement() as measurement:
        start = time.perf_counter()
        tokenizer, model = load_model()
        load_seconds = time.perf_counter() - start
        prompt = "summarize: " + " ".join(text.split())
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        count = tokens["input_ids"].shape[1]
        if count > 512:
            raise ValueError(f"Input has {count} tokens; maximum is 512. Split the document first.")
        start = time.perf_counter()
        torch = torch_cpu()
        with torch.inference_mode():
            output = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
            )
        inference_seconds = time.perf_counter() - start
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "task": "summarization",
        "model": SUMMARY_MODEL,
        "revision": SUMMARY_REVISION,
        "framework": "PyTorch + Transformers",
        "input": text,
        "summary": summary,
        "input_tokens": count,
        "generated_tokens": int(output.shape[1] - 1),
        "word_compression_ratio": round(len(summary.split()) / len(text.split()), 4),
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        **measurement.result(),
    }
