import re
import subprocess
import tempfile
from pathlib import Path
from .config import ROOT, LLM_FILE
from .common import Measurement


def validate_prompt(prompt):
    if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
        raise ValueError("Input must not contain ChatML control tokens.")
    return prompt


def generate(prompt, max_new_tokens=160):
    if not prompt.strip():
        raise ValueError("LLM prompt must contain text.")
    if len(prompt) > 2000:
        raise ValueError("This CPU demo accepts at most 2000 characters per prompt.")
    if not 16 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 16 and 512.")
    model_path = ROOT / "models" / LLM_FILE
    executable = ROOT / "runtime" / "llama.cpp" / "llama-completion.exe"
    if not executable.is_file() or not model_path.is_file():
        raise FileNotFoundError("Local LLM is missing. Run: python prepare.py --llm")
    with tempfile.TemporaryDirectory(prefix="ml-qwen-") as temporary:
        prompt_file = Path(temporary) / "prompt.txt"
        prompt_file.write_text(validate_prompt(prompt), encoding="utf-8")
        args = [
            str(executable),
            "-m",
            str(model_path),
            "-f",
            str(prompt_file),
            "-n",
            str(max_new_tokens),
            "-c",
            "4096",
            "-t",
            "4",
            "-ngl",
            "0",
            "--temp",
            "0",
            "-s",
            "42",
            "--conversation",
            "--single-turn",
            "--system-prompt",
            "You are a helpful assistant. Respond in English. Follow the requested output format exactly.",
            "--no-display-prompt",
            "--simple-io",
            "--no-warmup",
            "--no-escape",
            "--offline",
            "--perf",
            "--log-colors",
            "off",
        ]
        with Measurement() as measurement:
            process = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
    if process.returncode != 0:
        raise RuntimeError("llama.cpp failed: " + process.stderr[-2500:])
    answer = process.stdout.strip().removesuffix("[end of text]").strip()
    if not answer:
        raise RuntimeError("The local LLM returned an empty answer.")
    rate_match = re.search(
        "(?<!prompt )eval time\\s*=.*?([\\d.]+) tokens per second", process.stderr
    )
    token_match = re.search("(?<!prompt )eval time\\s*=.*?/\\s*(\\d+) runs", process.stderr)
    return {
        "task": "local_llm",
        "model": "Qwen2.5-3B-Instruct Q4_K_M",
        "runtime": "llama.cpp b11510 CPU",
        "prompt": prompt,
        "answer": answer,
        "inference_location": "local computer; no provider API",
        "max_new_tokens": max_new_tokens,
        "generated_eval_runs": int(token_match.group(1)) if token_match else None,
        "generation_tokens_per_second": float(rate_match.group(1)) if rate_match else None,
        **measurement.result(),
    }
