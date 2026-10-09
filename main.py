import argparse
import json
import os
import sys
from pathlib import Path
from ml_tasks.common import save_json, configure_offline


def main():
    parser = argparse.ArgumentParser(description="Four content tasks + a local LLM. CPU inference.")
    parser.add_argument("--offline", action="store_true", help="Use cached models only.")
    commands = parser.add_subparsers(dest="task", required=True)
    summary = commands.add_parser(
        "summarize", help="Summarize English text (max 512 input tokens)."
    )
    source = summary.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    summary.add_argument("--max-new-tokens", type=int, default=64)
    audio = commands.add_parser("transcribe", help="Transcribe an English WAV or FLAC file.")
    audio.add_argument("--file", type=Path, required=True)
    image = commands.add_parser("classify", help="Classify an image into ImageNet classes.")
    image.add_argument("--file", type=Path, required=True)
    image.add_argument("--top-k", type=int, default=5)
    video = commands.add_parser("detect", help="Detect COCO objects in video frames.")
    video.add_argument("--file", type=Path, required=True)
    video.add_argument("--video-output", type=Path, required=True)
    video.add_argument("--threshold", type=float, default=0.5)
    video.add_argument("--stride", type=int, default=1)
    video.add_argument("--max-frames", type=int)
    llm = commands.add_parser("llm", help="Run Qwen locally through llama.cpp.")
    llm.add_argument("--prompt", required=True)
    llm.add_argument("--max-new-tokens", type=int, default=160)
    for command in (summary, audio, image, video, llm):
        command.add_argument("--output", type=Path, help="Optional JSON output file.")
    args = parser.parse_args()
    if args.offline:
        configure_offline()
    try:
        if args.task == "summarize":
            from ml_tasks.summarization import summarize

            text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
            result = summarize(text, args.max_new_tokens)
        elif args.task == "transcribe":
            from ml_tasks.audio import transcribe

            result = transcribe(args.file)
        elif args.task == "classify":
            from ml_tasks.image import classify

            result = classify(args.file, args.top_k)
        elif args.task == "detect":
            from ml_tasks.video import detect_video

            result = detect_video(
                args.file, args.video_output, args.threshold, args.stride, args.max_frames
            )
        else:
            from ml_tasks.llm import generate

            result = generate(args.prompt, args.max_new_tokens)
        if args.output:
            save_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
