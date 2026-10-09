import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from ml_tasks.config import ROOT
from ml_tasks.common import save_json, configure_offline, package_versions
from ml_tasks.metrics import rouge, speech_metrics, normalize_words


def read_samples(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def run_demo(offline=False):
    os.chdir(ROOT)
    print(f"Python {platform.python_version()}", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("Use Python 3.12 and run.cmd for this project.", flush=True)
    if offline:
        configure_offline()
    from ml_tasks.summarization import summarize
    from ml_tasks.audio import transcribe
    from ml_tasks.image import classify
    from ml_tasks.video import detect_video
    from ml_tasks.llm import generate

    target = Path("results/generated")
    results = {
        "summarization": [],
        "speech_to_text": [],
        "image_classification": [],
        "local_llm": [],
    }
    print("1/5 English summarization", flush=True)
    for sample in read_samples("text_samples.json"):
        result = summarize(sample["text"])
        result.update(
            id=sample["id"],
            reference=sample["reference"],
            quality_metrics=rouge(sample["reference"], result["summary"]),
        )
        results["summarization"].append(result)
    print("2/5 Speech-to-text", flush=True)
    for sample in read_samples("audio_samples.json"):
        result = transcribe(Path("data/downloads") / sample["file"])
        result.update(
            id=sample["id"],
            reference=sample["reference"],
            quality_metrics=speech_metrics([sample["reference"]], [result["transcript"]]),
        )
        results["speech_to_text"].append(result)
    print("3/5 TensorFlow image classification", flush=True)
    for sample in read_samples("image_samples.json"):
        result = classify(Path("data/downloads") / sample["file"])
        labels = sample["acceptable_class_ids"]
        result.update(
            id=sample["id"],
            acceptable_class_ids=labels,
            top1_demo_match=result["predictions"][0]["class_id"] in labels,
            top5_demo_match=any((item["class_id"] in labels for item in result["predictions"])),
        )
        results["image_classification"].append(result)
    print("4/5 Video object detection", flush=True)
    video = detect_video(
        Path("data/downloads/vtest.avi"),
        target / "detected_video.mp4",
        threshold=0.25,
        stride=3,
        max_frames=90,
    )
    save_json(video, target / "video.json")
    results["video_object_detection"] = video
    print("5/5 Local Qwen 3B", flush=True)
    for sample in read_samples("llm_samples.json"):
        result = generate(sample["prompt"])
        result.update(id=sample["id"], expected_exact=sample["expected_exact"])
        if sample["expected_exact"] is not None:
            result["exact_match"] = normalize_words(result["answer"]) == normalize_words(
                sample["expected_exact"]
            )
        results["local_llm"].append(result)
    summary_metrics = {
        metric: round(
            sum((item["quality_metrics"][metric] for item in results["summarization"])) / 3, 6
        )
        for metric in ("rouge1_f1", "rouge2_f1", "rougeL_f1")
    }
    total_audio_seconds = sum((item["duration_seconds"] for item in results["speech_to_text"]))
    audio_metrics = speech_metrics(
        [item["reference"] for item in results["speech_to_text"]],
        [item["transcript"] for item in results["speech_to_text"]],
    )
    audio_metrics["aggregate_real_time_factor"] = round(
        sum((item["inference_seconds"] for item in results["speech_to_text"]))
        / total_audio_seconds,
        4,
    )
    checked = [item for item in results["local_llm"] if "exact_match" in item]
    versions = package_versions(("torch", "torchvision", "tensorflow", "transformers", "keras"))
    missing_metadata = [name for name, version in versions.items() if version is None]
    evaluation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "offline_mode": offline,
        "python_network_connections_blocked": offline,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "framework_versions": versions,
        "missing_framework_version_metadata": missing_metadata,
        "limitations": [
            "Demonstration samples only, not a benchmark.",
            "Speech is synthetic, not recordings of human speakers.",
            "Image labels accept several cat subclasses; not official ImageNet accuracy.",
            "Video has no ground truth boxes; mAP/precision/recall were not measured.",
            "Counts repeat the same objects across frames; no object tracking.",
            "RSS includes loaded libraries/models from earlier tasks and native children.",
            "First inference can include warm-up overhead; timing is not a repeated benchmark.",
        ],
        "aggregate_metrics": {
            "summarization": {"sample_count": 3, **summary_metrics},
            "speech_to_text": {"sample_count": 2, **audio_metrics},
            "image_classification": {
                "sample_count": 2,
                "top1_demo_match_rate": sum(
                    (item["top1_demo_match"] for item in results["image_classification"])
                )
                / 2,
                "top5_demo_match_rate": sum(
                    (item["top5_demo_match"] for item in results["image_classification"])
                )
                / 2,
            },
            "video_object_detection": {
                key: video[key]
                for key in (
                    "decoded_frames",
                    "inferred_frames",
                    "inference_fps",
                    "processing_fps",
                    "detection_counts_across_frames",
                )
            },
            "local_llm": {
                "total_prompts": 3,
                "exact_match_prompts": len(checked),
                "exact_match_rate": sum((item["exact_match"] for item in checked)) / len(checked),
            },
        },
        "results": results,
    }
    save_json(evaluation, "results/evaluation.json")
    if missing_metadata:
        print(
            "Version metadata unavailable (recorded as null): " + ", ".join(missing_metadata),
            flush=True,
        )
    print(json.dumps(evaluation["aggregate_metrics"], indent=2), flush=True)
    print("All five applications completed. Results: results/evaluation.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run_demo(args.offline)
