import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from .config import torch_cpu
from .common import Measurement, existing_file


@lru_cache(maxsize=1)
def load_model():
    torch_cpu()
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    return (model, weights.transforms(), weights.meta["categories"])


def detect_video(path, output_video, threshold=0.5, stride=1, max_frames=None):
    existing_file(path)
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive.")
    output_video = Path(output_video)
    if Path(path).resolve() == output_video.resolve():
        raise ValueError("The output video must differ from the input.")
    with Measurement() as measurement:
        import cv2

        start = time.perf_counter()
        model, transforms, categories = load_model()
        load_seconds = time.perf_counter() - start
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot decode video: {path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            capture.release()
            raise ValueError("Video has no valid frame rate.")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        records = []
        counts = Counter()
        index = 0
        inference_seconds = 0.0
        loop_start = time.perf_counter()
        torch = torch_cpu()
        try:
            while max_frames is None or index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise ValueError("Cannot create MP4 output with the available codec.")
                if index % stride == 0:
                    start = time.perf_counter()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
                    with torch.inference_mode():
                        prediction = model([transforms(tensor)])[0]
                    inference_seconds += time.perf_counter() - start
                    detections = []
                    for box, score, label_id in zip(
                        prediction["boxes"].tolist(),
                        prediction["scores"].tolist(),
                        prediction["labels"].tolist(),
                    ):
                        if score < threshold:
                            continue
                        label = categories[label_id]
                        detections.append(
                            {
                                "label": label,
                                "label_id": label_id,
                                "score": round(score, 6),
                                "box_xyxy": [round(value, 2) for value in box],
                            }
                        )
                        counts[label] += 1
                        x1, y1, x2, y2 = [int(value) for value in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label}: {score:.2f}",
                            (x1, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 210, 0),
                            1,
                        )
                    records.append(
                        {
                            "frame": index,
                            "timestamp_seconds": round(index / source_fps, 4),
                            "detections": detections,
                        }
                    )
                writer.write(frame)
                index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if not records:
            raise ValueError("No video frames were decoded.")
        loop_seconds = time.perf_counter() - loop_start
    return {
        "task": "video_object_detection",
        "model": "SSDLite320 MobileNetV3 Large COCO_V1",
        "framework": "PyTorch/TorchVision",
        "input_file": str(path),
        "output_video": str(output_video),
        "source_fps": source_fps,
        "decoded_frames": index,
        "inferred_frames": len(records),
        "stride": stride,
        "score_threshold": threshold,
        "detection_counts_across_frames": dict(counts),
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "inference_fps": round(len(records) / inference_seconds, 4),
        "processing_fps": round(index / loop_seconds, 4),
        "frames": records,
        **measurement.result(),
    }
