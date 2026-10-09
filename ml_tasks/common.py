import importlib.metadata
import json
import os
import socket
import threading
import time
from pathlib import Path


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def configure_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def block_connection(*args, **kwargs):
        raise RuntimeError("Network connection attempted in offline mode. Run prepare.py first.")

    socket.socket.connect = block_connection
    socket.socket.connect_ex = block_connection


class Measurement:

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak_mb = 0.0
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        import psutil

        while not self.stop.is_set():
            total = 0
            try:
                processes = [self.process] + self.process.children(recursive=True)
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.peak_mb = max(self.peak_mb, total / (1024 * 1024))
            self.stop.wait(0.05)

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()

    def result(self):
        return {
            "wall_seconds": round(self.elapsed, 4),
            "sampled_peak_process_rss_mb": round(self.peak_mb, 2),
        }


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def existing_file(value):
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path
