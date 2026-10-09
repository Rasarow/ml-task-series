Local machine learning applications

This project contains English text summarization, English speech recognition, image classification, video object detection and a local language model. All inference runs on the CPU. PyTorch and TensorFlow are both used. Existing model weights are reused without training or fine-tuning.

The tested environment is Windows x64 with Python 3.12. No provider API key is required.

Installation

Run the following command with Python 3.12 from the project directory.

python setup.py

Setup creates .venv, installs the packages and downloads sample media, pretrained weights and the local language-model runtime. The Qwen model download is about 2.1 GB. Internet is required for preparation.

Demo

run.cmd demo.py --offline

The demo saves results/evaluation.json and an annotated video at results/generated/detected_video.mp4. Offline mode blocks Python network connections. The prepared models must be available before using offline mode.

Individual applications

run.cmd main.py --offline summarize --file data/article.txt
run.cmd main.py --offline transcribe --file data/downloads/speech1.wav
run.cmd main.py --offline classify --file data/downloads/dog.jpg
run.cmd main.py --offline detect --file data/downloads/vtest.avi --video-output results/generated/video.mp4 --max-frames 90
run.cmd main.py --offline llm --prompt "Explain machine learning in two sentences."

Use main.py --help or the selected command followed by --help to see the available options. Every application can save its JSON result with --output.

Files

main.py provides the command interface. ml_tasks contains the model functions. prepare.py downloads the models and samples. demo.py runs the small evaluation set. REPORT.txt explains the methods, data, metrics and limitations. SOURCES.txt records model and example sources.

Model weights, downloaded samples, the Python environment and generated video are excluded from version control. Setup recreates them. The repository contains a measured evaluation result; it is a small demonstration rather than a benchmark.
