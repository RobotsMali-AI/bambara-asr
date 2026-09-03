# Bambara ASR

Research code, training configurations, evaluation resources, and release documentation for RobotsMali's Bambara speech models. The repository began with general-purpose automatic speech recognition (ASR) and now also covers speech translation, spoken-language understanding (SLU), and narrow models intended for concrete applications.

## Projects

| Directory | Purpose |
| --- | --- |
| [`early-experiments/`](early-experiments/) | First-generation (`v0`) Bambara ASR experiments. |
| [`kunkado/`](kunkado/) | `v1` ASR fine-tuning on spontaneous, code-switched radio speech. |
| [`afvoices/`](afvoices/) | `v2` ASR experiments on the African Next Voices Bambara corpus. |
| [`an-be-kalan-bench/`](an-be-kalan-bench/) | Child-speech and educational-reading ASR. |
| [`lau/`](lau/) | Bambara-to-French speech translation with semantic regularization. |
| [`quartznum/`](quartznum/) | Compact ASR specialized for spoken numbers and amounts. |
| [`slu-ic-slot-filling/`](slu-ic-slot-filling/) | Banking-domain intent classification and slot filling. |
| [`rlnf/`](rlnf/) | Experimental reinforcement learning from human feedback for ASR. |
| [`utils/`](utils/) | Shared training, preprocessing, tokenizer, and model-card tools. |

Each project has its own dependencies and instructions. Start with its README and install only its `requirements.txt`; most training workflows require CUDA, PyTorch, and NVIDIA NeMo.

## From Models to Applications

Recent work prioritizes small, task-specific speech systems over a single general model. [`mobilebamspeech`](https://github.com/RobotsMali-AI/mobilebamspeech) demonstrates on-device ASR, SLU, and TTS integration, while [`mobileBAMking`](https://github.com/RobotsMali-AI/mobileBAMking) applies the number recognizer and banking SLU model to a voice-first mobile-banking proof of concept.

Model training and mobile inference intentionally live in separate repositories. [`NeMoOnnxSharp`](https://github.com/RobotsMali-AI/NeMoOnnxSharp) provides the .NET/NativeAOT ONNX Runtime pipeline used to deploy compatible NeMo ASR and SLU exports on Android.

## Models, Data, and Documentation

Released checkpoints and datasets are published under the [`RobotsMali` Hugging Face organization](https://huggingface.co/RobotsMali). Local copies of Hub cards live in [`docs/models/`](docs/models/) and [`docs/data/`](docs/data/); mappings are recorded in [`docs/huggingface.yaml`](docs/huggingface.yaml). Use `python utils/python/hf_cards.py --help` to synchronize a card.

Some checkpoints created with NeMo 2.5.0 need a configuration patch when loaded by newer NeMo releases. See the affected model cards and [the upstream NeMo issue](https://github.com/NVIDIA-NeMo/Speech/issues/15658).

## Project Status

RLNF is a proof-of-concept and has been largely inactive since late 2025. Current work focuses on ASR evaluation, speech translation, narrow-domain models, and applications built on released checkpoints.

## Contributing and License

See [`AGENTS.md`](AGENTS.md) for contributor guidance. Open an issue or pull request for bugs, reproducibility improvements, or documentation corrections. Repository code is MIT-licensed; datasets and model checkpoints have their own licenses stated on their Hugging Face cards.
