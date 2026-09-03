# Bambara Speech Intent Classification and Slot Filling

This directory adapts NVIDIA NeMo's end-to-end speech intent/slot example to a narrow Bambara mobile-banking domain. The released [`RobotsMali/soloni-ic-slot-fintech-v0`](https://huggingface.co/RobotsMali/soloni-ic-slot-fintech-v0) model maps speech directly to a structured representation containing `scenario`, `action`, and `entities`; it does not first produce a general-purpose transcript.

The model supports the ontology collected for [`mobileBAMking`](https://github.com/RobotsMali-AI/mobileBAMking): navigation, balance queries, account/mobile-money transfers, logout, and a small FAQ path. It is also integrated in [`mobilebamspeech`](https://github.com/RobotsMali-AI/mobilebamspeech).

## Layout

- `prepare_slu_manifest.py`: downloads the archived voice collection, converts audio to mono WAV, normalizes numeric fillers, and creates train/test manifests for SLU, amount, and number tasks.
- `run_speech_intent_slot_train.py`: trains NeMo's `SLUIntentSlotBPEModel` and initializes its encoder from a local or Hugging Face ASR checkpoint.
- `run_speech_intent_slot_eval.py`: runs inference and computes scenario, action, intent, entity, and SLU F1 metrics.
- `checkpoint_averaging.py`: averages selected Lightning checkpoints.
- `configs/soloni-ic-slot-fintech-v0.yaml`: released-model configuration.
- `tokenizer/slu_tokenizer/`: 58-piece SentencePiece tokenizer.
- `data/testResults_soloni-ic-slot-fintech-v0.txt`: archived held-out evaluation output.

## Data Preparation

The checked-in export contains 588 annotated voice commands. Its annotations include 361 `Operation`, 222 `Navigate`, and 5 `FAQ` examples. Treat the data as application-specific and sensitive to class imbalance; do not infer broad banking-language coverage from it.

```bash
cd slu-ic-slot-filling
python prepare_slu_manifest.py \
  data/mobilebamking_voice_collection_export.json data/processed
```

The preparation script downloads referenced recordings, so failed URLs can reduce the resulting split. Inspect its summary before training. Generated audio and manifests should remain local.

## Training and Evaluation

Install a compatible NeMo 2.5.0 environment, then run from this directory because config paths are relative:

```bash
python run_speech_intent_slot_train.py \
  --config-path=configs \
  --config-name=soloni-ic-slot-fintech-v0

python run_speech_intent_slot_eval.py \
  dataset_manifest=data/test.jsonl \
  model_path=/path/to/soloni-ic-slot-fintech-v0.nemo \
  batch_size=16
```

The archived 60-example test run reports **91.67% intent F1**, **75.86% exact entity F1**, and **78.14% SLU F1**, with no syntax-invalid predictions. These figures are in-domain and the split is small; report results with the split size and ontology.

## Deployment and Limitations

The model uses a 17-layer, 512-dimensional Conformer encoder initialized from `soloni-114m-tdt-ctc-v3`, plus a three-layer Transformer decoder and token classifier. It expects mono speech and preprocesses at 16 kHz. Output is a Python-style structured string whose keys and supported actions must be validated before application execution. It is unsuitable for unrestricted financial instructions, unsupported intents, or production transaction authorization without additional data, safeguards, and testing.

The four-graph ONNX deployment path (encoder, embedding, decoder, classifier) is implemented by [`NeMoOnnxSharp`](https://github.com/RobotsMali-AI/NeMoOnnxSharp).

## Attribution

The training and evaluation foundations are adapted from NVIDIA NeMo's [speech intent and slot-filling example](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/slu/speech_intent_slot), including evaluation code derived from the [SLURP project](https://github.com/pswietojanski/slurp).
