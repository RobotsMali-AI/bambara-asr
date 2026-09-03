---
language:
- bm
library_name: nemo
license: cc-by-4.0
base_model: RobotsMali/stt-bm-quartznet15x5-v2
tags:
- automatic-speech-recognition
- speech
- audio
- CTC
- QuartzNet
- Bambara
- spoken-numbers
- NeMo
metrics:
- wer
pipeline_tag: automatic-speech-recognition
model-index:
- name: quartznum-v0
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: MobileBAMking spoken-number test split
      type: mobilebamking-spoken-numbers
      split: test
      args:
        language: bm
    metrics:
    - name: Test WER
      type: wer
      value: 17.241379618644714
---

# QuartzNum v0

QuartzNum is an 18M-parameter QuartzNet 15x5 CTC model specialized for recognizing **Bambara spoken amounts, account numbers, and phone numbers**. It is fine-tuned from [`RobotsMali/stt-bm-quartznet15x5-v2`](https://huggingface.co/RobotsMali/stt-bm-quartznet15x5-v2) and is intended as a narrow component, not a replacement for general-purpose Bambara ASR.

The model is deployed in the [`mobilebamspeech`](https://github.com/RobotsMali-AI/mobilebamspeech) integration demo and the voice-first [`mobileBAMking`](https://github.com/RobotsMali-AI/mobileBAMking) proof of concept. [`NeMoOnnxSharp`](https://github.com/RobotsMali-AI/NeMoOnnxSharp) provides the ONNX Runtime/.NET inference path used on Android.

## Intended Use

Use QuartzNum to transcribe short, isolated number expressions in the vocabulary and speaking conditions represented by the banking voice collection. Pass its text through a Bambara number parser/normalizer and validate the resulting value before application use.

Do not use this research checkpoint by itself to authorize transactions, identify speakers, or transcribe unrestricted speech.

## Use with NVIDIA NeMo

```bash
pip install "nemo-toolkit[asr]"
```

```python
from nemo.collections.asr.models import EncDecCTCModel

model = EncDecCTCModel.from_pretrained("RobotsMali/quartznum-v0")
model.eval()
predictions = model.transcribe(["spoken_number.wav"])
print(predictions[0].text)
```

Input audio is converted by the model preprocessor to mono 16 kHz features. Output is a greedy CTC hypothesis over a lowercase character vocabulary containing the Bambara letters `ŋ`, `ɔ`, `ɛ`, and `ɲ`.

## NeMo 2.5.0 Compatibility

This checkpoint was created with NeMo 2.5.0. Loading it with newer NeMo versions (observed with 2.7.x) can fail because the strict decoding schema expects `key_phrase_items_list`; see [NVIDIA-NeMo/Speech#15658](https://github.com/NVIDIA-NeMo/Speech/issues/15658). The following workaround was tested with Python 3.12 and allows the 2.5.0 checkpoint to load in the newer environment:

```python
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf

model_name = "RobotsMali/quartznum-v0"
cfg = ASRModel.from_pretrained(model_name, return_config=True)
OmegaConf.set_struct(cfg, False)

for decoder in ("greedy", "beam"):
    boosting_tree = OmegaConf.select(cfg, f"decoding.{decoder}.boosting_tree")
    if boosting_tree is not None:
        boosting_tree.key_phrase_items_list = None

config_path = Path("patched_config.yaml").resolve()
OmegaConf.save(cfg, config_path)
model = ASRModel.from_pretrained(
    model_name=model_name,
    override_config_path=str(config_path),
    strict=False,
)
config_path.unlink()
```

## Architecture and Training

QuartzNet 15x5 uses one-dimensional time-channel separable convolutions and a character CTC decoder. Training used NeMo 2.5.0, BF16 mixed precision, batches of 16, Adam at `5e-4`, and cosine annealing with 80 warm-up steps. The run allowed up to 50 epochs with early stopping patience of 15. The [training script, configuration, and archived result](https://github.com/RobotsMali-AI/bambara-asr/tree/main/quartznum) are public.

Training clips were derived from the 588-command MobileBAMking voice collection. For transfer commands, the preparation pipeline extracts separately recorded amount and account/phone-number audio and normalizes the labels with `bambara-normalizer`. This application-collected training set is not currently published as a standalone Hugging Face dataset.

## Evaluation

The archived held-out evaluation reports:

| Split | Decoding | WER (%) ↓ | Test loss |
| --- | --- | ---: | ---: |
| MobileBAMking spoken-number test split | Greedy CTC | **17.24** | 36.29 |

WER measures token transcription, not end-to-end numeric-value accuracy. The split is small, narrow-domain, and drawn from the same collection workflow as training, so the figure should not be generalized to arbitrary speakers or acoustic conditions.

## Limitations

- Long digit strings can be costly even when only one character is wrong; validate parsed values explicitly.
- Accuracy may degrade with noise, overlap, dialects, recording hardware, or number forms absent from training.
- The model does not provide confidence calibration, speaker verification, or transaction safety controls.
- General Bambara words and sentences are outside its intended domain.

## License

The model is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
