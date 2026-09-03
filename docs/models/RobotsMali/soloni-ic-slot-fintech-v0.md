---
language:
- bm
library_name: nemo
license: cc-by-4.0
base_model: RobotsMali/soloni-114m-tdt-ctc-v3
tags:
- audio
- speech
- Bambara
- spoken-language-understanding
- intent-classification
- slot-filling
- Conformer
- Transformer
- NeMo
pipeline_tag: audio-classification
model-index:
- name: soloni-ic-slot-fintech-v0
  results:
  - task:
      name: Speech Intent Classification and Slot Filling
      type: audio-classification
    dataset:
      name: MobileBAMking voice-command test split
      type: mobilebamking-voice-commands
      split: test
      args:
        language: bm
    metrics:
    - name: Intent F1
      type: f1
      value: 91.67
    - name: Entity F1
      type: f1
      value: 75.86
    - name: SLU F1
      type: f1
      value: 78.14
---

# Soloni IC/Slot Fintech v0

`soloni-ic-slot-fintech-v0` is an end-to-end Bambara spoken-language-understanding model for a **small mobile-banking ontology**. It maps audio directly to a structured representation with `scenario`, `action`, and `entities`, without requiring a separate transcript or general-purpose language model.

The checkpoint powers the SLU path in [`mobilebamspeech`](https://github.com/RobotsMali-AI/mobilebamspeech) and the [`mobileBAMking`](https://github.com/RobotsMali-AI/mobileBAMking) research/UX demonstration. Its multi-graph Android inference implementation is available in [`NeMoOnnxSharp`](https://github.com/RobotsMali-AI/NeMoOnnxSharp).

## Supported Domain

The collection contains these actions:

- `transfer_to_account` and `transfer_to_momo`, with `amount` and account/phone-number entities;
- `get_balance`;
- `Maps_to`, with a destination-page entity;
- `logout`;
- `search_faq`, with topic/category entities.

Consumers must treat this as a closed ontology and reject unsupported or malformed output. It is not a conversational banking agent.

## Use with NVIDIA NeMo

```bash
pip install "nemo-toolkit[asr]"
```

```python
from nemo.collections.asr.models import SLUIntentSlotBPEModel

model = SLUIntentSlotBPEModel.from_pretrained(
    "RobotsMali/soloni-ic-slot-fintech-v0"
)
model.eval()
predictions = model.transcribe(["banking_command.wav"])
print(predictions[0])
```

The model preprocesses speech at 16 kHz. Its decoder emits a Python-style serialized structure such as a mapping containing `scenario`, `action`, and an `entities` list. Parse defensively: validate syntax, allow-listed actions, entity types, and application state before executing anything.

## Architecture

The model uses NeMo's `SLUIntentSlotBPEModel`:

- a 17-layer Conformer encoder (`d_model=512`) initialized from [`soloni-114m-tdt-ctc-v3`](https://huggingface.co/RobotsMali/soloni-114m-tdt-ctc-v3);
- a 58-piece SentencePiece tokenizer;
- a three-layer, eight-head Transformer decoder;
- a token classifier trained with negative log-likelihood.

The `.nemo` archive is approximately 475 MB. The architecture is based on NVIDIA NeMo's [speech intent and slot-filling example](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/slu/speech_intent_slot).

## Training Data and Procedure

The archived collection contains 588 annotated Bambara voice commands: 361 `Operation`, 222 `Navigate`, and 5 `FAQ` examples. It is strongly imbalanced and was collected for the MobileBAMking prototype. The preparation pipeline and raw export are in [`slu-ic-slot-filling/`](https://github.com/RobotsMali-AI/bambara-asr/tree/main/slu-ic-slot-filling); referenced recordings are downloaded when manifests are built.

Training used NeMo 2.5.0, batches of 16, an unfrozen pretrained encoder, AdamW, a `5e-4` decoder learning rate, a `1e-4` encoder learning rate, cosine annealing, and up to 50 epochs. See the [released config](https://github.com/RobotsMali-AI/bambara-asr/blob/main/slu-ic-slot-filling/configs/soloni-ic-slot-fintech-v0.yaml).

## Evaluation

The repository retains the complete text output from a 60-example held-out evaluation:

| Metric | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| Scenario | 96.67 | 96.67 | 96.67 |
| Action | 91.67 | 91.67 | 91.67 |
| Combined intent | 91.67 | 91.67 | **91.67** |
| Exact entities | 84.62 | 68.75 | **75.86** |
| Entities, word distance | 85.71 | 70.59 | 77.42 |
| Entities, character distance | 87.50 | 71.79 | 78.87 |
| Overall SLU | 86.60 | 71.19 | **78.14** |

All 60 predictions were syntactically valid. These results are micro-averaged and in-domain; the tiny FAQ class and small test set do not establish broad coverage.

## Limitations and Safety

- The model recognizes only the application ontology and may confidently misroute unfamiliar speech.
- Training data are small, imbalanced, and tied to one collection workflow.
- Entity recall is materially lower than intent accuracy; never assume a missing or malformed amount/account value.
- This checkpoint does not authenticate a user, verify a speaker, confirm consent, or authorize financial activity.
- Any real application must use explicit confirmation, deterministic validation, secure backend controls, audit logging, and broader evaluation.

## License

The model is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source voice collection may have additional privacy or distribution constraints and is not published as a standalone dataset.
