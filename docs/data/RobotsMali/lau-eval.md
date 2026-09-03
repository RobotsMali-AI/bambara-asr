---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float64
  - name: bam
    dtype: string
  - name: french
    dtype: string
  - name: asr-ctc
    dtype: string
  - name: asr-tdt
    dtype: string
  - name: asr-mt-ctc
    dtype: string
  - name: asr-mt-tdt
    dtype: string
  - name: st-ctc
    dtype: string
  - name: st-tdt
    dtype: string
  - name: lau-tdt-k1
    dtype: string
  - name: lau-ctc-k1
    dtype: string
  - name: lau-tdt-k5
    dtype: string
  - name: lau-ctc-k5
    dtype: string
  - name: lau-tdt-k0.2
    dtype: string
  - name: lau-ctc-k0.2
    dtype: string
  - name: lau-tdt-mse-k1
    dtype: string
  - name: lau-ctc-mse-k1
    dtype: string
  - name: cluster_label
    dtype: string
  splits:
  - name: test
    num_bytes: 121749438
    num_examples: 1218
  download_size: 117816418
  dataset_size: 121749438
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# LAU eval dataset

This dataset was created while evaluating and comparing the models trained with Listen Attend Understand regularization and our [E2E-ST model](https://huggingface.co/RobotsMali/st-soloni-114m-tdt-ctc).
The audio is from jeli-asr test set; the regularization loss weight lambda in the paper is represented by the character "k" in the fields of this dataset, each field represent a model with a specific decoding strategy (CTC or TDT)

---
## Citation

```bibtex
@misc{diarra2026listenattendunderstandregularization,
      title={Listen, Attend, Understand: a Regularization Technique for Stable E2E Speech Translation Training on High Variance labels}, 
      author={Yacouba Diarra and Michael Leventhal},
      year={2026},
      eprint={2601.01121},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.01121}, 
}

```

