# **README — early-experiments**

This folder contains the configurations and code used for our earliest ASR training runs.
The resulting models are the ones suffixed **(v0)** on RobotsMali’s Hugging Face account.

These experiments were conducted on the **RobotsMali/bam-asr-early** dataset, which was itself derived largely from **RobotsMali/jeli-asr**. Both datasets are publicly available on Hugging Face.

The only exception is **soloba-ctc-v0.0.0.yaml**, which was used to train a model on a combined dataset consisting of **bam-asr-early + 120 hours of automatically annotated speech** from **RobotsMali/kunkado**.

## **Contents**

### **1. configs/**

YAML configuration files for the early-generation models.
Each file corresponds to one experimental setup used to produce a specific v0 model.
A single released model might have been trained using one or more of its associated configs. More details on their [HF model cards](https://huggingface.co/RobotsMali/models)

### **2. early-asr-tokenizer/**

SentencePiece models used by all v0 systems.

---

## **Datasets Used**

### **• RobotsMali/bam-asr-early**

Main dataset used for the v0 experiments.

### **• RobotsMali/kunkado**

Only used in the special configuration **soloba-ctc-v0.0.0.yaml**, where 120 hours of the kunkado auto-labeled portion were added.
