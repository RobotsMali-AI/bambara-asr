# **README — kunkado**

This folder contains the configurations, code, and tokenizers used for fine-tuning ASR models on the **RobotsMali/kunkado** dataset.
Kunkado is a speech corpus compiled to reflect **day-to-day Bambara spoken in Bamako and surrounding areas**, spontaneous radios recordings with automatic transcriptions and 40 hours human-reviwed subset, feeaturing background noise, music and a significant amount of code switching.

The experiments here correspond to the models suffixed **(v1)** on RobotsMali’s Hugging Face account.

These v1 models are fine-tuned from the earlier **v0** systems documented in the `early-experiments` folder.

---

## **Contents**

### **1. config/**

YAML configuration files used for fine-tuning models on kunkado.
Each config corresponds to a specific experiment on the dataset’s **40-hour human-reviewed subset**, which is the version used for the v1 models.

### **2. scripts/**

Helper utilities used during experimentation, including dataset preparation, normalization.

### **3. tokenizers/**

SentencePiece tokenizers used by all v1 models produced from this folder.

---

## **Models**

The resulting models are available under the **RobotsMali** organization on Hugging Face and are suffixed **v1**.
A single released model might have been trained using one or more of its associated configs.

---

## **Future Work**

We also plan to fine-tune the **v2 models** (from the afvoices experiments) on the kunkado dataset.

**Coming soon.**
