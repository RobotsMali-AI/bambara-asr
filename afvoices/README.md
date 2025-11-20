# **README — afvoices**

This folder contains the code, configurations, and resources used to fine-tune four ASR models, 3 from NVIDIA’s **Parakeet** family + QuartzNet. All experiments here were conducted on a **pre-completion subset** of the **RobotsMali/afvoices** dataset.
This subset is the one referenced in our dataset paper (insert link here), and is provided for full reproducibility.

## **Contents**

### **1. pre-manifests/**

This directory contains the **training/test manifests** for the *pre-completion* version of the afvoices dataset.
We used this subset because the dataset was still undergoing annotation when the experiments were run. The final dataset released on Hugging Face includes additional validated samples, but the results reported in our paper correspond exactly to the manifests stored here.

This folder is provided to ensure *full reproducibility* of the WER results reported in the dataset paper. Anyone wishing to evaluate models on the same test set can directly use the manifests provided here.

### **2. config/**

YAML configuration files used to train and fine-tune the four Parakeet models.
Each configuration corresponds to a model variant trained on the same pre-completion subset.

### **3. scripts/**

Utility scripts used throughout the experiments, including. These scripts were used as-is for the experiments whose results appear in the dataset paper.

### **4. tokenizers/**

SentencePiece models used for tokenization in all afvoices v2 models.

### **5. requirements.txt**

Python dependencies for running the experiments.

## **Models**

The ASR models trained from this folder are the **v2 versions** available on Hugging Face under the **RobotsMali** organization.
A single released model might have been trained using one or more of its associated configs. More details on their [HF model cards](https://huggingface.co/RobotsMali/models)

