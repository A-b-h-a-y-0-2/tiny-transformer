# Tiny Transformer Project for ECG Classification and CIFAR-100

## Overview

This project involves two main tasks:

### Task 1: Recreation of EdgeViT for CIFAR-100
1. **Model Recreation**: Recreated the EdgeViT model, designed specifically for image classification, and trained it on the CIFAR-100 dataset.
   
2. **Fine-tuning Approaches**:
   - **A) Initial Hyperparameter Fine-tuning**: Based on the approaches outlined in relevant research papers on fine-tuning models for CIFAR-10 and CIFAR-100 datasets.
   - **B) Optuna-based Hyperparameter Tuning**: Utilized Optuna to perform automated hyperparameter optimization, finding the best set of parameters for fine-tuning.

3. **EfficientFormer Model**: Imported the EfficientFormer model from Hugging Face and fine-tuned it based on the optimal hyperparameters identified via Optuna.

### Task 2: MSW-Transformer for ECG Classification
1. **Model Recreation**: Recreated the transformer backbone from the paper [MSW-Transformer: Multi-Scale Shifted Windows Transformer Networks for 12-Lead ECG Classification](https://arxiv.org/pdf/2306.12098).

2. **Data Loading Setup**: Implemented data loading based on findings from the paper ["Cardiac Arrhythmia Classification Using Advanced Deep Learning Techniques"](https://arxiv.org/pdf/2311.04229), which provides insights into handling ECG datasets.

3. **Training**: Due to limited computational resources, the model has not been trained yet. However, the implementation closely follows the architectural details provided in the paper.

---

## Intuition Behind the MSW-Transformer

The MSW-Transformer applies a **multi-scale windowed attention mechanism**. It partitions the input ECG signals into smaller overlapping windows, processes them, and shifts the windows in subsequent layers. This technique enables the model to capture both local and global features of the ECG data while keeping computational costs low.

### Why This Approach?
1. **Multi-Scale Feature Extraction**: ECG signals exhibit variability at different scales. The MSW-Transformer's design helps capture both short-term variations and long-term trends efficiently.
   
2. **Computational Efficiency**: Unlike traditional transformers that use global attention, the MSW-Transformer focuses on local dependencies while reducing computational overhead, making it more suitable for edge devices.

---

## Implementation Reasoning

1. **Window Partitioning**: The input ECG signals are split into overlapping windows, allowing the model to focus on local patterns while maintaining context.

2. **Relative Positional Encoding**: We employ relative positional encoding to capture spatial relationships between ECG signal segments, which improves the model’s ability to handle sequences without being tied to fixed positions.

3. **Attention Mechanism**: The attention module processes windows with added dropout to prevent overfitting. This helps the model prioritize important parts of the ECG signals.

4. **Multi-Scale Feature Fusion**: After window processing, we fuse information from different scales using the `MSWFeatureFusion` module. This combines both short-term variations and long-term trends into a unified representation.

5. **Patch Embedding**: Signal patches are converted into embeddings via the `PatchEmbed` layer, making it easier for the transformer to handle 1D ECG signals in a framework typically used for vision tasks.

---

## Training Details

- **Optimizer**: Adam with an initial learning rate of 0.0001.
- **Learning Rate Decay**: Gradually reduce the learning rate to fine-tune the model.
- **Early Stopping**: Monitor validation loss and stop training if it does not improve after a set number of epochs.
- **Data Augmentation**: Techniques like noise addition and signal scaling are applied to improve generalization.

---

## Relevant Papers

1. [Deep Learning for ECG Arrhythmia Detection and Classification: An Overview of Progress for the Period 2017–2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10542398/)
   
2. [Cardiac Arrhythmia Classification Using Advanced Deep Learning Techniques on Digitized ECG Datasets](https://www.mdpi.com/1424-8220/24/8/2484)

---

Feel free to reach out to us if  you have any questions about our project!
- **Abhay Chaudhary** - [GitHub Profile](https://github.com/A-b-h-a-y-0-2)
- **DivyRaj Saini** - [GitHub Profile](https://github.com/Dead-Bytes)


