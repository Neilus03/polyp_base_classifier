
# Polyp Base Classifier

## Introduction

This repository contains code for a Polyp Image Classification pipeline built using PyTorch. The pipeline includes data loading, model training, and evaluation steps. It uses a custom EfficientNet-based neural network for the classification task.

## Features

- **Data Loading**: Custom PyTorch `Dataset` class for loading polyp images and labels.
- **Model**: Custom neural network model based on EfficientNet.
- **Training**: Script for training the model on polyp image data.
- **Evaluation**: Script for evaluating the model performance using accuracy, precision, recall, and F1-score metrics.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- EfficientNet-PyTorch
- scikit-learn
- PIL
- pandas

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Neilus03/polyp_base_classifier.git
cd polyp_base_classifier
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place the polyp image data in the `POLYP_DATA` directory and update the CSV files with image names and labels.
   
2. **Training**: Run the training script:

    ```bash
    python train.py
    ```

    This will train the model and save the model parameters as `custom_efficientnet.pth`.

3. **Evaluation**: Run the evaluation script:

    ```bash
    python evaluation.py
    ```

    This will output the performance metrics for the trained model.

