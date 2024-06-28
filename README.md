# Deep-Learning-SP24-Final
# Experiment Instructions - Team 17

This repository contains the code necessary to reproduce the experiment conducted by Team 17. Please follow the instructions below to run the Python programs in the correct sequence.

## Prerequisites

- Python 3.x
- Necessary Python libraries (listed in `requirements.txt`)

## Instructions

Run the following Python programs in sequential order:

### 1. Data Preprocessing

**File Path:** `Segmentation/data_preprocessing.py`

**Description:** Preprocess the training data. This script stores images and masks of each frame in the training set into separate files for training the semantic segmentation model.

### 2. Train UNet Model

**File Path:** `Segmentation/unet_model.ipynb`

**Description:** Train the UNet semantic segmentation model on the preprocessed dataset and save the checkpoint of the trained model.

### 3. Generate Masks using UNet

**File Path:** `Segmentation/UNet_generateMasks.ipynb`

**Description:** Load a UNet model from the checkpoint and perform inference on the test dataset to generate masks for the first 11 frames of each video in the test dataset.

### 4. Train Conv-LSTM Video Prediction Model

**File Path:** `VideoPrediction/main.py`

**Description:** Train a Conv-LSTM video prediction model on semantic segmentation data and save the checkpoint of the trained model.

### 5. Generate Answer using Conv-LSTM

**File Path:** `VideoPrediction/generate_answer.py`

**Description:** Load a Conv-LSTM model from the checkpoint and perform inference on the semantic mask data of the test dataset.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Team 17 members for their contributions
- New York University

