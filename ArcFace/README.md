# ArcFace Celebrity Retrieval Pipeline

This repository contains a PyTorch implementation of an ArcFace-based machine learning pipeline for cross-domain celebrity image retrieval. The codebase was developed to train a lightweight Convolutional Neural Network (CNN) to match real-world query photos of celebrities to synthetic gallery images.

## Overview

The code implements a custom `CelebrityModel` featuring a CNN backbone and an Additive Angular Margin (`ArcMarginProduct`) classification head. It evaluates three distinct training strategies to bridge the domain gap between natural and synthetic images:
* **VGGFace2-HQ Only:** Pretraining and testing on high-quality real images.
* **Competition Data Only:** Training a randomly initialized model directly on the competition dataset.
* **Two-Stage Fine-Tuning:** Pretraining on VGGFace2-HQ followed by fine-tuning on the competition dataset.

---

## Prerequisites

Ensure you have a CUDA-enabled GPU for optimal training times, or connect to a VM with better computing capabilities if your machine doesn't meet the requirements. You will need Python installed along with the following libraries:

* `torch`
* `torchvision`
* `numpy`
* `Pillow` (PIL)

You can install the primary dependencies via pip:
`pip install torch torchvision numpy pillow`

---

## Dataset Configuration

Before running the code, you must update the hardcoded directory paths at the top of the script to point to your local dataset locations. 

Update the following variables in your local environment:
* `VGGFace2_DIR`: Path to the VGGFace2-HQ full dataset folder.
* `COMP_TRAIN_DIR`: Path to the competition training set.
* `COMP_QUERY_DIR`: Path to the competition query images (flat directory).
* `COMP_GALLERY_DIR`: Path to the competition gallery images (flat directory).

**Important Data Formatting Note:** The competition query and gallery directories must be "flat" (containing only images, no sub-directories). The evaluation metric relies on the filenames to extract labels. Ensure your query and gallery images are named such that the label can be extracted via `filename.split('_')[0]` (e.g., `123_image.jpg` implies class `123`).

---

## Code Architecture

* **ArcMarginProduct:** The ArcFace classification head that enforces greater intra-class compactness and inter-class separability using an angular margin ($m=0.50$).
* **CelebrityModel:** A lightweight CNN backbone (two convolutional blocks with max-pooling) projecting into a 512-dimensional embedding space, capped with the ArcFace head.
* **Data Loaders:** * `get_vggface_dataloaders`: Splits VGGFace2-HQ into 80/10/10 train/val/test splits using a fixed random seed (`seed=388404`).
  * `FlatImageDataset`: A custom dataset class to load images from unnested directories for the query and gallery sets during inference.

---

## Training Pipelines

The script contains three main executable functions, each representing a different training ablation path:

### 1. `run_vggface_only`
Trains the model exclusively on the 80% training split of VGGFace2-HQ for a specified number of epochs. It evaluates the model using self-retrieval on the held-out 10% test split. 

### 2. `run_vggface_then_competition`
A two-stage pipeline. It first trains the model on the VGGFace2-HQ dataset to learn general facial features. It then swaps the ArcFace classification head for a new one matching the competition classes and fine-tunes the network on the competition training data.

### 3. `run_competition_only`
Trains the model from random initialization directly on the competition training data. It evaluates retrieval performance (Query vs. Gallery) at multiple epoch checkpoints (e.g., 10, 20, 30, 40) to monitor for overfitting and saves the best-performing model.

---

## Usage Instructions

To operate the code, navigate to the `run_pipeline()` function at the very bottom of the script. 

Uncomment the specific pipeline you wish to execute.
It's not the most technically sound way to operate in OOP, but worked fast enough for competition time constraints.
By default, the script is set to run the fastest pipeline (`run_competition_only`):

```python
def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Uncomment ONE of the following lines to run the desired pipeline:
    
    # run_vggface_only(device, batch_size=64, epochs=10)
    # run_vggface_then_competition(device, batch_size=64, vgg_epochs=10, comp_epochs=10)
    run_competition_only(device, batch_size=64, epoch_checkpoints=(10, 20, 30, 40))