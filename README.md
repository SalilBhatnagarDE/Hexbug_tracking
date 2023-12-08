# About the Project
Computer vision is one of the major tasks and applications of artificial intelligence (AI). Gaining hands-on experience is therefore of great importance for future AI developers. 

In the Tracking Olympiad Project : Summer Semester 2023 at FAU Erlangen, Germany, students use the latest object detection and tracking algorithms to track a freely, randomly moving object ("HexBug") in a given arena. 

A set of videos are provided that contain the ground-truth positional information and we will implement an own tracking technique. 

[![See Hexbugs](1.jpg)](https://youtube.com/shorts/V4Rl51bUAsw?feature=share)

# Problem Statement
### Build an AI agent to track hexbugs
### 100 videos and annotations were provided as starting point
### 100 videos and annotations were provided as starting point
### It should generalize well even in very complex environments
[![Complex Enviornments](2.jpg)]
### Calculations of Scores 
[![Scores](3.jpg)]

# Methodology
### Used Detection Transformer (DETR) with different backbones
#### Used regression head to regress the x,y coordinates of head of hexbugs
#### ResNet 152 Dilated Convolutional backbone performs best

### Data Preprocessing includes : Normalization, Data Augmentation (Color Jitter, Gaussian blur filter, Random Rotation, Random Invert)

### Training
#### Hyperparameter tuning

- **4 Nvidia A100 GPU (40 GB each)** on Alex HPC cluster (total batch size kept as `56`)
- 
- **Coefficients for loss function**: 
  Total loss = `10 * bbox_loss + 2 * ce_loss + 2 * loss_giou + 10 * reg_loss`

- **Use of auxiliary cross entropy losses in each of the decoder layer**: Set to `True`

- **Coefficients for cost in Hungarian algorithm assignment**

- **Coefficient in the CE loss for class imbalance** due to more queries in the decoder

- **Learning rate** for backbone `0.00001`, and Lr for transformer heads `0.0001`

- **Early stopping, model checkpoint saving and tensorboard logging**








