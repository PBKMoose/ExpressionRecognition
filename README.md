# Expression Recognition

A deep learning application that recognises three facial expressions — **Happy**, **Sad**, and **Neutral** — from images or a live webcam feed.

## How It Works

The application uses a two-stage pipeline:

**1. Face Detection**
- Uses OpenCV's Haar Cascade classifier to detect faces.
- Crops each detected face for individual analysis.

**2. Expression Classification**
- Uses **ConvNeXt-Tiny**, pretrained on ImageNet-22k, as the backbone.
- Fine-tuned on the [FER2013 dataset](https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest).
- Trained on approximately **17,000 images**, filtered to the three target expressions.

## Model Training

Several techniques are used to improve generalisation and performance:

- **Data Augmentation:** Random cropping, horizontal flipping, rotation, brightness/contrast adjustments, and random erasing.
- **Label Smoothing:** Reduces overconfidence in predictions.
- **Class-Weighted Loss:** Addresses class imbalance in the dataset.
- **Optimisation:** AdamW optimiser with a One-Cycle learning rate scheduler.
- **Test-Time Augmentation:** Horizontal flipping during inference.

### Custom Facial Expression Data

The project also includes a custom data collection script (`capture.py`) that allows users to capture images of their own facial expressions for additional model training.

This was introduced to address weaker performance when recognising **sad expressions**, allowing additional examples to be collected to help improve classification accuracy for this class.

## Project Structure

| File | Purpose |
|---|---|
| `src/expression/config.py` | Model configuration, dataset settings, classes and device selection |
| `src/expression/data.py` | Dataset filtering, label mapping and image transformations |
| `src/expression/model.py` | Model architecture and checkpoint loading |
| `src/expression/train.py` | Training pipeline and test-set evaluation |
| `src/expression/predict.py` | Face detection and expression prediction for images or webcam feeds |
| `src/expression/capture.py` | Capture custom facial expression images for additional training and improving underperforming classes |

## Technologies

**Python** · **PyTorch** · **OpenCV** · **timm** · **ConvNeXt-Tiny** · **Hugging Face Datasets**