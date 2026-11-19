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

## Adapting to Your Face

FER2013's "sad" examples are mostly strong expressions, so subtle sadness tends to read as neutral.
Fine-tuning on a few hundred frames of your own face fixes this:

```bash
uv run python -m expression.capture     # h/s/n to record happy/sad/neutral, space to stop, q to quit
uv run python -m expression.finetune    # writes checkpoints/personal.pt only if it beats best.pt on held-out sessions
uv run python -m expression.predict --webcam   # uses personal.pt automatically
```

Record each expression in 3+ short sessions (vary head angle and lighting) using the natural expressions you want
recognised. Frames are saved to `data/personal/` (git-ignored).

## Project Structure

| File | Purpose |
|---|---|
| `src/expression/config.py` | Model configuration, dataset settings, classes and device selection |
| `src/expression/data.py` | Dataset filtering, label mapping and image transformations |
| `src/expression/model.py` | Model architecture and checkpoint loading |
| `src/expression/train.py` | Training pipeline and test-set evaluation |
| `src/expression/camera.py` | Face detection/cropping and webcam handling |
| `src/expression/capture.py` | Record labelled frames of your own face |
| `src/expression/finetune.py` | Adapt the model to your recorded frames |
| `src/expression/predict.py` | Face detection and expression prediction for images or webcam feeds |

## Technologies

**Python** · **PyTorch** · **OpenCV** · **timm** · **ConvNeXt-Tiny** · **Hugging Face Datasets**