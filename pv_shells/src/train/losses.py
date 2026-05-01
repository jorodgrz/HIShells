# src/train/losses.py
import tensorflow as tf
from tensorflow import keras

def make_loss_and_metrics(cfg):
    """
    Construct loss and metrics for training.
    Uses binary crossentropy (for mask segmentation),
    with optional weighting and PR AUC / precision / recall.
    """
    # Binary crossentropy loss
    loss = keras.losses.BinaryCrossentropy(from_logits=False)

    # Metrics
    metrics = [
        keras.metrics.AUC(curve="PR", name="pr_auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    return loss, metrics