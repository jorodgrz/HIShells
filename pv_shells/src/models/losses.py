# src/models/losses.py
from __future__ import annotations
import tensorflow as tf

def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-4, 1.0-1e-4)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1-y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1-y_pred))
    return (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    def _fn(y_true, y_pred):
        t = tversky(y_true, y_pred, alpha=alpha, beta=beta, smooth=smooth)
        return tf.pow((1.0 - t), gamma)
    return _fn

def dice_loss(smooth=1e-6):
    def _fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-4, 1.0-1e-4)
        inter = tf.reduce_sum(y_true*y_pred)
        denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2*inter + smooth) / (denom + smooth)
        return 1.0 - dice
    return _fn