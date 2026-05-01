import tensorflow as tf
from src.models.losses import dice_loss, focal_tversky_loss

def test_losses_numerics():
    y_true = tf.zeros((2,8,16,1)); y_pred = tf.zeros((2,8,16,1))
    assert float(dice_loss()(y_true, y_pred)) >= 0.0
    assert float(focal_tversky_loss()(y_true, y_pred)) >= 0.0 