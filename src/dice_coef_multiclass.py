#
# dice_coef for multi-classe 
#
import tensorflow as tf
import tensorflow.keras.backend as K

# Please see:
# Multiclass segmentation for different loss functions(Dice loss, Focal loss, Total loss = (Summation of Dice and focal loss)) in Tensorflow
# https://medium.com/@mb16biswas/multiclass-segmentation-for-different-loss-functions-dice-loss-focal-loss-total-loss-summation-455178517cea

# https://gist.github.com/sohiniroych/68ce46adfae0400acc5fe833d96f6464#file-loss_functions-py

# https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-pytho

# https://www.kaggle.com/code/mb16biswas/multiclass-segmentation-for-diff-loss-functions


# Dice_coef for multi-class segmentation
def dice_coef_multiclass(y_true, y_pred, smooth=1):
    """
    Dice coefficient for multi-class segmentation.
    Args:
        y_true: Ground truth tensor (one-hot encoded). Shape: (batch, height, width, num_classes)
        y_pred: Prediction tensor (probabilities). Shape: (batch, height, width, num_classes)
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss_multiclass(y_true, y_pred, smooth=1):
    """
    Dice loss, which can be minimized during training.
    """
    return 1 - dice_coef_multiclass(y_true, y_pred, smooth)

# 2026/09/23 Added
"""
This metric calculates the Dice coefficient exclusively for the foreground, excluding the background.
This is a straightforward way to mitigate the class imbalance problem that occurs when background 
pixels occupy an overwhelmingly large portion of a mask image.
"""
def dice_coef_foreground(y_true, y_pred, epsilon=1e-6):
    """
    Args:
        y_true: Ground truth tensor (one-hot encoded). Shape: (batch, height, width, num_classes)
        y_pred: Prediction tensor (probabilities). Shape: (batch, height, width, num_classes)
    """
    y_true = tf.cast(y_true, tf.float32)
    # Get foregrounds by excludeing backgrounds (channel 0)
    y_true_foreground = y_true[..., 1:]
    y_pred_foreground = y_pred[..., 1:]
    
    intersection = K.sum(y_true_foreground * y_pred_foreground, axis=[1, 2])
    sum_ = K.sum(y_true_foreground + y_pred_foreground, axis=[1, 2])
    
    dice = (2. * intersection + epsilon) / (sum_ + epsilon)
    return dice


def dice_loss_foreground(y_true, y_pred, epsilon=1e-6):
    dice = dice_coef_foreground(y_true, y_pred, epsilon=1e-6)
    return 1.0 - K.mean(dice)

# 2026/09/24
# Experimantal metric : 
# This is an arithmetic average as shown below, not weighted average.
# dice_coef_hybrid =  (bg_fg_dice + fg_only_dice) /2.0
def dice_coef_hybrid(y_true, y_pred):
    bg_fg_dice   = dice_coef_multiclass(y_true, y_pred, smooth=1)
    fg_only_dice = dice_coef_foreground(y_true, y_pred, epsilon=1e-6)
    mean_dice    = (bg_fg_dice + fg_only_dice) / 2.0
    return mean_dice
