import keras.backend as K
from keras.losses import binary_crossentropy

# Class weight for foreground firing cells (corresponding to y_true=1)
# relative to the background (y_true=0) to compensate for the fact
# that there are more background pixels than foreground.
WEIGHT = 5.0

def weighted_bce(y_true, y_pred):
    losses = K.binary_crossentropy(y_true, y_pred) # returns a tensor
    weights = y_true * (WEIGHT - 1) + 1 # WEIGHT for y_true=1, 1 for y_true=0
    loss = K.mean(weights * losses) # scale the losses before taking the mean
    return loss

def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = K.mean(y_true * y_pred)
    score = (2 * intersection + smooth) / (K.mean(y_true + y_pred) + smooth)
    return 1.0 - score

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def iou_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    #union = K.sum(1 - (1 - y_true) * (1 - y_pred))
    union = K.sum(y_true + y_pred) - intersection
    loss = 1 - intersection / union
    return loss
