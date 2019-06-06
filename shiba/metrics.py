import numpy as np


def accuracy(prediction, target):
    """
    N-dimensional accuracy.
    Args:
      prediction:
      target:
    """
    correct = (prediction == target).sum().item()
    total = np.prod(target.shape)
    return correct / total


def categorical_accuracy(output, target):
    prediction = output.argmax(dim=1)
    return accuracy(prediction, target)


def binary_accuracy(output, target, threshold=0.5):
    prediction = (output.sigmoid() > threshold).float()
    return accuracy(prediction, target)


def threshold_f1(output, target, threshold=0.5):
    prediction = (output.sigmoid() > threshold).float()
    return f1(prediction, target)


def f1(prediction, target):
    """
    prediction: thresholded prediction (zeros or ones)
    """
    EPS = 1e-10
    intersection = (prediction * target).sum().float()
    union = (prediction.sum() + target.sum()).float()
    return (2 * (intersection + EPS) / (union + 2 * EPS)).item()
