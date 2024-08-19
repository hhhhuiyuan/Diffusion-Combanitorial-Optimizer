import numpy as np
import torch
from torchmetrics import Metric

def flp_decode(predictions, num_facility):
    """Decode the labels to solution of FLP."""
    sorted_predict_labels = np.argsort(- predictions)
    selected_fac = sorted_predict_labels[:num_facility]

    return selected_fac

def flp_evaluate(points, solution):
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    nearest_distances = np.min(distances[:, solution], axis=1)
    solu_obj = np.sum(nearest_distances)

    return torch.tensor(solu_obj)
    
