"""Shared feature extraction utilities."""
import torch
import numpy as np


def extract_penultimate_features(model, data, batch_size=128):
    """Extract penultimate-layer features in batches."""
    model.eval()
    device = next(model.parameters()).device
    feats = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device).float()
            f = model.forward(batch, return_penultimate=True)
            feats.append(f.detach().cpu().numpy().reshape(f.shape[0], -1))
    return np.vstack(feats) if feats else np.zeros((0, 1))
