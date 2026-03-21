"""
Lee et al. (2018) "A Simple Unified Framework for Detecting
Out-of-Distribution Examples and Adversarial Examples" (NeurIPS 2018).

Multi-layer Mahalanobis distance detector -- the standard baseline
for adversarial/OOD detection papers.

Key differences from our single-representation Mahalanobis:
  - Uses features from ALL layers (with learned per-layer weights)
  - Adds input preprocessing (small adversarial perturbation to enhance features)
  - Trains a logistic regression on per-layer Mahalanobis scores
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import TruncatedSVD


class MultiLayerMahalanobisDetector:
    """Multi-layer Mahalanobis distance detector (Lee et al., 2018).

    Extracts intermediate features from every activation layer in a model,
    computes per-class Mahalanobis distances at each layer, then combines
    the per-layer scores via logistic regression.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier (must be in eval mode).
    num_classes : int
        Number of output classes.
    device : str or torch.device
        Device for inference (default ``'cpu'``).
    max_components : int
        Maximum feature dimension per layer; layers with higher
        dimensionality are reduced via TruncatedSVD (default 256).
    batch_size : int
        Batch size used during feature extraction to avoid OOM
        (default 64).
    """

    def __init__(self, model, num_classes, device='cpu',
                 max_components=256, batch_size=64):
        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.max_components = max_components
        self.batch_size = batch_size

        # Per-layer statistics (populated by ``fit``)
        self.class_means = {}   # layer_name -> {class_idx -> mean_vector}
        self.precision = {}     # layer_name -> precision matrix (tied)
        self.svd = {}           # layer_name -> TruncatedSVD or None
        self.layer_names = []   # ordered list of hooked layer names

        # Logistic regression combiner (populated by ``fit_logistic``)
        self.logreg = None

        # Forward-hook bookkeeping
        self._features = {}
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Attach forward hooks to all activation layers in the model."""
        activation_types = (
            nn.ReLU, nn.ELU, nn.Tanh,
            nn.LeakyReLU, nn.Sigmoid, nn.GELU,
        )
        for name, module in self.model.named_modules():
            if isinstance(module, activation_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, inp, output):
            self._features[name] = output.detach().cpu()
        return hook_fn

    def cleanup(self):
        """Remove all forward hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x):
        """Run a forward pass and return per-layer flattened features.

        Parameters
        ----------
        x : torch.Tensor
            Input batch ``(N, C, H, W)``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from layer name to array of shape ``(N, D_layer)``.
        """
        self._features = {}
        with torch.no_grad():
            _ = self.model(x.to(self.device).float())
        result = {}
        for name in sorted(self._features.keys()):
            feat = self._features[name]
            result[name] = feat.reshape(feat.shape[0], -1).numpy()
        return result

    def _extract_features_batched(self, data):
        """Extract features in batches and concatenate across samples.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Full dataset ``(N, C, H, W)``.

        Returns
        -------
        dict[str, np.ndarray]
            Per-layer arrays of shape ``(N, D_layer)``.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        all_features = {}  # layer_name -> list of arrays
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_features = self._extract_features(batch)
            for name, feat in batch_features.items():
                all_features.setdefault(name, []).append(feat)

        return {
            name: np.vstack(parts) for name, parts in all_features.items()
        }

    # ------------------------------------------------------------------
    # Fitting (training phase)
    # ------------------------------------------------------------------

    def fit(self, train_loader_or_data, labels):
        """Compute per-layer, per-class Mahalanobis statistics.

        Parameters
        ----------
        train_loader_or_data : torch.Tensor, np.ndarray, or DataLoader
            Training inputs.  If a ``DataLoader`` is passed the *labels*
            argument is ignored and labels are read from the loader.
        labels : np.ndarray or None
            Integer class labels aligned with the data tensor.
        """
        self.model.eval()

        # ----- Extract features from training data -----
        if isinstance(train_loader_or_data, torch.utils.data.DataLoader):
            all_features = {}
            all_labels = []
            for batch_x, batch_y in train_loader_or_data:
                feats = self._extract_features(batch_x)
                for name, feat in feats.items():
                    all_features.setdefault(name, []).append(feat)
                all_labels.append(batch_y.numpy())
            features_dict = {
                n: np.vstack(parts) for n, parts in all_features.items()
            }
            labels = np.concatenate(all_labels).astype(int)
        else:
            data = train_loader_or_data
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            features_dict = self._extract_features_batched(data)
            labels = np.asarray(labels, dtype=int)

        self.layer_names = sorted(features_dict.keys())

        # ----- Per-layer statistics -----
        for layer_name in self.layer_names:
            raw = features_dict[layer_name]         # (N, D)
            N, D = raw.shape

            # Dimensionality reduction if needed
            n_comp = min(self.max_components, D, max(1, N - 1))
            if D > n_comp:
                svd = TruncatedSVD(n_components=n_comp, random_state=0)
                proj = svd.fit_transform(raw)
                self.svd[layer_name] = svd
            else:
                proj = raw
                self.svd[layer_name] = None

            # Tied (shared) covariance across all classes
            try:
                lw = LedoitWolf().fit(proj)
                precision = lw.precision_
            except Exception:
                precision = np.eye(proj.shape[1])

            self.precision[layer_name] = precision

            # Per-class means
            self.class_means[layer_name] = {}
            global_mean = np.mean(proj, axis=0)
            for c in range(self.num_classes):
                mask = (labels == c)
                class_data = proj[mask]
                if class_data.shape[0] >= 1:
                    self.class_means[layer_name][c] = np.mean(class_data, axis=0)
                else:
                    self.class_means[layer_name][c] = global_mean

    # ------------------------------------------------------------------
    # Per-layer Mahalanobis scores
    # ------------------------------------------------------------------

    def _mahalanobis_scores_per_layer(self, features_dict):
        """Compute min-over-classes Mahalanobis distance at each layer.

        Parameters
        ----------
        features_dict : dict[str, np.ndarray]
            Per-layer features for N samples.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_layers)`` -- one score per layer per sample.
        """
        N = None
        per_layer_scores = []

        for layer_name in self.layer_names:
            raw = features_dict[layer_name]
            if N is None:
                N = raw.shape[0]

            # Project if SVD was fitted
            svd = self.svd[layer_name]
            if svd is not None:
                proj = svd.transform(raw.reshape(raw.shape[0], -1))
            else:
                proj = raw.reshape(raw.shape[0], -1)

            prec = self.precision[layer_name]

            # Min Mahalanobis distance over classes
            min_dists = np.full(proj.shape[0], np.inf)
            for c in range(self.num_classes):
                if c not in self.class_means[layer_name]:
                    continue
                diff = proj - self.class_means[layer_name][c]
                dists_sq = np.einsum('ij,jk,ik->i', diff, prec, diff)
                dists = np.sqrt(np.maximum(dists_sq, 0.0))
                min_dists = np.minimum(min_dists, dists)

            per_layer_scores.append(min_dists)

        return np.column_stack(per_layer_scores)  # (N, num_layers)

    # ------------------------------------------------------------------
    # Logistic regression combiner
    # ------------------------------------------------------------------

    def fit_logistic(self, clean_scores, adv_scores):
        """Train logistic regression to combine per-layer scores.

        Parameters
        ----------
        clean_scores : np.ndarray
            Per-layer scores for clean examples, shape ``(N_clean, L)``.
        adv_scores : np.ndarray
            Per-layer scores for adversarial examples, shape ``(N_adv, L)``.
        """
        X = np.vstack([clean_scores, adv_scores])
        y = np.concatenate([
            np.zeros(len(clean_scores)),
            np.ones(len(adv_scores)),
        ])
        self.logreg = LogisticRegressionCV(
            cv=3, max_iter=1000, random_state=0,
        )
        self.logreg.fit(X, y)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, data):
        """Compute anomaly scores for a batch of inputs.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input data ``(N, C, H, W)``.

        Returns
        -------
        np.ndarray
            1-D array of length N (higher = more anomalous).
        """
        self.model.eval()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        features_dict = self._extract_features_batched(data)
        layer_scores = self._mahalanobis_scores_per_layer(features_dict)

        if self.logreg is not None:
            # Logistic regression probability of being adversarial
            return self.logreg.predict_proba(layer_scores)[:, 1]
        else:
            # Fallback: simple average across layers
            return layer_scores.mean(axis=1)

    def score_per_layer(self, data):
        """Return raw per-layer scores (useful for ``fit_logistic``).

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input data ``(N, C, H, W)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_layers)``.
        """
        self.model.eval()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        features_dict = self._extract_features_batched(data)
        return self._mahalanobis_scores_per_layer(features_dict)
