import torch
import torch.nn as nn
from diffusiongym import D

from .uncertainty_estimator import UncertaintyEstimator


class EnsembleUncertaintyEstimator(UncertaintyEstimator[D]):
    """Uncertainty estimator based on an ensemble of MLPs.

    This is the same setup as in ALDE (https://www.nature.com/articles/s41467-025-55987-8).
    """

    def _init_estimator(self):
        self._construct_models()

    def _construct_models(self):
        # 5 models in the ensemble
        self.models = nn.ModuleList([
            MLP(self.feat_dim, 1, hid_dim=100, p_dropout=0.1).to(self.device)
            for _ in range(5)
        ])

    def _update_estimator(self, feats: torch.Tensor, labels: torch.Tensor):
        # Re-initialize the models and train from scratch (very cheap so its fine)
        self._construct_models()

        criterion = nn.MSELoss()

        # 90% bootstrap fraction and 30 early stopping window size
        n = feats.shape[0]
        k = int(0.9 * n)
        w = 30
        num_steps = 1000

        for model in self.models:
            idx = torch.randperm(n)[:k]
            x = feats[idx].to(self.device)
            y = labels[idx].to(self.device)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3) 

            losses = torch.zeros(num_steps)

            for i in range(num_steps):
                model.train()
                opt.zero_grad()
                pred = model(x)
                loss = criterion(pred, y.unsqueeze(-1))
                loss.backward()
                opt.step()

                losses[i] = loss.item()

                if i > w:
                    recent_min = losses[i-w+1:i+1].min() 
                    overall_min = losses[:i-w+1].min()
                    if overall_min <= recent_min:
                        break

            model.eval()

    def _mean_and_uncertainty(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(feats) for m in self.models], dim=1)
        std, mean = torch.std_mean(preds, dim=1, correction=0)
        return mean, std


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: int = 100,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
