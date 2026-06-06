import torch
import torch.nn as nn


class HyperbolicTripletLoss(nn.Module):
    """Poincaré-ball contrastive triplet loss for scale-free knowledge graphs."""
    # ---------------------------------------------------------------------------
    # Hyperbolic Triplet Loss  (Poincaré ball, curvature c)
    #
    # Distance formula (Ganea et al. 2018):
    #   d_c(u,v) = (1/√c) · arccosh( 1 + 2c‖u−v‖² / ((1−c‖u‖²)(1−c‖v‖²)) )
    #
    # Fixes applied vs. original:
    #   1. project(): clamp norm to ≥ 1e-8 before division to avoid NaN when
    #      norm=0 (torch.where evaluates both branches eagerly).
    #   2. poincare_dist(): clamp squnorm/sqvnorm to keep denominators strictly
    #      positive even if projection is slightly imperfect at fp16 boundaries.
    #   3. poincare_dist(): clamp denom to ≥ 1e-8 to guard against zero division.
    #   4. poincare_dist(): clamp arccosh argument x to ≥ 1+1e-7 so that
    #      √(x²−1) is always real and log(x+z) is always finite.
    # ---------------------------------------------------------------------------

    def __init__(self, model, margin: float = 0.3, c: float = 0.1):
        super().__init__()
        self.model = model
        self.margin = margin
        self.c = c
        # Keeping √c as a buffer so it moves with the module to the right device
        self.register_buffer('sqrt_c', torch.tensor(float(c)).sqrt())

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project x onto the open Poincaré ball of radius 1/√c."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = (1.0 / self.sqrt_c) - 1e-5
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    def poincare_dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Poincaré distance between two batches of vectors.

        d_c(u,v) = (1/√c) · arccosh(x)
        where x = 1 + 2c‖u−v‖² / ((1−c‖u‖²)(1−c‖v‖²))
        and   arccosh(x) = log(x + √(x²−1))
        """
        u = self.project(u)
        v = self.project(v)

        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1).clamp(max=1.0 / self.c - 1e-5)
        sqvnorm = torch.sum(v ** 2, dim=-1).clamp(max=1.0 / self.c - 1e-5)

        denom = (1.0 - self.c * squnorm) * (1.0 - self.c * sqvnorm)
        denom = denom.clamp(min=1e-8)

        x = 1.0 + 2.0 * self.c * sqdist / denom
        x = x.clamp(min=1.0 + 1e-7)

        z = torch.sqrt(x ** 2 - 1.0)
        return torch.log(x + z) / self.sqrt_c

    def forward(self, sentence_features, labels):
        reps = [
            self.model(sf)['sentence_embedding']
            for sf in sentence_features
        ]
        anchor, positive, negative = reps

        dist_pos = self.poincare_dist(anchor, positive)
        dist_neg = self.poincare_dist(anchor, negative)

        loss = torch.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
