# Loss function and gradient penalty for MuseGAN.
#
# Copyright (c) 2023 Cliff Njoroge.
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# SPDX-License-Identifier: Apache-2.0

import torch as t
from torch import nn


class WassersteinLoss(nn.Module):
    """Wasserstein loss."""

    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_pred: t.Tensor, y_target: t.Tensor) -> t.Tensor:
        """Calculate Wasserstein loss.

        Parameters
        ----------
        y_pred: Tensor
            Prediction.
        y_target: Tensor
            Target.

        Returns
        -------
        Tensor:
            Loss value.
        """
        loss = -t.mean(y_pred * y_target)
        return loss


class GradientPenalty(nn.Module):
    """Gradient penalty."""

    def __init__(self):
        super(GradientPenalty, self).__init__()

    def forward(self, inputs: t.Tensor, outputs: t.Tensor) -> t.Tensor:
        """Calculate gradient penalty.

        Parameters
        ----------
        inputs: Tensor
            Input from which to track gradient.
        outputs: Tensor
            Output to which to track gradient.

        Returns
        -------
        Tensor:
            Penalty value.
        """
        grad = t.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=t.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad = t.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        penalty = t.mean((1.0 - grad) ** 2)
        return penalty
