# Utils.
#
# Copyright (c) 2023 Cliff Njoroge.
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# SPDX-License-Identifier: Apache-2.0

import torch as t
from torch import nn


def initialize_weights(layer: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Initialize module with normal distribution.

    Parameters
    ----------
    layer: nn.Module
        Layer.
    mean: float, (default=0.0)
        Mean value.
    std: float, (default=0.02)
        Standard deviation value.
    """
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, mean, std)
        nn.init.constant_(layer.bias, 0)


class Reshape(nn.Module):
    """Reshape layer.

    Parameters
    ----------
    shape: List[int]
        Dimensions after number of batches.
    """

    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Perform forward.

        Parameters
        ----------
        x: Tensor
            Input batch.

        Returns
        -------
        Tensor:
            Preprocessed input batch.
        """
        return x.view(x.size(0), *self.shape)
