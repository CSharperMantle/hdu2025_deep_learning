# Generator.
#
# Copyright (c) 2023 Cliff Njoroge.
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# SPDX-License-Identifier: Apache-2.0

import torch as t
from torch import nn

from .bar_generator import BarGenerator
from .temporal import TemporalNetwork


class MuseGenerator(nn.Module):
    """Muse generator.

    Parameters
    ----------
    z_dimension: int, (default=32)
        Noise space dimension.
    hid_channels: int, (default=1024)
        Number of hidden channels.
    hid_features: int, (default=1024)
        Number of hidden features.
    out_channels: int, (default=1)
        Number of output channels.
    """

    def __init__(
        self,
        z_dimension: int = 32,
        hid_channels: int = 1024,
        hid_features: int = 1024,
        out_channels: int = 1,
        n_tracks: int = 4,
        n_bars: int = 2,
        n_steps_per_bar: int = 16,
        n_pitches: int = 84,
    ) -> None:
        super().__init__()
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches
        # chords generator
        self.chords_network = TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars)
        # melody generators
        self.melody_networks = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.melody_networks.add_module(
                "melodygen_" + str(n),
                TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars),
            )
        # bar generators
        self.bar_generators = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.bar_generators.add_module(
                "bargen_" + str(n),
                BarGenerator(
                    z_dimension,
                    hid_features,
                    hid_channels // 2,
                    out_channels,
                    n_steps_per_bar=n_steps_per_bar,
                    n_pitches=n_pitches,
                ),
            )
        # musegan generator compiled

    def forward(
        self, chords: t.Tensor, style: t.Tensor, melody: t.Tensor, groove: t.Tensor
    ) -> t.Tensor:
        """Perform forward.

        Parameters
        ----------
        chords: t.Tensor
            Chords.
        style: t.Tensor
            Style.
        melody: t.Tensor
            Melody.
        groove: t.Tensor
            Groove.

        Returns
        -------
        t.Tensor:
            Preprocessed input batch.
        """

        # Very important note
        # chords shape: (batch_size, z_dimension)
        # style shape: (batch_size, z_dimension)
        # melody shape: (batch_size, n_tracks, z_dimension)
        # groove shape: (batch_size, n_tracks, z_dimension)
        chord_outs = self.chords_network(chords)
        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []
            chord_out = chord_outs[:, :, bar]
            style_out = style
            for track in range(self.n_tracks):
                melody_in = melody[:, track, :]
                melody_out = self.melody_networks["melodygen_" + str(track)](melody_in)[
                    :, :, bar
                ]
                groove_out = groove[:, track, :]
                z = t.cat([chord_out, style_out, melody_out, groove_out], dim=1)
                track_outs.append(self.bar_generators["bargen_" + str(track)](z))
            track_out = t.cat(track_outs, dim=1)
            bar_outs.append(track_out)
        out = t.cat(bar_outs, dim=2)
        # Very important note
        # The tensor output shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        return out
