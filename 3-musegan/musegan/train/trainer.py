# Trainer.
#
# Copyright (c) 2023 Cliff Njoroge.
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# SPDX-License-Identifier: Apache-2.0

import os
import typing as ty

import torch as t
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm_std
from tqdm.notebook import tqdm as tqdm_notebook

from ..model import MuseCritic, MuseGenerator
from .criterion import GradientPenalty, WassersteinLoss


class TrainerCkpt(ty.TypedDict):
    epoch: int
    optimizer: dict
    state_dict: dict


class Trainer:
    """Trainer."""

    def __init__(
        self,
        generator: MuseGenerator,
        critic: MuseCritic,
        g_optimizer: t.optim.Optimizer,  # generator
        c_optimizer: t.optim.Optimizer,  # discriminator
        ckpt_path: str,  # checkpoint path
        device: str = "cuda:0",  # torch device
    ) -> None:
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.ckpt_path = ckpt_path
        self.device = device

        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)

        # Pre-allocate tensors for alpha to avoid repeated allocation
        self.alpha = None

    def save_ckpt(
        self,
        state: TrainerCkpt,
        checkpoint_path: str,
    ) -> None:
        """
        state: type dict
        checkpoint_path: path to save checkpoint
        """
        t.save(state, checkpoint_path)

    def load_ckpt(
        self,
        checkpoint_fpath: str,
        model: nn.Module,
        optimizer: t.optim.Optimizer,
    ) -> tuple[nn.Module, t.optim.Optimizer, int]:
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into
        optimizer: optimizer we defined in previous training
        """
        # load check point
        checkpoint: TrainerCkpt = t.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint["state_dict"])
        # initialize optimizer from checkpoint to optimizer
        optimizer.load_state_dict(checkpoint["optimizer"])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        # return model, optimizer, epoch value, min validation loss
        return model, optimizer, checkpoint["epoch"]

    def train(
        self,
        dataloader: DataLoader,
        start_epoch: int = 0,
        epochs: int = 500,
        batch_size: int = 64,
        repeat: int = 5,
        melody_groove: int = 4,
        save_checkpoint: bool = True,
        model_name: str = "musegan",
        tqdm: ty.Union[type[tqdm_std], type[tqdm_notebook]] = tqdm_notebook,
    ) -> None:
        os.makedirs(self.ckpt_path, exist_ok=True)

        # Pre-allocate tensors for alpha if not already done
        if self.alpha is None:
            self.alpha = t.rand(
                (batch_size, 1, 1, 1, 1), requires_grad=True, device=self.device
            )
        noise_tensors = {
            "cords": t.empty(batch_size, 32, device=self.device),
            "style": t.empty(batch_size, 32, device=self.device),
            "melody": t.empty(batch_size, melody_groove, 32, device=self.device),
            "groove": t.empty(batch_size, melody_groove, 32, device=self.device),
        }
        ones = t.ones(batch_size, 1, device=self.device)
        neg_ones = -t.ones(batch_size, 1, device=self.device)

        self.data = {
            "gloss": [],
            "closs": [],
            "cfloss": [],
            "crloss": [],
            "cploss": [],
        }

        for epoch in range(start_epoch, epochs):
            e_gloss = 0
            e_cfloss = 0
            e_crloss = 0
            e_cploss = 0
            e_closs = 0

            with tqdm(dataloader, unit="it") as train_loader:
                for real in train_loader:
                    real = real.to(self.device, non_blocking=True)

                    b_closs = 0
                    b_cfloss = 0
                    b_crloss = 0
                    b_cploss = 0

                    # Generate noise once and reuse for all critic updates
                    t.randn(batch_size, 32, out=noise_tensors["cords"])
                    t.randn(batch_size, 32, out=noise_tensors["style"])
                    t.randn(batch_size, melody_groove, 32, out=noise_tensors["melody"])
                    t.randn(batch_size, melody_groove, 32, out=noise_tensors["groove"])

                    for _ in range(repeat):
                        self.c_optimizer.zero_grad(set_to_none=True)

                        with t.no_grad():
                            fake = self.generator(
                                noise_tensors["cords"],
                                noise_tensors["style"],
                                noise_tensors["melody"],
                                noise_tensors["groove"],
                            )

                        # mix `real` and `fake` melody
                        realfake = self.alpha * real + (1.0 - self.alpha) * fake
                        fake_pred = self.critic(fake)
                        real_pred = self.critic(real)
                        realfake_pred = self.critic(realfake)
                        fake_loss = self.c_criterion(fake_pred, neg_ones)
                        real_loss = self.c_criterion(real_pred, ones)
                        penalty = self.c_penalty(realfake, realfake_pred)
                        closs = fake_loss + real_loss + 10 * penalty

                        closs.backward(retain_graph=True)
                        self.c_optimizer.step()

                        b_cfloss += fake_loss.item() / repeat
                        b_crloss += real_loss.item() / repeat
                        b_cploss += 10 * penalty.item() / repeat
                        b_closs += closs.item() / repeat

                    e_cfloss += b_cfloss / len(train_loader)
                    e_crloss += b_crloss / len(train_loader)
                    e_cploss += b_cploss / len(train_loader)
                    e_closs += b_closs / len(train_loader)

                    self.g_optimizer.zero_grad(set_to_none=True)
                    fake = self.generator(
                        noise_tensors["cords"],
                        noise_tensors["style"],
                        noise_tensors["melody"],
                        noise_tensors["groove"],
                    )
                    fake_pred = self.critic(fake)
                    b_gloss = self.g_criterion(fake_pred, ones)
                    b_gloss.backward()
                    self.g_optimizer.step()

                    e_gloss += b_gloss.item() / len(train_loader)
                    train_loader.set_postfix(
                        epoch=epoch,
                        generator_loss=f"{e_gloss:.3f}",
                        critic_loss=f"{e_closs:.3f}",
                        fake_loss=f"{e_cfloss:.3f}",
                        real_loss=f"{e_crloss:.3f}",
                        penalty=f"{e_cploss:.3f}",
                    )

            # Append Losses
            self.data["gloss"].append(e_gloss)
            self.data["closs"].append(e_closs)
            self.data["cfloss"].append(e_cfloss)
            self.data["crloss"].append(e_crloss)
            self.data["cploss"].append(e_cploss)

            if save_checkpoint:
                g_ckpt: TrainerCkpt = {
                    "epoch": epoch + 1,
                    "state_dict": self.generator.state_dict(),
                    "optimizer": self.g_optimizer.state_dict(),
                }
                self.save_ckpt(
                    g_ckpt,
                    os.path.join(self.ckpt_path, f"{model_name}_Net_G-{epoch}.pth"),
                )
                c_ckpt: TrainerCkpt = {
                    "epoch": epoch + 1,
                    "state_dict": self.critic.state_dict(),
                    "optimizer": self.c_optimizer.state_dict(),
                }
                self.save_ckpt(
                    c_ckpt,
                    os.path.join(self.ckpt_path, f"{model_name}_Net_D-{epoch}.pth"),
                )
