import argparse
import os
import random

import musegan
import numpy as np
import torch as t
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

N_BARS = 4
N_TRACKS = 5
N_STEPS_PER_BAR = 48
BATCH_SIZE = 16
Z_DIM = 32
HID_FEATURES = 1152
HID_CHANNELS = 192
N_PITCHES = 84
DEVICE = "cuda:0"
DATASET_PATH = "prepared/train_x_lpd_5_phr.npz"
DATASET_REDUCE_FACTOR = 0.3
CKPT_PATH = "ckpt/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MuseGAN model with optional checkpoint resuming"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint",
    )
    parser.add_argument(
        "-e",
        "--from-epoch",
        type=int,
        help="Epoch number to resume from (required if --resume is set). Uses default checkpoint naming pattern.",
    )
    parser.add_argument(
        "-g",
        "--gen-ckpt",
        type=str,
        help="Path to generator checkpoint file (optional, overrides default pattern)",
    )
    parser.add_argument(
        "-d",
        "--discrim-ckpt",
        type=str,
        help="Path to discriminator/critic checkpoint file (optional, overrides default pattern)",
    )
    args = parser.parse_args()

    if (
        args.resume
        and args.from_epoch is None
        and (args.gen_ckpt is None or args.discrim_ckpt is None)
    ):
        parser.error(
            "--resume requires either --from-epoch or both --gen-ckpt and --discrim-ckpt"
        )
    if args.from_epoch is not None and not args.resume:
        parser.error("--from-epoch requires --resume flag")
    if (args.gen_ckpt is not None or args.discrim_ckpt is not None) and not args.resume:
        parser.error("--gen-ckpt and --discrim-ckpt require --resume flag")

    return args


args = parse_args()

t.random.manual_seed(0x0D000721)
random.seed(0x0D000721)
np.random.seed(0x0D000721)
g = t.Generator()
g.manual_seed(0x0D000721)

bar_gen = musegan.bar_generator.BarGenerator(
    z_dimension=Z_DIM,
    hid_features=HID_FEATURES,
    hid_channels=HID_CHANNELS,
    n_steps_per_bar=N_STEPS_PER_BAR,
    n_pitches=N_PITCHES,
)

muse_gen = musegan.generator.MuseGenerator(
    z_dimension=Z_DIM,
    hid_channels=HID_CHANNELS * 2,
    hid_features=HID_FEATURES,
    n_tracks=N_TRACKS,
    n_bars=N_BARS,
    n_steps_per_bar=N_STEPS_PER_BAR,
    n_pitches=N_PITCHES,
)

critic = musegan.critic.MuseCritic(
    hid_channels=128,
    n_tracks=N_TRACKS,
    n_bars=N_BARS,
    n_steps_per_bar=N_STEPS_PER_BAR,
    n_pitches=N_PITCHES,
)


def seed_worker(_):
    worker_seed = t.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dataset = musegan.dataset.LPDDataset(DATASET_PATH)
dataset_len = int(len(dataset) * DATASET_REDUCE_FACTOR)
dataset, _ = random_split(
    dataset,
    lengths=(dataset_len, len(dataset) - dataset_len),
    generator=g,
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
)

muse_gen = muse_gen.to(DEVICE)
g_optimizer = t.optim.Adam(muse_gen.parameters(), lr=0.0001, betas=(0.2, 0.9))
muse_gen = muse_gen.apply(musegan.utils.initialize_weights)

critic = critic.to(DEVICE)
c_optimizer = t.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.2, 0.9))
critic = critic.apply(musegan.utils.initialize_weights)
trainer = musegan.train.Trainer(
    muse_gen, critic, g_optimizer, c_optimizer, CKPT_PATH, DEVICE
)

start_epoch = 0
if args.resume:
    print(f"Resuming training from epoch {args.from_epoch}")
    gen_ckpt_path = (
        args.gen_ckpt
        if args.gen_ckpt
        else os.path.join(CKPT_PATH, f"musegan_Net_G-{args.from_epoch}.pth")
    )
    discrim_ckpt_path = (
        args.discrim_ckpt
        if args.discrim_ckpt
        else os.path.join(CKPT_PATH, f"musegan_Net_D-{args.from_epoch}.pth")
    )

    muse_gen, g_optimizer, gen_epoch = trainer.load_ckpt(
        gen_ckpt_path, muse_gen, g_optimizer
    )
    critic, c_optimizer, discrim_epoch = trainer.load_ckpt(
        discrim_ckpt_path, critic, c_optimizer
    )

    if gen_epoch != discrim_epoch:
        print(
            f"Warning: Generator epoch ({gen_epoch}) and discriminator epoch ({discrim_epoch}) don't match"
        )

    start_epoch = gen_epoch
    print(f"Starting from epoch {start_epoch}")

trainer.train(
    loader,
    start_epoch=start_epoch,
    epochs=10,
    batch_size=BATCH_SIZE,
    melody_groove=N_TRACKS,
    tqdm=tqdm,
)
