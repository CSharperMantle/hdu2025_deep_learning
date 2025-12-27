# Load and save an array to a processed .npz file.
#
# Rewritten from clifford's implementation, and without Linux-only SharedArray.
#
# Copyright (c) 2023 Cliff Njoroge.
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# Link: https://github.com/cliffordkleinsr/musegan-pytorch/blob/41dc616a00444835aa6b94946577c8ee1e7b5f30/prepare_data.py
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import typing as ty

import numpy as np


def parse_arguments() -> tuple[ty.Optional[str], ...]:
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to the data file.")
    parser.add_argument(
        "--out-dir", default="prepared", help="Output directory."
    )
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--name",
        help="File name to save. Defaults to the original file name.",
    )
    name_group.add_argument(
        "--prefix",
        help="Prefix to the file name to save. Only effective when --name is not given.",
    )
    parser.add_argument(
        "--dtype", default="bool", help="Datatype of the array. Defaults to bool."
    )
    args = parser.parse_args()
    return args.filepath, args.out_dir, args.name, args.prefix, args.dtype


def main():
    filepath, out_dir, name, prefix, dtype = parse_arguments()

    assert filepath is not None
    assert out_dir is not None

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + "_" + name

    print(f"Loading data from '{filepath}'.")

    if filepath.endswith(".npy"):
        data = np.load(filepath)
        # Assert shape
        if len(data.shape) == 6:
            RESHAPE_PARAMS = {
                "num_bar": 4,
                "num_beat": 4,
                "num_pitch": 84,
                "num_track": 8,
                "num_timestep": 96,
                "beat_resolution": 24,
                "lowest_pitch": 24,
            }
            data = data.reshape(
                -1,
                RESHAPE_PARAMS["num_bar"],
                RESHAPE_PARAMS["num_timestep"],
                RESHAPE_PARAMS["num_pitch"],
                RESHAPE_PARAMS["num_track"],
            )
        data = data.astype(dtype)
    else:
        # Handle sparse .npz format
        with np.load(filepath) as loaded:
            shape = tuple(loaded["shape"])
            data = np.zeros(shape, dtype=dtype)
            data[tuple(loaded["nonzero"])] = 1

    print(f"Original shape: (name='{name}', shape={data.shape}, dtype={data.dtype})")

    # Given you have a Piano-roll Dataset with shape :
    #     num of phrases:102378,
    #     num of bar:4,
    #     time resolution:48,
    #     pitch range:84,
    #     num of tracks:5
    #     summary: (102378, 4, 48, 84, 5)
    #
    # You will need to transpose it to match the Generator/Critic random noise with tensor dimensions:
    #     (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches).
    #     i.e. (102378, 5, 4, 48, 84)
    #
    # Therefore: (102378, 4, 48, 84, 5) -> (102378, 5, 4, 48, 84)
    data = data.transpose(0, 4, 1, 2, 3)

    print(f"Transposed shape: (name='{name}', shape={data.shape}, dtype={data.dtype})")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, name) + ".npz", data)


if __name__ == "__main__":
    main()
