"""Loading 4-channel OFDR captures.

A capture is a NumPy ``.npz`` file with one array per detector channel
(``ch1`` ... ``ch4``), sampled uniformly in time, plus optional metadata
(``step_us`` and, for synthetic data, the ground truth ``truth_*``).
"""

import numpy as np

CHANNELS = (1, 2, 3, 4)


def load_channels(path):
    """-> (channels, meta)

    channels: dict {1: array, 2: array, ...} with every channel present
    meta:     dict of all remaining scalar/array entries in the file
    """
    with np.load(path, allow_pickle=False) as f:
        channels = {n: np.asarray(f[f"ch{n}"], float)
                    for n in CHANNELS if f"ch{n}" in f}
        meta = {k: (f[k].item() if f[k].ndim == 0 else f[k])
                for k in f.files if not k.startswith("ch")}
    if not channels:
        raise ValueError(f"{path}: no channels ch1..ch4 found")
    return channels, meta
