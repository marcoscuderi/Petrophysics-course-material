"""Plot every recorded sample from selected AE chunks and channels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS: edit only these values
# =============================================================================

# Folder containing ae_chunk0, ae_chunk1, ... . This default expects the data
# folder beside this script; replace it with an absolute Path if necessary.
DATA_FOLDER = Path(__file__).resolve().parent / "./data/UCS_sandstone/UCS_sandstone_rec2"

# Chunks and channels to plot. They do not need to be consecutive.
SELECTED_CHUNKS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
SELECTED_CHANNELS = [1]

SAMPLING_RATE_HZ = 3.125e6

# False plots the original voltage. True removes the mean of each chunk.
REMOVE_MEAN = False

# Keep this equal to 1 to plot every sample. If plotting is too slow, try 10.
PLOT_EVERY_NTH_SAMPLE = 1


# =============================================================================
# LOAD AND PLOT
# =============================================================================

if PLOT_EVERY_NTH_SAMPLE < 1:
    raise ValueError("PLOT_EVERY_NTH_SAMPLE must be at least 1.")

figure, axes = plt.subplots(
    len(SELECTED_CHANNELS),
    1,
    figsize=(13, 3.2 * len(SELECTED_CHANNELS)),
    sharex=True,
    constrained_layout=True,
)

# np.atleast_1d also works when only one channel is selected.
axes = np.atleast_1d(axes)

for axis, channel in zip(axes, SELECTED_CHANNELS):
    for chunk in SELECTED_CHUNKS:
        file_path = (
            DATA_FOLDER
            / f"ae_chunk{chunk}"
            / f"ae_chunk{chunk}_ch{channel}.bin"
        )

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found:\n{file_path}")

        # Although the extension is .bin, these files contain a NumPy header.
        # np.load reads the header and therefore obtains float16 automatically.
        waveform = np.load(file_path, mmap_mode="r", allow_pickle=False)

        if waveform.ndim != 1:
            raise ValueError(f"Expected a 1-D waveform in {file_path.name}.")

        voltage = np.asarray(waveform[::PLOT_EVERY_NTH_SAMPLE], dtype=np.float32)

        if REMOVE_MEAN:
            voltage = voltage - float(np.mean(waveform, dtype=np.float64))

        # Chunk N occupies the interval N to N+1 seconds. This leaves visible
        # gaps when non-consecutive chunks are selected.
        sample_numbers = np.arange(
            0,
            waveform.size,
            PLOT_EVERY_NTH_SAMPLE,
            dtype=np.float64,
        )
        time_s = chunk + sample_numbers / SAMPLING_RATE_HZ

        axis.plot(time_s, voltage, linewidth=0.45, label=f"chunk {chunk}")

        print(
            f"Loaded {file_path.name}: {waveform.size:,} samples, "
            f"dtype={waveform.dtype}"
        )

    axis.set_ylabel(f"Channel {channel} (V)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=min(4, len(SELECTED_CHUNKS)), fontsize=8)

axes[0].set_title("Raw AE waveforms — selected chunks")
axes[-1].set_xlabel("Time from chunk 0 (s)")

plt.show()
