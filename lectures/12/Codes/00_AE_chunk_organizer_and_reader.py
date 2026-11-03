"""Organize and combine acoustic-emission (AE) binary waveform files.

This program performs the same operations as the original ``Move_files.py``:

1. Find loose files named ``ae_chunkN_chM.bin``.
2. Move them into folders named ``ae_chunkN``.
3. Preserve the organized directory tree required by the AE Analyzer.
4. Optionally read selected chunks and channels.
5. Optionally combine consecutive chunks along the time axis.

The program validates the complete request before allocating the output array.
For large datasets, it writes directly to a memory-mapped ``.npy`` file so it
does not need to keep a second full copy of the data in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Sequence

import numpy as np


# =============================================================================
# USER SETTINGS
# Edit this section before running the program.
# =============================================================================

# The data folder is expected beside this Python file. You may instead enter an
# absolute path, for example Path("/Users/name/Desktop/UCS_sandstone_test1").
DATA_FOLDER = Path(__file__).resolve().parent / "./data_g2/UCS_sandstone_2/UCS_sandstone_2_test1"

# By default the program ONLY organizes the original .bin files. This preserves
# the directory structure expected by the AE Analyzer. Change this to True only
# if you also want to create a separate combined .npy file for other analyses.
COMBINE_AFTER_ORGANIZING = False

# The following settings are used only when COMBINE_AFTER_ORGANIZING = True.
# FIRST_CHUNK = 0 and NUMBER_OF_CHUNKS = 2 reads chunks 0 and 1.
FIRST_CHUNK = 0
NUMBER_OF_CHUNKS = 110

# Channels to read. Any selection from 1 to 32 can be used.
CHANNELS = [1, 2]

# Acquisition parameters for the laboratory files.
SAMPLING_RATE_HZ = 3.125e6
CHUNK_DURATION_S = 1.0
SAMPLE_DTYPE = "float16"
HEADER_BYTES = 128

# Move loose binary files into one folder per chunk before reading them.
ORGANIZE_FILES = True

# Name of the optional combined NumPy file created inside DATA_FOLDER.
OUTPUT_FILENAME = "combined_AE_chunks.npy"


# =============================================================================
# INTERNAL DEFINITIONS
# Normally, nothing below this line needs to be edited.
# =============================================================================

FILE_PATTERN = re.compile(
    r"^ae_chunk(?P<chunk>\d+)_ch(?P<channel>\d+)\.bin$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class WaveformFile:
    """Description of one validated waveform file."""

    chunk: int
    channel: int
    path: Path
    samples: int


def _positive_integer(value, name: str) -> int:
    """Convert *value* to an integer and reject fractional/invalid values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error

    integer = int(numeric)
    if not np.isfinite(numeric) or numeric != integer or integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def organize_waveform_files(folder_path) -> list[Path]:
    """Move loose AE files into their corresponding chunk folders.

    Example
    -------
    ``ae_chunk3_ch2.bin`` is moved to
    ``ae_chunk3/ae_chunk3_ch2.bin``.

    Existing files are not overwritten. If source and destination both exist
    with the same size, the source is left untouched and a message is printed.
    """
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found:\n{root}")

    moved = []
    skipped_names = []

    for source in sorted(root.glob("*.bin")):
        if source.name.startswith("."):
            continue

        match = FILE_PATTERN.fullmatch(source.name)
        if match is None:
            skipped_names.append(source.name)
            continue

        chunk = int(match.group("chunk"))
        destination_folder = root / f"ae_chunk{chunk}"
        destination_folder.mkdir(parents=True, exist_ok=True)
        destination = destination_folder / source.name

        if destination.exists():
            if source.stat().st_size == destination.stat().st_size:
                print(f"Already organized; loose copy left unchanged: {source.name}")
                continue
            raise FileExistsError(
                "Cannot organize the files because the destination already "
                f"exists with a different size:\n{destination}"
            )

        shutil.move(str(source), str(destination))
        moved.append(destination)

    for name in skipped_names:
        print(f"Ignored .bin file with an unexpected name: {name}")

    return moved


def _waveform_path(root: Path, chunk: int, channel: int) -> Path:
    """Return the expected path for one organized waveform file."""
    return root / f"ae_chunk{chunk}" / f"ae_chunk{chunk}_ch{channel}.bin"


def inspect_directory_tree(folder_path) -> dict[int, list[int]]:
    """Inspect the organized tree without opening or changing waveform files.

    Returns
    -------
    dict
        Mapping ``chunk_number -> sorted channel numbers``.

    Notes
    -----
    Only this structure is accepted::

        experiment_folder/
        ├── ae_chunk0/
        │   ├── ae_chunk0_ch1.bin
        │   └── ae_chunk0_ch2.bin
        └── ae_chunk1/
            ├── ae_chunk1_ch1.bin
            └── ae_chunk1_ch2.bin

    The function is read-only and therefore safe to run repeatedly.
    """
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found:\n{root}")

    tree: dict[int, list[int]] = {}

    for chunk_folder in sorted(root.iterdir()):
        if not chunk_folder.is_dir():
            continue

        folder_match = re.fullmatch(r"ae_chunk(\d+)", chunk_folder.name)
        if folder_match is None:
            continue

        folder_chunk = int(folder_match.group(1))
        channels = []

        for file_path in sorted(chunk_folder.glob("*.bin")):
            if file_path.name.startswith("."):
                continue

            file_match = FILE_PATTERN.fullmatch(file_path.name)
            if file_match is None:
                print(f"Ignored unexpected file: {file_path}")
                continue

            file_chunk = int(file_match.group("chunk"))
            channel = int(file_match.group("channel"))

            if file_chunk != folder_chunk:
                raise ValueError(
                    f"{file_path.name} belongs to chunk {file_chunk}, but it is "
                    f"inside folder ae_chunk{folder_chunk}."
                )
            if not 1 <= channel <= 32:
                raise ValueError(
                    f"Invalid channel {channel} in {file_path.name}; expected 1–32."
                )

            channels.append(channel)

        if channels:
            if len(channels) != len(set(channels)):
                raise ValueError(
                    f"Duplicate channel files were found in {chunk_folder}."
                )
            tree[folder_chunk] = sorted(channels)

    return dict(sorted(tree.items()))


def print_directory_summary(tree: dict[int, list[int]]) -> None:
    """Print a compact summary of the AE Analyzer input directories."""
    if not tree:
        print("No organized AE chunk folders were found.")
        return

    print("\nDirectory structure ready for the AE Analyzer:")
    for chunk, channels in tree.items():
        channel_text = ", ".join(str(channel) for channel in channels)
        print(f"  ae_chunk{chunk}: channels [{channel_text}]")

    all_channel_sets = {tuple(channels) for channels in tree.values()}
    if len(all_channel_sets) > 1:
        print(
            "\nNote: the available channels are not identical in every chunk. "
            "This can be valid, but select only existing channels in the analyzer."
        )


def _sample_count(file_path: Path, dtype: np.dtype, header_bytes: int) -> int:
    """Calculate the number of samples from file size without loading data."""
    file_bytes = file_path.stat().st_size
    data_bytes = file_bytes - header_bytes

    if data_bytes <= 0:
        raise ValueError(
            f"{file_path} contains no data after its {header_bytes}-byte header."
        )
    if data_bytes % dtype.itemsize:
        raise ValueError(
            f"{file_path.name}: {data_bytes} data bytes cannot be decoded as "
            f"{dtype.name} ({dtype.itemsize} bytes per sample). Check "
            "SAMPLE_DTYPE and HEADER_BYTES."
        )

    return data_bytes // dtype.itemsize


def validate_request(
    folder_path,
    chunk: int,
    list_channels: Sequence[int],
    step: int,
    fs: float,
    chunk_duration_s: float = 1.0,
    float_type="float16",
    header_bytes: int = 128,
) -> tuple[list[WaveformFile], list[int], np.dtype, int]:
    """Validate every requested file and acquisition parameter before loading."""
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found:\n{root}")

    first_chunk = int(chunk)
    number_of_chunks = _positive_integer(step, "step")
    chunk_numbers = list(range(first_chunk, first_chunk + number_of_chunks))

    channels = [int(channel) for channel in list_channels]
    if not channels:
        raise ValueError("list_channels must contain at least one channel.")
    if len(channels) != len(set(channels)):
        raise ValueError("list_channels contains duplicate channel numbers.")
    if any(channel < 1 or channel > 32 for channel in channels):
        raise ValueError("Channel numbers must be between 1 and 32.")

    try:
        fs = float(fs)
        duration = float(chunk_duration_s)
    except (TypeError, ValueError) as error:
        raise ValueError("fs and chunk_duration_s must be numbers.") from error

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sampling frequency.")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("chunk_duration_s must be positive and finite.")

    header_bytes = int(header_bytes)
    if header_bytes < 0:
        raise ValueError("header_bytes cannot be negative.")

    try:
        dtype = np.dtype(float_type)
    except TypeError as error:
        raise ValueError(f"Unsupported SAMPLE_DTYPE: {float_type!r}") from error
    if dtype.kind != "f":
        raise ValueError("SAMPLE_DTYPE must be a floating-point dtype.")

    expected_samples = int(round(fs * duration))
    missing = []
    descriptions = []

    for chunk_number in chunk_numbers:
        for channel in channels:
            path = _waveform_path(root, chunk_number, channel)
            if not path.is_file():
                missing.append(path)
                continue

            samples = _sample_count(path, dtype, header_bytes)
            descriptions.append(
                WaveformFile(chunk_number, channel, path, samples)
            )

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The complete request cannot be loaded. These files are missing:\n"
            f"{missing_text}\n\n"
            "Check FIRST_CHUNK, NUMBER_OF_CHUNKS, CHANNELS, and the filenames."
        )

    wrong_lengths = [item for item in descriptions if item.samples != expected_samples]
    if wrong_lengths:
        details = "\n".join(
            f"  - chunk {item.chunk}, channel {item.channel}: "
            f"{item.samples:,} samples"
            for item in wrong_lengths
        )
        raise ValueError(
            f"Expected {expected_samples:,} samples per file from fs={fs:g} Hz "
            f"and duration={duration:g} s, but found:\n{details}\n\n"
            "Check SAMPLING_RATE_HZ, CHUNK_DURATION_S, SAMPLE_DTYPE, and "
            "HEADER_BYTES."
        )

    return descriptions, chunk_numbers, dtype, expected_samples


def read_waveform(file_path, float_type="float16", header_bytes: int = 128):
    """Read one waveform after skipping its acquisition header."""
    path = Path(file_path)
    dtype = np.dtype(float_type)
    expected_count = _sample_count(path, dtype, int(header_bytes))

    with path.open("rb") as binary_file:
        binary_file.seek(int(header_bytes))
        waveform = np.fromfile(binary_file, dtype=dtype, count=expected_count)

    if waveform.size != expected_count:
        raise OSError(
            f"Could read only {waveform.size:,} of {expected_count:,} samples "
            f"from {path}."
        )
    return waveform


def combine_chunks(
    folder_path,
    chunk,
    list_channels,
    step,
    fs,
    float_type="float16",
    chunk_duration_s=1.0,
    header_bytes=128,
    output_path=None,
):
    """Combine selected consecutive chunks for all requested channels.

    This keeps the original function's main arguments for compatibility.

    If ``output_path`` is provided, the result is written directly to a
    memory-mapped NumPy file. This is recommended for many channels/chunks.
    If it is ``None``, the combined array is held in RAM.
    """
    files, chunk_numbers, dtype, samples_per_chunk = validate_request(
        folder_path=folder_path,
        chunk=chunk,
        list_channels=list_channels,
        step=step,
        fs=fs,
        chunk_duration_s=chunk_duration_s,
        float_type=float_type,
        header_bytes=header_bytes,
    )

    channels = [int(channel) for channel in list_channels]
    total_samples = samples_per_chunk * len(chunk_numbers)
    output_shape = (len(channels), total_samples)

    if output_path is None:
        combined = np.empty(output_shape, dtype=dtype)
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() != ".npy":
            output_path = output_path.with_suffix(".npy")
        combined = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=dtype,
            shape=output_shape,
        )

    file_lookup = {(item.chunk, item.channel): item for item in files}

    for chunk_index, chunk_number in enumerate(chunk_numbers):
        left = chunk_index * samples_per_chunk
        right = left + samples_per_chunk

        for channel_index, channel in enumerate(channels):
            item = file_lookup[(chunk_number, channel)]
            combined[channel_index, left:right] = read_waveform(
                item.path,
                float_type=dtype,
                header_bytes=header_bytes,
            )

        print(
            f"Loaded chunk {chunk_number} "
            f"({chunk_index + 1}/{len(chunk_numbers)})"
        )

    if isinstance(combined, np.memmap):
        combined.flush()

    return combined, chunk_numbers


def bandpass_filter(signal_data, fs, lowcut, highcut, order=6):
    """Apply an optional zero-phase Butterworth band-pass filter.

    SciPy is imported only when this optional function is called.
    """
    try:
        from scipy import signal
    except ImportError as error:
        raise ImportError(
            "SciPy is required only for filtering. Install it with "
            "'conda install scipy'."
        ) from error

    fs = float(fs)
    lowcut = float(lowcut)
    highcut = float(highcut)
    nyquist = fs / 2

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            "Filter frequencies must satisfy "
            "0 < lowcut < highcut < fs/2."
        )

    sos = signal.butter(
        int(order),
        [lowcut / nyquist, highcut / nyquist],
        btype="bandpass",
        output="sos",
    )
    return signal.sosfiltfilt(sos, signal_data, axis=-1)


def main():
    """Organize files for the AE Analyzer and optionally combine them."""
    root = DATA_FOLDER.expanduser().resolve()

    print("AE waveform organizer and reader")
    print(f"Data folder: {root}")

    if not root.is_dir():
        raise FileNotFoundError(
            f"The configured DATA_FOLDER does not exist:\n{root}\n\n"
            "Edit DATA_FOLDER in the USER SETTINGS section."
        )

    if ORGANIZE_FILES:
        moved = organize_waveform_files(root)
        print(f"Organized {len(moved)} loose waveform file(s).")

    # Check and display the persistent directory tree used by the AE Analyzer.
    # No waveform file is renamed, converted, or deleted during this step.
    directory_tree = inspect_directory_tree(root)
    print_directory_summary(directory_tree)

    if not directory_tree:
        raise FileNotFoundError(
            "No usable ae_chunkN/ae_chunkN_chM.bin files were found."
        )

    if not COMBINE_AFTER_ORGANIZING:
        print("\nOrganization completed successfully.")
        print(f"Select this experiment folder in the AE Analyzer:\n{root}")
        return

    output_path = root / OUTPUT_FILENAME
    combined, chunk_numbers = combine_chunks(
        folder_path=root,
        chunk=FIRST_CHUNK,
        list_channels=CHANNELS,
        step=NUMBER_OF_CHUNKS,
        fs=SAMPLING_RATE_HZ,
        float_type=SAMPLE_DTYPE,
        chunk_duration_s=CHUNK_DURATION_S,
        header_bytes=HEADER_BYTES,
        output_path=output_path,
    )

    duration = combined.shape[1] / float(SAMPLING_RATE_HZ)
    size_mb = output_path.stat().st_size / 1024**2

    print("\nOrganization and optional combination completed successfully")
    print(f"Chunks: {chunk_numbers}")
    print(f"Channels: {CHANNELS}")
    print(f"Array shape: {combined.shape}")
    print(f"Combined duration: {duration:g} s per channel")
    print(f"Saved file: {output_path}")
    print(f"Saved size: {size_mb:.1f} MiB")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nThe program stopped because of the following problem:")
        print(error)
        raise SystemExit(1)
