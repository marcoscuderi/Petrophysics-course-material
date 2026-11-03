"""Data loading and single-channel AE detection for the lecture prototype.

Requirements: numpy, scipy, numba; pandas for NPZ files containing Series.
Binary files use the format in Example_Marco.py:
The current acquisition files are NumPy arrays saved with a .bin extension.
Their header records dtype and shape. Legacy raw files with a 128-byte header
remain supported when dtype and sampling frequency are supplied manually.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from numba import njit


CHUNK_RE = re.compile(r"^(?:ae_)?chunk(\d+)$", re.IGNORECASE)
FILE_RE = re.compile(r"^(?:ae_)?chunk(\d+)_ch(\d+)\.bin$", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingWindow:
    onset_row: int
    first_chunk: int
    last_chunk: int
    onset_time_s: float | None = None

    def anchor_time(self, mech_fs: float) -> float:
        return self.onset_time_s if self.onset_time_s is not None else self.onset_row / mech_fs

    def mechanical_time(self, chunk: int, sample: int, ae_fs: float, mech_fs: float) -> float:
        return self.anchor_time(mech_fs) + (chunk - self.first_chunk) + sample / ae_fs


@dataclass(frozen=True)
class DetectorConfig:
    sta_us: float = 20.0
    lta_us: float = 200.0
    trigger: float = 3.5
    detrigger: float = 1.5
    min_amplitude: float = 0.01
    highpass_khz: float = 0.0
    bandpass_enabled: bool = False
    lowcut_khz: float = 50.0
    highcut_khz: float = 1000.0

    def validate(self, fs: float) -> None:
        if not (0 < self.sta_us < self.lta_us):
            raise ValueError("STA must be positive and shorter than LTA")
        if not (1 <= self.detrigger < self.trigger):
            raise ValueError("Require 1 ≤ detrigger < trigger")
        nyquist_khz = fs / 2000
        if self.min_amplitude < 0:
            raise ValueError("Minimum amplitude cannot be negative")
        if self.bandpass_enabled:
            if not (0 < self.lowcut_khz < self.highcut_khz < nyquist_khz):
                raise ValueError(
                    f"Band-pass must satisfy 0 < low < high < {nyquist_khz:g} kHz"
                )
        elif not (0 <= self.highpass_khz < nyquist_khz):
            raise ValueError("Invalid high-pass frequency")


@dataclass(frozen=True)
class WaveformFormat:
    """Format information read from one waveform file."""

    dtype: str | None
    samples: int | None
    sampling_rate_hz: float | None
    header_bytes: int
    numpy_header: bool


def inspect_waveform_file(path: str | Path, chunk_duration_s: float = 1.0) -> WaveformFormat:
    """Read dtype and sample count from a NumPy-formatted waveform header.

    Files produced by the current recorder start with NumPy's ``\x93NUMPY``
    signature even though their extension is ``.bin``. For legacy raw files,
    automatic detection is not possible and the returned dtype/sample count
    are ``None``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Waveform file not found: {path}")
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")

    with path.open("rb") as stream:
        signature = stream.read(6)

    if signature != b"\x93NUMPY":
        return WaveformFormat(None, None, None, 128, False)

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path.name} must contain one floating-point waveform")
    dtype = np.dtype(array.dtype)
    if dtype.name not in ("float16", "float32", "float64"):
        raise ValueError(f"Unsupported waveform dtype {dtype} in {path.name}")
    header_bytes = int(getattr(array, "offset", 128))
    samples = int(array.size)
    return WaveformFormat(
        dtype=dtype.name,
        samples=samples,
        sampling_rate_hz=samples / float(chunk_duration_s),
        header_bytes=header_bytes,
        numpy_header=True,
    )


def detect_waveform_format(
    chunk_map: dict[int, dict[int, Path]],
    chunk_duration_s: float = 1.0,
) -> WaveformFormat:
    """Detect and verify one consistent format across a waveform directory."""
    paths = [path for channels in chunk_map.values() for path in channels.values()]
    if not paths:
        raise ValueError("No waveform files were found")

    first = inspect_waveform_file(paths[0], chunk_duration_s)
    if not first.numpy_header:
        raise ValueError(
            "This directory contains legacy raw files without a NumPy header. "
            "Disable automatic format detection and enter dtype/rate manually."
        )

    for path in paths[1:]:
        item = inspect_waveform_file(path, chunk_duration_s)
        if (not item.numpy_header or item.dtype != first.dtype or
                item.samples != first.samples):
            raise ValueError(
                f"Waveform format differs in {path.name}; expected "
                f"{first.dtype} with {first.samples:,} samples"
            )
    return first


def available_chunks(
    data_dir: str | Path,
    *,
    return_ignored_empty: bool = False,
) -> (dict[int, dict[int, Path]] |
      tuple[dict[int, dict[int, Path]], list[Path]]):
    """Index waveform files while ignoring inactive, empty channels.

    Some multi-channel recorders create one NumPy file for every hardware
    channel even when only a subset is active. A file whose waveform contains
    zero samples is therefore treated as an inactive channel rather than a
    format error. Set ``return_ignored_empty=True`` when the caller also needs
    the paths that were skipped so it can inform the operator.

    Non-empty and legacy files remain in the index. Malformed files therefore
    still reach the usual validation code and produce an explicit error.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Data directory does not exist: {root}")
    result: dict[int, dict[int, Path]] = {}
    ignored_empty: list[Path] = []
    if not any(x.is_dir() and CHUNK_RE.fullmatch(x.name) for x in root.iterdir()):
        candidates = [x for x in root.iterdir() if x.is_dir() and
                      any(y.is_dir() and CHUNK_RE.fullmatch(y.name) for y in x.iterdir())]
        if len(candidates) == 1:
            root = candidates[0]
        elif len(candidates) > 1:
            raise ValueError("Multiple AE subfolders found; select the intended data folder")
    for folder in root.iterdir():
        match = CHUNK_RE.fullmatch(folder.name)
        if not (folder.is_dir() and match):
            continue
        chunk = int(match.group(1))
        if chunk in result:
            raise ValueError(f"Duplicate folders for chunk {chunk}")
        channels = {}
        for path in folder.iterdir():
            m = FILE_RE.fullmatch(path.name)
            if path.is_file() and m and int(m.group(1)) == chunk:
                # Zero-length NumPy arrays are placeholders for inactive
                # acquisition channels. They must not appear in the channel
                # selector or take part in automatic format detection.
                try:
                    waveform_format = inspect_waveform_file(path)
                except Exception:
                    waveform_format = None
                if (waveform_format is not None and
                        waveform_format.numpy_header and
                        waveform_format.samples == 0):
                    ignored_empty.append(path)
                    continue
                channels[int(m.group(2))] = path
        result[chunk] = channels
    ordered = dict(sorted(result.items()))
    if return_ignored_empty:
        return ordered, ignored_empty
    return ordered


def chunk_file(data_dir: str | Path, chunk: int, channel: int) -> Path:
    return Path(data_dir) / f"ae_chunk{chunk}" / f"ae_chunk{chunk}_ch{channel}.bin"


def read_chunk(path: Path, dtype: str, fs: float) -> np.memmap:
    if dtype not in ("float16", "float32", "float64"):
        raise ValueError("Select float16, float32, or float64")
    detected = inspect_waveform_file(path)
    if detected.numpy_header:
        if detected.dtype != np.dtype(dtype).name:
            raise ValueError(
                f"{path.name} header says {detected.dtype}, but the operator "
                f"selected {np.dtype(dtype).name}"
            )
        if abs(float(detected.samples) - float(fs)) > 1:
            raise ValueError(
                f"{path.name}: {detected.samples} samples; expected {fs:g} "
                "for one second"
            )
        return np.load(path, mmap_mode="r", allow_pickle=False)

    dt = np.dtype(dtype)
    size = path.stat().st_size
    if size < 128 or (size - 128) % dt.itemsize:
        raise ValueError(f"Invalid binary length or header: {path.name}")
    n = (size - 128) // dt.itemsize
    if abs(n - fs) > 1:
        raise ValueError(f"{path.name}: {n} samples; expected {fs:g} for one second")
    return np.memmap(path, mode="r", dtype=dt, offset=128, shape=(n,))


def filter_sos(config: DetectorConfig, fs: float):
    """Return the configured Butterworth filter or ``None``."""
    config.validate(fs)
    if config.bandpass_enabled:
        return butter(
            4,
            [config.lowcut_khz * 1000, config.highcut_khz * 1000],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
    if config.highpass_khz:
        return butter(
            4,
            config.highpass_khz * 1000,
            btype="highpass",
            fs=fs,
            output="sos",
        )
    return None


def filter_waveform(raw: np.ndarray, fs: float, config: DetectorConfig) -> np.ndarray:
    """Filter one independent waveform for plotting, returning float32."""
    samples = np.asarray(raw, dtype=np.float32)
    sos = filter_sos(config, fs)
    if sos is None or not len(samples):
        return samples
    zi = sosfilt_zi(sos) * float(samples[0])
    filtered, _ = sosfilt(sos, samples, zi=zi)
    return filtered.astype(np.float32, copy=False)


@dataclass
class MechanicalData:
    values: np.ndarray
    names: list[str]
    time_s: np.ndarray | None = None


def _mechanical_result(arr: np.ndarray, names: list[str]) -> MechanicalData:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number) or not len(arr):
        raise ValueError("Mechanical data must be a nonempty numeric 1D or 2D array")
    if len(names) != arr.shape[1]:
        raise ValueError("Mechanical column-name count differs from the array")
    time_idx = next((i for i, name in enumerate(names)
                     if name.strip().lower() in ("time_s", "time (s)", "time [s]")), None)
    times = np.asarray(arr[:, time_idx], dtype=float) if time_idx is not None else None
    if times is not None and (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("Mechanical time_s must be finite and strictly increasing")
    return MechanicalData(arr, names, times)


def _mechanical_matrix(columns: dict[str, object]) -> MechanicalData:
    if not columns:
        raise ValueError("No mechanical channels found")
    names = list(columns)
    arrays = [np.asarray(columns[k]) for k in names]
    n = len(arrays[0])
    if not all(a.ndim == 1 and len(a) == n and np.issubdtype(a.dtype, np.number)
               for a in arrays):
        raise ValueError("All mechanical channels must be numeric 1D arrays of equal length")
    return _mechanical_result(np.column_stack(arrays), names)


def load_mechanical(path: str | Path) -> MechanicalData:
    """Read NumPy arrays, named NPZ channels, or numeric text tables.

    Object-valued NPZ files, including the provided pandas-Series dictionary,
    use Python pickle and should be opened only from trusted sources.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        if arr.dtype.names:
            return _mechanical_matrix({name: arr[name] for name in arr.dtype.names})
        ncols = 1 if arr.ndim == 1 else arr.shape[1] if arr.ndim == 2 else 0
        return _mechanical_result(arr, [f"column {i+1}" for i in range(ncols)])
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            if len(archive.files) == 1:
                data = archive[archive.files[0]]
                if data.dtype == object and data.shape == ():
                    item = data.item()
                    if not isinstance(item, dict):
                        raise ValueError("Expected an NPZ dictionary of named mechanical channels")
                    return _mechanical_matrix(item)
                if data.dtype.names:
                    return _mechanical_matrix({name: data[name] for name in data.dtype.names})
                if data.ndim in (1, 2) and np.issubdtype(data.dtype, np.number):
                    ncols = 1 if data.ndim == 1 else data.shape[1]
                    return _mechanical_result(data, [f"column {i+1}" for i in range(ncols)])
            return _mechanical_matrix({key: archive[key] for key in archive.files})
    # In the supplied text example the first row names columns, the second gives units.
    with path.open("rt", encoding="utf-8-sig") as file:
        first = file.readline().strip()
        second = file.readline().strip()
    if not first:
        raise ValueError("Empty mechanical text file")
    delimiter = "\t" if "\t" in first else "," if "," in first else None
    split = (lambda s: s.split(delimiter)) if delimiter else (lambda s: s.split())
    first_fields = [x.strip() for x in split(first)]
    second_fields = [x.strip() for x in split(second)]
    def numeric_row(fields):
        try:
            for value in fields:
                float(value)
            return bool(fields)
        except ValueError:
            return False
    names = first_fields if not numeric_row(first_fields) else [
        f"column {i+1}" for i in range(len(first_fields))]
    skip = (0 if numeric_row(first_fields) else 1) + (
        0 if numeric_row(second_fields) else 1)
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skip, ndmin=2)
    return _mechanical_result(arr, names)


def peak_preserving_plot(y: np.ndarray, x0: float, dt: float, max_bins: int = 3000):
    """Min/max bins retain short AE spikes that ordinary decimation hides."""
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    stride = max(1, int(np.ceil(n / max_bins)))
    if stride == 1:
        return x0 + np.arange(n) * dt, np.asarray(y)
    usable = n // stride * stride
    block = np.asarray(y[:usable]).reshape(-1, stride)
    lo = block.argmin(axis=1)
    hi = block.argmax(axis=1)
    offsets = np.arange(len(block)) * stride
    ix = np.stack((offsets + lo, offsets + hi), axis=1)
    ix.sort(axis=1)
    ix = ix.ravel()
    if usable < n:
        ix = np.r_[ix, np.arange(usable, n)]
    return x0 + ix * dt, np.asarray(y[ix])


@njit(cache=True)
def _scan(signal, nsta, nlta, trigger, detrigger, min_amp,
          sta_ring, lta_ring, sta_sum, lta_sum, seen, active,
          onset, max_val, min_val, lta_frozen, offset, preview):
    """Causal STA/LTA. Pause LTA updates during an event; retain state across chunks."""
    n = signal.size
    sta_arr = np.empty(n if preview else 0, dtype=np.float32)
    lta_arr = np.empty(n if preview else 0, dtype=np.float32)
    ratio_arr = np.empty(n if preview else 0, dtype=np.float32)
    events = []
    for i in range(n):
        value = float(signal[i])
        a = abs(value)
        si = seen % nsta
        sta_sum += a - sta_ring[si]
        sta_ring[si] = a
        sta = sta_sum / min(seen + 1, nsta)
        if not active:
            # The LTA baseline is held fixed while an AE is active.
            li = seen % nlta  # adjusted after the event below using a separate counter
            # The ring index itself is passed independently in the lta state.
            li = int(lta_ring[-1]) % nlta
            lta_sum += a - lta_ring[li]
            lta_ring[li] = a
            lta_ring[-1] = li + 1
            lta = lta_sum / min(int(lta_ring[-2]) + 1, nlta)
            lta_ring[-2] += 1
        else:
            lta = lta_frozen
        ratio = sta / max(lta, 1e-12)
        if not active and int(lta_ring[-2]) >= nlta and ratio >= trigger:
            active = True
            onset = offset + i
            max_val = value
            min_val = value
            lta_frozen = lta
        elif active:
            max_val = max(max_val, value)
            min_val = min(min_val, value)
            if ratio <= detrigger:
                amp = max_val - min_val
                events.append((onset, offset + i, amp, amp >= min_amp))
                active = False
        if preview:
            sta_arr[i] = sta
            lta_arr[i] = lta
            ratio_arr[i] = ratio
        seen += 1
    return (events, sta_arr, lta_arr, ratio_arr, sta_sum, lta_sum,
            seen, active, onset, max_val, min_val, lta_frozen)


class AEPicker:
    def __init__(self, fs: float, config: DetectorConfig, preview=False):
        config.validate(fs)
        self.fs, self.config, self.preview = fs, config, preview
        self.nsta = max(1, round(config.sta_us * 1e-6 * fs))
        self.nlta = max(self.nsta + 1, round(config.lta_us * 1e-6 * fs))
        self.sta_ring = np.zeros(self.nsta, dtype=np.float64)
        # Two trailing slots hold the number of LTA updates and ring index.
        self.lta_ring = np.zeros(self.nlta + 2, dtype=np.float64)
        self.sta_sum = self.lta_sum = 0.0
        self.seen = 0
        self.active = False
        self.onset = -1
        self.max_val = self.min_val = self.lta_frozen = 0.0
        self.offset = 0
        self.sos = filter_sos(config, fs)
        self.zi = None

    def process(self, raw: np.ndarray):
        samples = np.asarray(raw, dtype=np.float32)
        if self.sos is not None:
            if self.zi is None:
                self.zi = sosfilt_zi(self.sos) * float(samples[0])
            samples, self.zi = sosfilt(self.sos, samples, zi=self.zi)
            samples = samples.astype(np.float32, copy=False)
        result = _scan(samples, self.nsta, self.nlta, self.config.trigger,
                       self.config.detrigger, self.config.min_amplitude,
                       self.sta_ring, self.lta_ring, self.sta_sum, self.lta_sum,
                       self.seen, self.active, self.onset, self.max_val,
                       self.min_val, self.lta_frozen, self.offset, self.preview)
        (events, sta, lta, ratio, self.sta_sum, self.lta_sum, self.seen,
         self.active, self.onset, self.max_val, self.min_val, self.lta_frozen) = result
        self.offset += len(samples)
        return samples, events, sta, lta, ratio

    def close(self):
        """At a recording boundary, mark an unfinished candidate as truncated."""
        if self.active:
            amp = self.max_val - self.min_val
            self.active = False
            return (self.onset, self.offset - 1, amp, False)
        return None


def catalog_row(event, recording: RecordingWindow, fs: float, mech_fs: float,
                channel: int, first_chunk_samples: int, selected_first_chunk: int):
    start, end, amp, accepted = event
    chunk_offset, local_sample = divmod(start, first_chunk_samples)
    chunk = selected_first_chunk + chunk_offset
    return dict(channel=channel, chunk=chunk, sample_in_chunk=local_sample,
                onset_s=recording.mechanical_time(chunk, local_sample, fs, mech_fs),
                end_s=(recording.anchor_time(mech_fs) +
                       selected_first_chunk-recording.first_chunk + end / fs),
                duration_us=(end - start) / fs * 1e6,
                peak_to_peak=amp, accepted=accepted)
