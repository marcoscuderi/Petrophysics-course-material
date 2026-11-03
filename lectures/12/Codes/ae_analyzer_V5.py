"""Lecture 12/13 acoustic emission analyzer (first prototype).

Install: python -m pip install PyQt6 matplotlib numpy scipy numba pandas
Run:     python ae_analyzer.py

The mechanical tab defines each AE recording window with an onset row and
inclusive first/last chunk. The AE tab operates on one sensor at a time.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
from threading import Event

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QComboBox,
    QTabWidget, QGroupBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QListWidget,
    QAbstractItemView, QCheckBox, QProgressBar,
)
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from ae_core import (
    AEPicker, DetectorConfig, RecordingWindow, available_chunks,
    read_chunk, load_mechanical as load_mechanical_standard,
    peak_preserving_plot, catalog_row, detect_waveform_format,
    filter_waveform,
)


class MechanicalData:
    """Small container returned by the robust text-file reader."""

    def __init__(self, values, names, time_s=None, notices=None):
        self.values = values
        self.names = names
        self.time_s = time_s
        self.notices = notices or []


def _is_numeric_row(fields):
    """Return True only when every field in a row is a valid number."""
    if not fields:
        return False
    try:
        for value in fields:
            float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_mechanical_text(path):
    """Read a CSV/TSV/space-delimited mechanical file safely.

    Empty cells are converted to NaN. A column that contains no numerical
    values is ignored and reported to the user instead of stopping the GUI.
    The common two-row format (column names followed by units) is supported.
    """
    path = Path(path)

    # Inspect the first two rows to determine the delimiter, column names and
    # whether a second units row must be skipped.
    with path.open("rt", encoding="utf-8-sig") as file:
        first = file.readline().rstrip("\r\n")
        second = file.readline().rstrip("\r\n")

    if not first.strip():
        raise ValueError("The mechanical text file is empty.")

    delimiter = "\t" if "\t" in first else "," if "," in first else None
    split = (lambda text: text.split(delimiter)) if delimiter else (lambda text: text.split())
    first_fields = [value.strip() for value in split(first)]
    second_fields = [value.strip() for value in split(second)]

    first_is_numeric = _is_numeric_row(first_fields)
    names = ([f"column {index + 1}" for index in range(len(first_fields))]
             if first_is_numeric else first_fields)
    skip_rows = (0 if first_is_numeric else 1)
    if second and not _is_numeric_row(second_fields):
        skip_rows += 1

    # Preserve the cells as text initially. This lets us distinguish a truly
    # missing value from an unexpected word inside the numerical data.
    separator = delimiter if delimiter is not None else r"\s+"
    table = pd.read_csv(
        path,
        sep=separator,
        header=None,
        names=names,
        skiprows=skip_rows,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
    )

    if table.empty:
        raise ValueError("The mechanical text file contains no data rows.")

    text_table = table.apply(lambda column: column.astype(str).str.strip())
    missing_tokens = {"", "na", "nan", "n/a", "null", "none"}
    missing = text_table.apply(
        lambda column: column.str.lower().isin(missing_tokens)
    )
    numeric_table = text_table.apply(pd.to_numeric, errors="coerce")

    # Do not silently hide spelling mistakes or other unexpected text in a
    # channel that should contain numbers.
    invalid = numeric_table.isna() & ~missing
    if invalid.any().any():
        row_index, column_index = np.argwhere(invalid.to_numpy())[0]
        column_name = str(numeric_table.columns[column_index])
        bad_value = text_table.iat[row_index, column_index]
        raise ValueError(
            f"Non-numeric value {bad_value!r} in column {column_name!r}, "
            f"data row {row_index + 1}."
        )

    notices = []

    # A completely empty channel carries no information and would previously
    # make np.loadtxt fail. Remove it while keeping all valid channels.
    empty_columns = [str(name) for name in numeric_table.columns
                     if numeric_table[name].isna().all()]
    if empty_columns:
        numeric_table = numeric_table.drop(columns=empty_columns)
        notices.append(
            "Ignored completely empty column(s): " + ", ".join(empty_columns)
        )

    if numeric_table.shape[1] == 0:
        raise ValueError("No numerical mechanical channels were found.")

    # Partly missing channels remain available as NaN, which Matplotlib can
    # plot as gaps. Report them so the user knows that data are absent.
    partial_missing = {
        str(name): int(count)
        for name, count in numeric_table.isna().sum().items()
        if 0 < count < len(numeric_table)
    }
    if partial_missing:
        details = ", ".join(
            f"{name} ({count} missing)" for name, count in partial_missing.items()
        )
        notices.append("Retained partially incomplete column(s) as NaN: " + details)

    values = numeric_table.to_numpy(dtype=float)
    names = [str(name) for name in numeric_table.columns]

    time_index = next(
        (index for index, name in enumerate(names)
         if name.strip().lower() in ("time_s", "time (s)", "time [s]")),
        None,
    )
    time_s = values[:, time_index] if time_index is not None else None
    if time_s is not None:
        if not np.all(np.isfinite(time_s)):
            raise ValueError("The mechanical time column contains missing or invalid values.")
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("Mechanical time must be strictly increasing.")

    return MechanicalData(values, names, time_s, notices)


def load_mechanical(path):
    """Use the established loader for NumPy files and robust text handling."""
    if Path(path).suffix.lower() in {".npy", ".npz"}:
        return load_mechanical_standard(path)
    return load_mechanical_text(path)


def numeric(value: str, label: str, integer=False):
    try:
        result = int(value) if integer else float(value)
        if not np.isfinite(result):
            raise ValueError()
        return result
    except ValueError as exc:
        raise ValueError(f"Enter a valid {label}") from exc


class AnalysisWorker(QThread):
    finished_catalog = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int, int)

    def __init__(self, tasks, config, parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self.config = config
        self.stop_requested = Event()

    def cancel(self):
        self.stop_requested.set()

    def run(self):
        try:
            rows = []
            total_files = sum(len(task["chunks"]) for task in self.tasks)
            completed_files = 0

            for task in self.tasks:
                picker = AEPicker(task["fs"], self.config)
                samples_per_chunk = None
                first_chunk = task["chunks"][0]

                for chunk in task["chunks"]:
                    if self.stop_requested.is_set():
                        self.cancelled.emit()
                        return
                    raw = read_chunk(
                        task["chunk_map"][chunk][task["channel"]],
                        task["dtype"],
                        task["fs"],
                    )
                    if samples_per_chunk is None:
                        samples_per_chunk = len(raw)
                    if len(raw) != samples_per_chunk:
                        raise ValueError(
                            "Chunk sample counts differ; cannot preserve event timing"
                        )
                    _, events, _, _, _ = picker.process(raw)
                    for event in events:
                        row = catalog_row(
                            event,
                            task["recording"],
                            task["fs"],
                            task["mech_fs"],
                            task["channel"],
                            samples_per_chunk,
                            first_chunk,
                        )
                        row.update({
                            "recording_start_chunk": task["recording"].first_chunk,
                            "window_index": task["window_index"],
                            "window": task["window_name"],
                            "window_start_s": task["recording"].anchor_time(task["mech_fs"]),
                            "window_end_s": (
                                task["recording"].anchor_time(task["mech_fs"])
                                + task["recording"].last_chunk
                                - task["recording"].first_chunk + 1
                            ),
                            "source_dir": str(task["source_dir"]),
                        })
                        rows.append(row)
                    if len(rows) > 1_000_000:
                        raise ValueError(
                            "More than one million candidate intervals. Increase "
                            "the trigger threshold or inspect the noise."
                        )
                    completed_files += 1
                    self.progress.emit(completed_files, total_files, len(rows))

                tail = picker.close()
                if tail:
                    row = catalog_row(
                        tail,
                        task["recording"],
                        task["fs"],
                        task["mech_fs"],
                        task["channel"],
                        samples_per_chunk,
                        first_chunk,
                    )
                    row.update({
                        "recording_start_chunk": task["recording"].first_chunk,
                        "window_index": task["window_index"],
                        "window": task["window_name"],
                        "window_start_s": task["recording"].anchor_time(task["mech_fs"]),
                        "window_end_s": (
                            task["recording"].anchor_time(task["mech_fs"])
                            + task["recording"].last_chunk
                            - task["recording"].first_chunk + 1
                        ),
                        "source_dir": str(task["source_dir"]),
                        "truncated_at_end": True,
                    })
                    rows.append(row)
            self.finished_catalog.emit(rows)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Analyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mechanical and AE Analyzer — Lecture 12/13")
        self.resize(1720, 1000)
        self.mech = None
        self.mech_times = None
        self.data_dir = None
        self.chunk_map = {}
        self.window_sources = []
        self._source_cache = {}
        self.windows = []
        self.rows = []
        self.catalog_ready = False
        self.worker = None
        self.overview_data = None
        self.preview_range = None
        self._reset_view_on_next_preview = False
        self._preview_request = 0
        self._updating_format_controls = False
        self._pending_replace_catalog = False
        self.full_resolution_windows = []
        self._building = True
        self.build_ui()
        self._building = False

    def err(self, exc):
        QMessageBox.warning(self, "Check input", str(exc))

    def mech_fs(self):
        return self.mech_rate.value()

    def ae_fs(self):
        return self.ae_rate.value() * 1e6

    def dtype(self):
        return self.dtype_box.currentText()

    def apply_detected_format(self, source):
        """Display the header-derived dtype/rate for the current window."""
        if not self.auto_format.isChecked() or source is None:
            return
        detected = source.get("detected")
        if detected is None:
            self.statusBar().showMessage(
                source.get("detection_error") or "Automatic format detection failed",
                12000,
            )
            return
        self._updating_format_controls = True
        self.ae_rate.setValue(detected.sampling_rate_hz / 1e6)
        self.dtype_box.setCurrentText(detected.dtype)
        self._updating_format_controls = False
        self.statusBar().showMessage(
            f"Detected {detected.dtype}, {detected.sampling_rate_hz/1e6:g} MHz "
            f"from waveform header" + self.empty_channel_notice(source)
        )

    @staticmethod
    def empty_channel_notice(source):
        """Describe recorder-created empty channels skipped during indexing."""
        ignored = source.get("ignored_empty", []) if source is not None else []
        if not ignored:
            return ""
        channels = set()
        for path in ignored:
            match = re.search(r"_ch(\d+)\.bin$", path.name, re.IGNORECASE)
            if match:
                channels.add(int(match.group(1)))
        channel_text = ", ".join(str(channel) for channel in sorted(channels))
        detail = f" (channels {channel_text})" if channel_text else ""
        return f" · ignored {len(ignored)} empty channel file(s){detail}"

    def format_for_window(self, window_index):
        """Return dtype and sampling rate for one recording window."""
        source = self.window_sources[window_index]
        if source is None:
            raise ValueError(f"Window {window_index + 1} has no waveform folder")
        if self.auto_format.isChecked():
            detected = source.get("detected")
            if detected is None:
                raise ValueError(
                    source.get("detection_error")
                    or f"Could not detect format for window {window_index + 1}"
                )
            return detected.dtype, float(detected.sampling_rate_hz)
        return self.dtype(), self.ae_fs()

    def on_auto_format_changed(self, _state):
        automatic = self.auto_format.isChecked()
        self.ae_rate.setEnabled(not automatic)
        self.dtype_box.setEnabled(not automatic)
        index = self.recording_box.currentIndex()
        if automatic and 0 <= index < len(self.window_sources):
            self.apply_detected_format(self.window_sources[index])
        if self.data_dir is not None and self.channel_box.count():
            QTimer.singleShot(0, self.show_overview)

    def on_format_override_changed(self, *_):
        if self._updating_format_controls or self.auto_format.isChecked():
            return
        if self.data_dir is not None and self.channel_box.count():
            QTimer.singleShot(0, self.show_overview)

    def build_ui(self):
        self.setStyleSheet("""
          QWidget {background:#292d33; color:#f4f4f4;}
          QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget
            {background:#171b20; color:#f4f4f4; border:1px solid #59616a; padding:3px;}
          QPushButton {background:#3d4851; color:white; border:1px solid #70818b;
            padding:5px 9px; border-radius:3px;}
          QPushButton:hover {background:#536670;}
          QPushButton:disabled {color:#888;}
          QGroupBox {border:1px solid #63727a; margin-top:12px; padding-top:12px;
            color:#9ed2e4; font-weight:bold;}
          QTabBar::tab {background:#3d4851; padding:9px 15px;}
          QTabBar::tab:selected {background:#287c73;}
        """)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.build_mechanical_tab()
        self.build_ae_tab()
        self.build_catalog_tab()
        self.statusBar().showMessage("Load mechanical data, then define recording windows.")

    def make_canvas(self, axes=1, height=6):
        fig = Figure(figsize=(10, height), layout="constrained")
        fig.get_layout_engine().set(w_pad=.02, h_pad=.015, hspace=.02)
        if axes == 1:
            axs = fig.add_subplot(111)
        else:
            axs = fig.subplots(axes, 1, sharex=False,
                               gridspec_kw={'height_ratios': [2.0, 1.4, 1.3, 1.3]})
            # The overview remains on the full recording. The three analysis
            # panels zoom and pan together within the selected chunk.
            for ax in axs[2:]:
                ax.sharex(axs[1])
            # The analysis panels use one time scale; only the lowest panel
            # needs numeric x tick labels.
            axs[1].tick_params(axis="x", labelbottom=False)
            axs[2].tick_params(axis="x", labelbottom=False)
        canvas = FigureCanvas(fig)
        return fig, axs, canvas

    def build_mechanical_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        first = QHBoxLayout()
        load = QPushButton("Load mechanical file")
        load.clicked.connect(self.open_mechanical)
        self.mech_label = QLabel("No mechanical file")
        self.mech_rate = QDoubleSpinBox()
        self.mech_rate.setRange(.001, 1e7); self.mech_rate.setDecimals(3)
        self.mech_rate.setValue(1000); self.mech_rate.setSuffix(" Hz")
        self.mech_rate.valueChanged.connect(self.validate_windows)
        first.addWidget(load); first.addWidget(self.mech_label, 1)
        first.addWidget(QLabel("Rate if no time_s:")); first.addWidget(self.mech_rate)
        layout.addLayout(first)

        second = QHBoxLayout()
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText("Column names, comma separated: shear stress, slip velocity, ...")
        self.names_edit.editingFinished.connect(self.reset_columns)
        second.addWidget(QLabel("Column names:")); second.addWidget(self.names_edit, 1)
        layout.addLayout(second)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        side = QWidget(); side.setMinimumWidth(300); side.setMaximumWidth(420)
        sl = QVBoxLayout(side)
        graph = QGroupBox("Mechanical graph")
        gf = QFormLayout(graph)
        self.x_box = QComboBox(); self.y_list = QListWidget()
        self.y_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.x_box.currentIndexChanged.connect(self.plot_mechanical)
        self.y_list.itemSelectionChanged.connect(self.plot_mechanical)
        gf.addRow("Horizontal axis", self.x_box)
        gf.addRow("Plot columns", self.y_list)
        sl.addWidget(graph)

        windows_group = QGroupBox("AE recording windows")
        wl = QVBoxLayout(windows_group)
        wl.addWidget(QLabel("Each onset row aligns with sample 0 of its first chunk.\n"
                            "Assign a separate waveform folder to every window."))
        self.window_table = QTableWidget(0, 4)
        self.window_table.setHorizontalHeaderLabels(
            ["Onset row (0-based)", "First chunk", "Last chunk", "Waveform folder"]
        )
        self.window_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.window_table.itemChanged.connect(self.validate_windows)
        self.window_table.itemSelectionChanged.connect(self.plot_mechanical)
        wl.addWidget(self.window_table)
        buttons = QHBoxLayout()
        add = QPushButton("Add window"); add.clicked.connect(self.add_window)
        remove = QPushButton("Remove selected"); remove.clicked.connect(self.remove_window)
        choose = QPushButton("Set folder"); choose.clicked.connect(self.choose_window_folder)
        buttons.addWidget(add); buttons.addWidget(remove); buttons.addWidget(choose)
        wl.addLayout(buttons)
        self.window_status = QLabel("Add the row and chunk range for each recording interval.")
        wl.addWidget(self.window_status)
        sl.addWidget(windows_group, 2)
        splitter.addWidget(side)
        panel = QWidget(); pl = QVBoxLayout(panel)
        self.mech_fig, self.mech_ax, self.mech_canvas = self.make_canvas()
        pl.addWidget(NavigationToolbar(self.mech_canvas, panel))
        pl.addWidget(self.mech_canvas)
        splitter.addWidget(panel); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "1 — Mechanical data & synchronization")

    def build_ae_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4); layout.setSpacing(3)
        top = QHBoxLayout()
        btn = QPushButton("Set folder for this window"); btn.clicked.connect(self.open_data)
        self.ae_label = QLabel("No folder assigned to this window")
        self.ae_rate = QDoubleSpinBox(); self.ae_rate.setRange(.001, 1000)
        self.ae_rate.setDecimals(4); self.ae_rate.setValue(3.125); self.ae_rate.setSuffix(" MHz")
        self.dtype_box = QComboBox(); self.dtype_box.addItems(["float32", "float16", "float64"])
        self.dtype_box.setCurrentText("float16")
        self.auto_format = QCheckBox("Auto detect")
        self.auto_format.setChecked(True)
        self.auto_format.stateChanged.connect(self.on_auto_format_changed)
        self.ae_rate.valueChanged.connect(self.on_format_override_changed)
        self.dtype_box.currentIndexChanged.connect(self.on_format_override_changed)
        self.ae_rate.setEnabled(False)
        self.dtype_box.setEnabled(False)
        top.addWidget(btn); top.addWidget(self.ae_label, 1)
        top.addWidget(self.auto_format)
        top.addWidget(QLabel("AE sampling:")); top.addWidget(self.ae_rate)
        top.addWidget(QLabel("Binary type:")); top.addWidget(self.dtype_box)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget(); controls.setMinimumWidth(300); controls.setMaximumWidth(360)
        c = QVBoxLayout(controls)
        c.setContentsMargins(3, 3, 3, 3); c.setSpacing(4)
        nav = QGroupBox("Recording and channel")
        nf = QFormLayout(nav)
        self.recording_box = QComboBox(); self.channel_box = QComboBox()
        self.recording_box.setMinimumWidth(230)
        self.recording_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.recording_box.setMinimumContentsLength(24)
        self.chunk_box = QComboBox()
        self.mech_overlay_box=QComboBox(); self.mech_overlay_box.addItem("None")
        self.mech_overlay_box.currentIndexChanged.connect(self.redraw_overview)
        self.recording_box.currentIndexChanged.connect(self.set_recording)
        self.chunk_box.currentIndexChanged.connect(self.on_chunk_changed)
        self.channel_box.currentIndexChanged.connect(self.on_channel_changed)
        nf.addRow("Recording window", self.recording_box)
        nf.addRow("Chunk to analyze", self.chunk_box)
        nf.addRow("Channel to analyze", self.channel_box)
        nf.addRow("Mechanical overlay", self.mech_overlay_box)
        self.recording_details = QLabel("No recording selected")
        self.recording_details.setWordWrap(True)
        self.recording_details.setStyleSheet("color:#b9c6cc; font-weight:normal;")
        nf.addRow(self.recording_details)
        show = QPushButton("Refresh complete overview"); show.clicked.connect(self.show_overview)
        nf.addRow(show); c.addWidget(nav)

        detect = QGroupBox("Single-channel detection")
        df = QFormLayout(detect)
        self.params = {}
        specs = [
            ("sta_us", "STA (µs)", 0.1, 50000, 20, 1),
            ("lta_us", "LTA (µs)", 0.2, 500000, 200, 1),
            ("trigger", "Trigger ratio", 1.01, 100, 3.5, 2),
            ("detrigger", "Detrigger ratio", 1, 100, 1.5, 2),
            ("min_amplitude", "Min. peak-to-peak (V)", 0, 100, .01, 4),
            ("lowcut_khz", "Band-pass low (kHz)", 0.1, 2000, 50, 1),
            ("highcut_khz", "Band-pass high (kHz)", 0.2, 3000, 1000, 1),
        ]
        for key, label, lo, hi, value, decimals in specs:
            w = QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(decimals)
            w.setSingleStep(.1 if decimals == 2 else .001 if decimals == 4 else 1)
            w.setValue(value); w.valueChanged.connect(self.schedule_preview)
            df.addRow(label, w); self.params[key] = w
        self.bandpass_box = QCheckBox("Apply band-pass filter")
        self.bandpass_box.setChecked(True)
        self.bandpass_box.stateChanged.connect(self.schedule_preview)
        df.addRow(self.bandpass_box)
        c.addWidget(detect)
        c.addWidget(QLabel("The top plot shows every chunk in the recording window.\n"
                           "Choose a chunk above, or click its colored segment.\n"
                           "Zoom/pan stays synchronized in the three analysis plots."))
        self.show_rejected = QCheckBox("Show rejected candidates")
        self.show_rejected.setChecked(True); self.show_rejected.stateChanged.connect(self.schedule_preview)
        c.addWidget(self.show_rejected)
        self.full_resolution_button = QPushButton("Open selected chunk — all samples")
        self.full_resolution_button.clicked.connect(self.open_full_resolution_plot)
        c.addWidget(self.full_resolution_button)
        self.run_button = QPushButton("Detect selected chunk and append")
        self.run_button.clicked.connect(self.run_detection); c.addWidget(self.run_button)
        self.run_all_button = QPushButton("Detect ALL windows, chunks and channels")
        self.run_all_button.clicked.connect(self.run_all_detection)
        c.addWidget(self.run_all_button)
        self.cancel_button = QPushButton("Cancel detection")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_detection); c.addWidget(self.cancel_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0); c.addWidget(self.progress_bar)
        self.export_button = QPushButton("Export channel catalog CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_catalog); c.addWidget(self.export_button)
        self.result_label = QLabel("No detections yet")
        self.result_label.setWordWrap(True)
        c.addWidget(self.result_label)
        c.addStretch()
        splitter.addWidget(controls)

        plots = QWidget(); pp = QVBoxLayout(plots)
        pp.setContentsMargins(0, 0, 0, 0); pp.setSpacing(2)
        self.fig, axes, self.canvas = self.make_canvas(4, 9)
        self.ax_over, self.ax_wave, self.ax_env, self.ax_ratio = axes
        self.ax_mech_overlay=self.ax_over.twinx()
        self.ax_over.set_zorder(self.ax_mech_overlay.get_zorder()+1)
        self.ax_over.patch.set_visible(False)
        self.ax_over.set_title("Selected recording: click or drag to inspect one second")
        self.ax_wave.set_ylabel("AE waveform (V)")
        self.ax_env.set_ylabel("|signal| / STA / LTA")
        self.ax_ratio.set_ylabel("STA / LTA"); self.ax_ratio.set_xlabel("Experiment time (s)")
        self.canvas.mpl_connect('button_press_event', self.on_overview_click)
        self.canvas.mpl_connect('button_release_event', self.on_analysis_zoom_finished)
        pp.addWidget(NavigationToolbar(self.canvas, plots)); pp.addWidget(self.canvas, 1)
        splitter.addWidget(plots); splitter.setStretchFactor(1, 1)
        splitter.setSizes([325, 1395])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "2 — AE waveform & STA/LTA")
        self.preview_timer = QTimer(self); self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.show_preview)

    def build_catalog_tab(self):
        page=QWidget(); layout=QVBoxLayout(page)
        layout.addWidget(QLabel("Single-channel candidates. Green/accepted status is decided by the amplitude threshold. "
                                "Double-click a row to inspect it in the AE tab."))
        self.catalog_table=QTableWidget(0, 7)
        self.catalog_table.setHorizontalHeaderLabels(
            ["Window", "Accepted", "Channel", "Chunk", "Onset (s)",
             "Duration (µs)", "Peak-to-peak (V)"])
        self.catalog_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.catalog_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.catalog_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.catalog_table.cellDoubleClicked.connect(self.jump_to_candidate)
        layout.addWidget(self.catalog_table)
        self.tabs.addTab(page,"3 — Channel catalog")

    def open_mechanical(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mechanical data", "",
            "Mechanical data (*.npy *.npz *.txt *.tsv *.csv);;All files (*)")
        if not path: return
        try:
            data = load_mechanical(path)
            self.mech = data.values
            self.mech_times = data.time_s
            self.mech_label.setText(f"{Path(path).name} · {len(self.mech):,} rows × {self.mech.shape[1]} columns")
            self.names_edit.setText(", ".join(data.names))
            self.mech_rate.setEnabled(self.mech_times is None)
            if self.mech_times is not None and len(self.mech_times)>1:
                self.mech_rate.setValue(min(self.mech_rate.maximum(),
                    1/float(np.median(np.diff(self.mech_times)))))
            self.reset_columns()
            self.validate_windows()
            notices = getattr(data, "notices", [])
            if notices:
                QMessageBox.information(
                    self,
                    "Mechanical file loaded with warnings",
                    "The file was loaded successfully.\n\n" + "\n".join(notices),
                )
                self.statusBar().showMessage(notices[0], 12000)
        except Exception as exc: self.err(exc)

    def names(self):
        if self.mech is None: return []
        names = [s.strip() for s in self.names_edit.text().split(',')]
        if len(names) != self.mech.shape[1] or not all(names):
            raise ValueError(f"Provide exactly {self.mech.shape[1]} nonempty column names")
        return names

    def reset_columns(self):
        if self.mech is None: return
        try: names = self.names()
        except Exception as exc: self.err(exc); return
        self.x_box.blockSignals(True); self.y_list.blockSignals(True)
        self.x_box.clear(); self.x_box.addItems(["Time (s)", "Row"] + names)
        self.y_list.clear(); self.y_list.addItems(names)
        self.y_list.item(0).setSelected(True)
        if hasattr(self,'mech_overlay_box'):
            self.mech_overlay_box.blockSignals(True)
            self.mech_overlay_box.clear(); self.mech_overlay_box.addItems(["None"]+names)
            self.mech_overlay_box.blockSignals(False)
        self.x_box.blockSignals(False); self.y_list.blockSignals(False)
        self.plot_mechanical()

    def add_window(self):
        r = self.window_table.rowCount()
        self.window_table.blockSignals(True)
        self.window_table.insertRow(r)
        for col, value in enumerate((0, 0, 0, "")):
            self.window_table.setItem(r, col, QTableWidgetItem(str(value)))
        self.window_table.blockSignals(False)
        self.validate_windows()

    def choose_window_folder(self):
        """Assign an independent waveform directory to the selected window."""
        row = self.window_table.currentRow()
        if row < 0:
            row = self.recording_box.currentIndex()
        if row < 0 or row >= self.window_table.rowCount():
            self.err("Add or select a recording window first")
            return
        current_item = self.window_table.item(row, 3)
        start = current_item.text() if current_item and current_item.text() else ""
        folder = QFileDialog.getExistingDirectory(
            self, f"Choose waveform folder for window {row + 1}", start
        )
        if not folder:
            return
        self.window_table.setItem(row, 3, QTableWidgetItem(folder))
        self.validate_windows()

    def remove_window(self):
        rows = sorted({i.row() for i in self.window_table.selectedIndexes()}, reverse=True)
        for r in rows: self.window_table.removeRow(r)
        self.validate_windows()

    def index_waveform_source(self, folder_text):
        """Index one window directory and cache its detected file format."""
        text = str(folder_text).strip()
        if not text:
            return None
        path = Path(text).expanduser().resolve()
        key = str(path)
        if key in self._source_cache:
            return self._source_cache[key]
        chunks, ignored_empty = available_chunks(
            path, return_ignored_empty=True
        )
        if not chunks:
            raise ValueError(f"No ae_chunkN directories found in {path}")
        detected = None
        detection_error = None
        try:
            detected = detect_waveform_format(chunks)
        except Exception as exc:
            detection_error = str(exc)
        source = {
            "path": path,
            "chunk_map": chunks,
            "detected": detected,
            "detection_error": detection_error,
            "ignored_empty": ignored_empty,
        }
        self._source_cache[key] = source
        return source

    def validate_windows(self):
        try:
            windows = []
            sources = []
            for r in range(self.window_table.rowCount()):
                vals = [numeric(self.window_table.item(r, c).text(), f"row {r+1}, column {c+1}", True)
                        for c in range(3)]
                if min(vals) < 0 or vals[1] > vals[2]:
                    raise ValueError(f"Invalid range on recording window {r+1}")
                if self.mech is not None and vals[0]>=len(self.mech):
                    raise ValueError(f"Onset row of window {r+1} is outside mechanical data")
                anchor = float(self.mech_times[vals[0]]) if self.mech_times is not None else None
                w = RecordingWindow(*vals, onset_time_s=anchor)
                end_time = w.anchor_time(self.mech_fs())+(w.last_chunk-w.first_chunk+1)
                last_time = (float(self.mech_times[-1]) if self.mech_times is not None
                             else (len(self.mech)-1)/self.mech_fs()) if self.mech is not None else None
                if last_time is not None and end_time > last_time+1:
                    raise ValueError(f"Window {r+1} extends beyond the mechanical record")
                windows.append(w)
                folder_item = self.window_table.item(r, 3)
                folder_text = folder_item.text() if folder_item is not None else ""
                sources.append(self.index_waveform_source(folder_text))
            self.windows = windows
            self.window_sources = sources
            assigned = sum(source is not None for source in sources)
            self.window_status.setText(
                f"{len(windows)} windows defined · {assigned} waveform folders assigned"
            )
            self.window_status.setStyleSheet("color:#9fe0b7")
            old = self.recording_box.currentIndex()
            self.recording_box.blockSignals(True); self.recording_box.clear()
            for i, w in enumerate(windows):
                short_label = f"Window {i+1}: chunks {w.first_chunk}–{w.last_chunk}"
                source_name = sources[i]["path"].name if sources[i] is not None else "no folder"
                full_label = (f"Window {i+1} · onset row {w.onset_row} · "
                              f"chunks {w.first_chunk}–{w.last_chunk} · "
                              f"{w.last_chunk-w.first_chunk+1} s · {source_name}")
                self.recording_box.addItem(short_label)
                self.recording_box.setItemData(i, full_label, Qt.ItemDataRole.ToolTipRole)
            self.recording_box.setCurrentIndex(min(old if old >= 0 else 0, len(windows)-1))
            self.recording_box.blockSignals(False)
            self.set_recording()
        except Exception as exc:
            self.windows = []
            self.window_sources = []
            self.recording_box.clear()
            self.recording_details.setText("No valid recording selected")
            self.window_status.setText(str(exc)); self.window_status.setStyleSheet("color:#ffac99")
        self.plot_mechanical()

    def plot_mechanical(self):
        if not hasattr(self, 'mech_ax'): return
        ax = self.mech_ax; ax.clear()
        if self.mech is None:
            ax.set_title("Load mechanical data to explore the experiment")
        else:
            try:
                names = self.names(); ix = self.x_box.currentIndex()
                step = max(1, int(np.ceil(len(self.mech) / 120_000)))
                rows = np.arange(0, len(self.mech), step)
                x = (self.mech_times[::step] if self.mech_times is not None else
                     rows/self.mech_fs()) if ix <= 0 else rows if ix == 1 else self.mech[::step, ix-2]
                selected = self.y_list.selectedIndexes()
                for item in selected:
                    ax.plot(x, self.mech[::step, item.row()], lw=.8, label=names[item.row()])
                if ix == 0:
                    for k, w in enumerate(self.windows):
                        t0 = w.anchor_time(self.mech_fs())
                        t1 = t0 + w.last_chunk-w.first_chunk+1
                        ax.axvspan(t0, t1, alpha=.14, color="#12b89e")
                        ax.axvline(t0, alpha=.7, color="#0d8272", lw=1)
                if selected: ax.legend(loc="best")
                ax.set_xlabel(self.x_box.currentText()); ax.grid(alpha=.25)
            except Exception: pass
        self.mech_canvas.draw_idle()

    def open_data(self):
        row = self.recording_box.currentIndex()
        if row < 0 or row >= self.window_table.rowCount():
            self.err("Add and select a recording window first")
            return
        current_item = self.window_table.item(row, 3)
        start = current_item.text() if current_item and current_item.text() else ""
        folder = QFileDialog.getExistingDirectory(
            self, f"Choose waveform folder for window {row + 1}", start
        )
        if not folder:
            return
        self.window_table.setItem(row, 3, QTableWidgetItem(folder))
        self.validate_windows()
        self.statusBar().showMessage(
            f"Waveform folder assigned to window {row + 1}"
        )

    def set_recording(self):
        i = self.recording_box.currentIndex()
        if i < 0 or i >= len(self.windows): return
        w = self.windows[i]
        source = self.window_sources[i] if i < len(self.window_sources) else None
        self.data_dir = source["path"] if source is not None else None
        self.chunk_map = source["chunk_map"] if source is not None else {}
        duration = w.last_chunk - w.first_chunk + 1
        self.recording_details.setText(
            f"Onset row {w.onset_row} · {duration} s total\n"
            f"Chunks {w.first_chunk}–{w.last_chunk}\n"
            f"Folder: {self.data_dir if self.data_dir is not None else 'not assigned'}")
        if source is None:
            self.ae_label.setText("No folder assigned to this window")
        else:
            self.ae_label.setText(f"{self.data_dir.name} · {len(self.chunk_map)} chunks")
            self.apply_detected_format(source)

        previous_channel = self.channel_box.currentText()
        channels = sorted({ch for item in self.chunk_map.values() for ch in item})
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems([str(ch) for ch in channels])
        channel_index = self.channel_box.findText(previous_channel)
        if channel_index >= 0:
            self.channel_box.setCurrentIndex(channel_index)
        self.channel_box.blockSignals(False)
        previous = self.chunk_box.currentData()
        self.chunk_box.blockSignals(True)
        self.chunk_box.clear()
        for chunk in range(w.first_chunk, w.last_chunk+1):
            suffix = "" if chunk in self.chunk_map else " (not found)"
            self.chunk_box.addItem(f"Chunk {chunk}{suffix}", chunk)
        target = self.chunk_box.findData(previous)
        self.chunk_box.setCurrentIndex(target if target >= 0 else 0)
        self.chunk_box.blockSignals(False)
        self.recording_box.setToolTip(
            self.recording_box.itemData(i, Qt.ItemDataRole.ToolTipRole) or
            self.recording_box.currentText())
        if self.data_dir is not None and self.channel_box.count():
            QTimer.singleShot(0, self.show_overview)

    def on_channel_changed(self, _index):
        if self.data_dir is not None and self.windows and self.channel_box.count():
            QTimer.singleShot(0, self.show_overview)

    def on_chunk_changed(self, _index):
        if self.overview_data is not None:
            self.redraw_overview()
            self.show_selected_chunk(reset_view=True)

    def selected(self):
        if self.mech is None or self.data_dir is None or not self.windows:
            raise ValueError("Load mechanical and acoustic files, then define a recording window")
        i = self.recording_box.currentIndex()
        if not 0 <= i < len(self.windows): raise ValueError("Select a recording window")
        w = self.windows[i]
        chunk = self.chunk_box.currentData()
        if chunk is None:
            raise ValueError("Select a chunk to analyze")
        start = end = int(chunk)
        ch = int(self.channel_box.currentText())
        missing = [k for k in range(start, end+1) if ch not in self.chunk_map.get(k, {})]
        if missing: raise ValueError(f"Missing channel {ch} in chunks: {missing[:8]}"
                                     + (" ..." if len(missing)>8 else ""))
        return w, start, end, ch

    def config(self):
        values = {key: widget.value() for key, widget in self.params.items()}
        config = DetectorConfig(
            **values,
            highpass_khz=0.0,
            bandpass_enabled=self.bandpass_box.isChecked(),
        )
        config.validate(self.ae_fs())
        return config

    def show_overview(self):
        try:
            w, selected_chunk, _, ch = self.selected()
            window_index = self.recording_box.currentIndex()
            dtype, fs = self.format_for_window(window_index)
            start, end = w.first_chunk, w.last_chunk
            missing = [chunk for chunk in range(start, end+1)
                       if ch not in self.chunk_map.get(chunk, {})]
            if missing:
                raise ValueError(
                    f"Channel {ch} is missing from chunks {missing[:8]}"
                    + (" ..." if len(missing)>8 else ""))
            segments = []
            for chunk in range(start, end+1):
                raw = read_chunk(self.chunk_map[chunk][ch], dtype, fs)
                t0 = w.mechanical_time(chunk, 0, fs, self.mech_fs())
                x, y = peak_preserving_plot(raw, t0, 1/fs, 1600)
                segments.append((chunk, x, y))
            self.overview_data = (w, start, end, ch)
            self.overview_series = segments
            self.redraw_overview(preserve_view=False)
            self.show_selected_chunk(reset_view=True)
        except Exception as exc: self.err(exc)

    def redraw_overview(self, *_, preserve_view=True):
        if not self.overview_data or not hasattr(self,'overview_series'): return
        w,start,end,ch=self.overview_data
        try:
            _dtype, current_fs = self.format_for_window(self.recording_box.currentIndex())
        except Exception:
            current_fs = self.ae_fs()
        previous_xlim=self.ax_over.get_xlim() if preserve_view else None
        ax=self.ax_over; ax.clear(); self.ax_mech_overlay.clear()
        selected = self.chunk_box.currentData()
        n = max(1, len(self.overview_series)-1)
        for i, (chunk, x, y) in enumerate(self.overview_series):
            color = colormaps["turbo"](i/n)
            active = chunk == selected
            ax.plot(x, y, lw=1.0 if active else .55, color=color,
                    alpha=1.0 if active else .78, rasterized=True)
            if active:
                ax.axvspan(x[0], x[-1], color=color, alpha=.10)
            ax.text((x[0]+x[-1])/2, .96, str(chunk),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color=color, fontweight="bold" if active else "normal")
        ax.set_ylabel(f"AE channel {ch} (V)")
        ax.set_title(f"Complete recording window: chunks {start}–{end} · "
                     f"selected chunk {selected}")
        ax.grid(alpha=.25)
        col=self.mech_overlay_box.currentIndex()-1
        self.ax_mech_overlay.set_visible(col >= 0)
        if self.mech is not None and col>=0:
            t0=w.mechanical_time(start,0,current_fs,self.mech_fs())
            t1=w.mechanical_time(end+1,0,current_fs,self.mech_fs())
            if self.mech_times is not None:
                lo=int(np.searchsorted(self.mech_times,t0,side="left"))
                hi=int(np.searchsorted(self.mech_times,t1,side="right"))
            else:
                lo=max(0,int(t0*self.mech_fs()))
                hi=min(len(self.mech),int(t1*self.mech_fs())+1)
            if hi>lo:
                stride=max(1,int(np.ceil((hi-lo)/5000)))
                rows=np.arange(lo,hi,stride)
                t=self.mech_times[rows] if self.mech_times is not None else rows/self.mech_fs()
                self.ax_mech_overlay.plot(t,self.mech[rows,col],
                                          color="#d2703b",lw=1.25)
                self.ax_mech_overlay.set_ylabel(self.names()[col],color="#bd643a")
        if previous_xlim is not None:
            self.ax_over.set_xlim(previous_xlim)
        else:
            self.ax_over.set_xlim(
                w.mechanical_time(start,0,current_fs,self.mech_fs()),
                w.mechanical_time(end+1,0,current_fs,self.mech_fs()))
        self.canvas.draw_idle()

    def on_overview_click(self, event):
        if event.inaxes is self.ax_over and event.xdata is not None and self.canvas.toolbar.mode == "":
            w, _, _, _ = self.overview_data
            chunk = w.first_chunk + int(np.floor(event.xdata-w.anchor_time(self.mech_fs())))
            chunk = min(max(chunk, w.first_chunk), w.last_chunk)
            index = self.chunk_box.findData(chunk)
            if index >= 0:
                self.chunk_box.setCurrentIndex(index)

    def on_analysis_zoom_finished(self, event):
        """Reload original samples after a zoom or pan in the analysis panels."""
        if event.inaxes not in (self.ax_wave, self.ax_env, self.ax_ratio):
            return
        if self.canvas.toolbar.mode == "":
            return
        try:
            w, chunk, _, _ = self.selected()
            _dtype, fs = self.format_for_window(self.recording_box.currentIndex())
            chunk_left = w.mechanical_time(chunk, 0, fs, self.mech_fs())
            chunk_right = chunk_left + 1.0
            left, right = sorted(event.inaxes.get_xlim())
            left = max(chunk_left, left)
            right = min(chunk_right, right)
            if right <= left:
                return
            self.preview_range = (left, right)
            self._reset_view_on_next_preview = True
            self.schedule_preview()
        except Exception as exc:
            self.statusBar().showMessage(f"Zoom refresh: {exc}")

    def open_full_resolution_plot(self):
        """Open every original sample from the selected chunk in a new window."""
        try:
            w, chunk, _, channel = self.selected()
            window_index = self.recording_box.currentIndex()
            dtype, fs = self.format_for_window(window_index)
            raw = read_chunk(self.chunk_map[chunk][channel], dtype, fs)
            voltage = np.asarray(raw, dtype=np.float32)
            config = self.config()
            if config.bandpass_enabled:
                voltage = filter_waveform(voltage, fs, config)
            time0 = w.mechanical_time(chunk, 0, fs, self.mech_fs())
            time = time0 + np.arange(len(voltage), dtype=np.float64) / fs

            window = QMainWindow(self)
            suffix = "band-pass filtered" if config.bandpass_enabled else "raw"
            window.setWindowTitle(
                f"Full-resolution AE · window {window_index + 1} · "
                f"chunk {chunk} · channel {channel}"
            )
            window.resize(1300, 650)
            panel = QWidget()
            layout = QVBoxLayout(panel)
            figure = Figure(figsize=(12, 5), layout="constrained")
            axis = figure.add_subplot(111)
            canvas = FigureCanvas(figure)
            layout.addWidget(NavigationToolbar(canvas, panel))
            layout.addWidget(canvas)
            window.setCentralWidget(panel)

            axis.plot(time, voltage, lw=.45, color="#26759b")
            axis.set_title(
                f"All {len(voltage):,} samples · {dtype} · {fs/1e6:g} MHz · {suffix}"
            )
            axis.set_xlabel("Mechanical experiment time (s)")
            axis.set_ylabel("AE (V)")
            axis.grid(alpha=.25)
            window.show()
            self.full_resolution_windows.append(window)
        except Exception as exc:
            self.err(exc)

    def show_selected_chunk(self, reset_view=False):
        try:
            w, chunk, _, _ = self.selected()
            _dtype, fs = self.format_for_window(self.recording_box.currentIndex())
        except Exception:
            return
        begin = w.mechanical_time(chunk, 0, fs, self.mech_fs())
        self.preview_range = (begin, begin+1.0)
        if reset_view:
            self._reset_view_on_next_preview = True
        self.schedule_preview()

    def schedule_preview(self, *_):
        if getattr(self, 'preview_range', None) is not None:
            self.preview_timer.start(180)

    def show_preview(self):
        try:
            w, start, end, ch = self.selected()
            window_index = self.recording_box.currentIndex()
            dtype, fs = self.format_for_window(window_index)
            begin, finish = self.preview_range
            # A new overview selection sets a fresh view. A parameter change
            # keeps the student's current zoom/pan limits on every shared axis.
            display_xlim = ((begin, finish) if self._reset_view_on_next_preview
                            else self.ax_ratio.get_xlim())
            mech_fs = self.mech_fs()
            base = w.anchor_time(mech_fs)
            # Read only the close-up plus a short lead-in to establish the LTA.
            lead = max(.005, self.params['lta_us'].value()*1e-6*2)
            first_sample = max(int((begin-base)*fs) - int(lead*fs), (start-w.first_chunk)*round(fs))
            last_sample = min(int(np.ceil((finish-base)*fs)), (end-w.first_chunk+1)*round(fs))
            parts=[]
            n_per = round(fs)
            global_index=first_sample
            while global_index < last_sample:
                chunk = w.first_chunk + global_index//n_per
                local = global_index%n_per
                take = min(last_sample-global_index, n_per-local)
                raw = read_chunk(self.chunk_map[chunk][ch], dtype, fs)
                parts.append(np.asarray(raw[local:local+take], dtype=np.float32))
                global_index += take
            if not parts: return
            raw = np.concatenate(parts)
            picker = AEPicker(fs, self.config(), preview=True)
            filtered, events, sta, lta, ratio = picker.process(raw)
            if picker.close():
                # A candidate reaching the right edge is incomplete and remains unclassified.
                pass
            abs_t = base+first_sample/fs
            keep0 = max(0, int((begin-abs_t)*fs))
            keep1 = min(len(raw), int(np.ceil((finish-abs_t)*fs)))
            if keep1 <= keep0: return
            visible_samples = keep1 - keep0
            if visible_samples <= 250_000:
                # At close zoom, plot every original sample exactly.
                time = abs_t + np.arange(keep0, keep1, dtype=float) / fs
                wave = filtered[keep0:keep1]
            else:
                time, wave = peak_preserving_plot(
                    filtered[keep0:keep1], abs_t + keep0/fs, 1/fs, 5000
                )
            self.ax_wave.clear(); self.ax_env.clear(); self.ax_ratio.clear()
            self.ax_wave.plot(time, wave, lw=.6, color="#276991")
            time_a, abs_a = peak_preserving_plot(
                np.abs(filtered[keep0:keep1]), abs_t + keep0/fs, 1/fs, 3500
            )
            self.ax_env.plot(time_a, abs_a, lw=.5, color="#adb9bf", label="|signal|")
            for arr, color, label in ((sta, '#d0583f', 'STA'),(lta,'#3281c2','LTA (frozen in AE)')):
                tx, yy = peak_preserving_plot(arr[keep0:keep1],abs_t+keep0/fs,1/fs,3500)
                self.ax_env.plot(tx, yy, lw=.85, color=color, label=label)
            tx, rr = peak_preserving_plot(ratio[keep0:keep1],abs_t+keep0/fs,1/fs,3500)
            self.ax_ratio.plot(tx, rr, lw=.8, color="#1b1b1b")
            self.ax_ratio.axhline(self.config().trigger,color="#cb5745",ls="--",label="Trigger")
            self.ax_ratio.axhline(self.config().detrigger,color="#4090cc",ls="--",label="Detrigger")
            visible=0
            for a,b,amp,accepted in events:
                x0, x1=abs_t+a/fs,abs_t+b/fs
                if x1 < begin or x0 > finish or (not accepted and not self.show_rejected.isChecked()):
                    continue
                color="#21946b" if accepted else "#b87e3b"
                for ax in (self.ax_wave,self.ax_ratio):
                    ax.axvspan(max(begin,x0),min(finish,x1),alpha=.19,color=color)
                visible+=1
            resolution_text = ("all original samples" if visible_samples <= 250_000
                               else "peak-preserving overview")
            self.ax_wave.set_title(
                f"Chunk {start} · channel {ch} · {begin:.6f}–{finish:.6f} s · "
                f"{resolution_text} · {visible} candidates"
            )
            self.ax_wave.set_ylabel("Filtered AE (V)")
            self.ax_env.set_ylabel("|AE|, STA, LTA")
            self.ax_ratio.set_ylabel("STA / LTA")
            self.ax_ratio.set_xlabel("Mechanical experiment time (s)")
            for ax in (self.ax_wave,self.ax_env,self.ax_ratio):
                ax.grid(alpha=.22)
            self.ax_ratio.set_xlim(display_xlim)
            self._reset_view_on_next_preview = False
            self.ax_env.legend(loc="upper right",fontsize=8)
            self.ax_ratio.legend(loc="upper right",fontsize=8)
            self.canvas.draw_idle()
        except Exception as exc:
            self.statusBar().showMessage(f"Preview: {exc}")

    def run_detection(self):
        try:
            w, start, _end, channel = self.selected()
            window_index = self.recording_box.currentIndex()
            task = self.make_analysis_task(window_index, [start], channel)
            self.start_analysis([task], replace_catalog=False,
                                description=f"window {window_index + 1}, "
                                            f"chunk {start}, channel {channel}")
        except Exception as exc:
            self.run_button.setEnabled(True)
            self.run_all_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.export_button.setEnabled(self.catalog_ready)
            self.err(exc)

    @staticmethod
    def consecutive_groups(values):
        """Split sorted integers into consecutive groups."""
        groups = []
        for value in sorted(values):
            if not groups or value != groups[-1][-1] + 1:
                groups.append([value])
            else:
                groups[-1].append(value)
        return groups

    def make_analysis_task(self, window_index, chunks, channel):
        """Build one continuous window/channel analysis task."""
        source = self.window_sources[window_index]
        if source is None:
            raise ValueError(f"Window {window_index + 1} has no waveform folder")
        dtype, fs = self.format_for_window(window_index)
        recording = self.windows[window_index]
        for chunk in chunks:
            if channel not in source["chunk_map"].get(chunk, {}):
                raise ValueError(
                    f"Window {window_index + 1}: channel {channel} is missing "
                    f"from chunk {chunk}"
                )
        return {
            "window_index": window_index,
            "window_name": f"Window {window_index + 1}",
            "recording": recording,
            "chunk_map": source["chunk_map"],
            "chunks": list(chunks),
            "channel": int(channel),
            "dtype": dtype,
            "fs": fs,
            "mech_fs": self.mech_fs(),
            "source_dir": source["path"],
        }

    def run_all_detection(self):
        """Apply the current picker parameters to every loaded waveform."""
        try:
            if not self.windows:
                raise ValueError("Define at least one recording window")
            tasks = []
            for window_index, (recording, source) in enumerate(
                    zip(self.windows, self.window_sources)):
                if source is None:
                    raise ValueError(
                        f"Assign a waveform folder to window {window_index + 1}"
                    )
                requested = range(recording.first_chunk, recording.last_chunk + 1)
                channels = sorted({
                    channel
                    for chunk in requested
                    for channel in source["chunk_map"].get(chunk, {})
                })
                if not channels:
                    raise ValueError(
                        f"Window {window_index + 1} contains no files in its "
                        "configured chunk range"
                    )
                for channel in channels:
                    available = [
                        chunk for chunk in requested
                        if channel in source["chunk_map"].get(chunk, {})
                    ]
                    for group in self.consecutive_groups(available):
                        tasks.append(
                            self.make_analysis_task(window_index, group, channel)
                        )
            self.start_analysis(
                tasks,
                replace_catalog=True,
                description=(f"all loaded data: {len(tasks)} continuous "
                             "window/channel series"),
            )
        except Exception as exc:
            self.run_button.setEnabled(True)
            self.run_all_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.export_button.setEnabled(self.catalog_ready)
            self.err(exc)

    def start_analysis(self, tasks, replace_catalog, description):
        """Start a background picker run for one or many waveform tasks."""
        if not tasks:
            raise ValueError("No waveform files were selected for analysis")
        config = self.config()
        for task in tasks:
            config.validate(task["fs"])
        total_files = sum(len(task["chunks"]) for task in tasks)
        self._pending_replace_catalog = bool(replace_catalog)
        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(0)
        self.result_label.setText(f"Starting {description} · {total_files} files")
        self.worker = AnalysisWorker(tasks, config, self)
        self.worker.progress.connect(self.detection_progress)
        self.worker.finished_catalog.connect(self.finish_detection)
        self.worker.failed.connect(self.detection_failed)
        self.worker.cancelled.connect(self.detection_cancelled)
        self.worker.finished.connect(self.detection_finished)
        self.worker.start()

    def detection_progress(self, done, total, candidates):
        self.progress_bar.setValue(done)
        message=f"Processed {done}/{total} chunks · {candidates:,} candidates"
        self.result_label.setText(message)
        self.statusBar().showMessage(message)

    def cancel_detection(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.result_label.setText("Stopping after the current chunk...")

    def detection_cancelled(self):
        self.result_label.setText("Detection cancelled; previous catalog retained")
        self.statusBar().showMessage("Detection cancelled")

    def finish_detection(self, rows):
        if self._pending_replace_catalog:
            self.rows = []
            self.catalog_table.setRowCount(0)
        first_table_row = self.catalog_table.rowCount()
        self.rows.extend(rows)
        self.catalog_ready=True
        new_accepted=sum(r['accepted'] for r in rows)
        total_accepted=sum(r['accepted'] for r in self.rows)
        self.result_label.setText(
            f"Appended {len(rows)} candidates ({new_accepted} accepted) · "
            f"catalog total {len(self.rows)} ({total_accepted} accepted)")
        self.catalog_table.setRowCount(first_table_row + len(rows))
        for offset, row in enumerate(rows):
            table_row = first_table_row + offset
            values = (
                str(row.get("window", "")),
                "Yes" if row["accepted"] else "No",
                str(row["channel"]), str(row["chunk"]),
                f"{row['onset_s']:.6f}", f"{row['duration_us']:.2f}",
                f"{row['peak_to_peak']:.5f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    color = "#4bbd83" if row["accepted"] else "#cf9866"
                    item.setForeground(QBrush(QColor(color)))
                self.catalog_table.setItem(table_row, column, item)
        self.statusBar().showMessage("Selected chunk appended to the channel catalog")

    def jump_to_candidate(self, row, _column):
        if row>=len(self.rows): return
        item=self.rows[row]
        rec_index = int(item.get("window_index", -1))
        if rec_index < 0:
            rec_index=next((i for i,w in enumerate(self.windows)
                            if w.first_chunk==item['recording_start_chunk']),-1)
        if rec_index >= 0:
            self.recording_box.setCurrentIndex(rec_index)
        ch_index=self.channel_box.findText(str(item['channel']))
        if ch_index >= 0:
            self.channel_box.setCurrentIndex(ch_index)
        chunk_index=self.chunk_box.findData(item['chunk'])
        if chunk_index >= 0:
            self.chunk_box.setCurrentIndex(chunk_index)
        self.tabs.setCurrentIndex(1)
        self.show_selected_chunk(reset_view=True)

    def detection_failed(self, message):
        self.result_label.setText(f"Detection failed: {message}")
        self.statusBar().showMessage(f"Detection failed: {message}")
        QMessageBox.warning(self,"Detection failed",message)

    def detection_finished(self):
        self.run_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(self.catalog_ready)
        self.worker = None
        self._pending_replace_catalog = False

    def export_catalog(self):
        if not self.catalog_ready:
            self.err("Run detection first")
            return
        path,_=QFileDialog.getSaveFileName(self,"Export AE catalog","ae_channel_catalog.csv","CSV (*.csv)")
        if not path: return
        columns=["window_index","window","window_start_s","window_end_s",
                 "source_dir","recording_start_chunk","channel","chunk",
                 "sample_in_chunk","onset_s","end_s","duration_us",
                 "peak_to_peak","accepted","truncated_at_end"]
        with open(path,"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader(); writer.writerows(self.rows)
        self.statusBar().showMessage(f"Catalog exported: {path}")


if __name__ == '__main__':
    app=QApplication(sys.argv)
    gui=Analyzer(); gui.show()
    sys.exit(app.exec())
