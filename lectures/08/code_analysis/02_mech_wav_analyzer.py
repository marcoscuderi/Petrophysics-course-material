import sys
import os
from io import StringIO
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# PyQt UI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QComboBox, QSplitter, QDialog,
    QGroupBox, QTabWidget, QSlider
)
from PyQt6.QtCore import Qt

# Matplotlib Embedding Imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


# ==========================================
# 1. PARSERS & CAUSAL DETECTION LOGIC
# ==========================================
def read_bscan_tsv(file_path):
    """Parses custom B-scan TSV file format with header metadata."""
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            
            if '\t' in line:
                if line.startswith('#'):
                    line = line.lstrip('#').strip() + '\n'
                data_lines.append(line)
            else:
                clean_line = line.lstrip('#').strip()
                if ':' in clean_line:
                    key, val = clean_line.split(':', 1)
                    metadata[key.strip()] = val.strip()

    waveforms_df = pd.read_csv(StringIO(''.join(data_lines)), sep='\t')
    return metadata, waveforms_df


def find_5v_falling_peaks(df):
    """Detects 5V plateau falling-edge peaks from sync channel."""
    sync_vals = df['sync'].values
    is_high = sync_vals > 2.0
    
    diff_high = np.diff(is_high.astype(int))
    starts = np.where(diff_high == 1)[0] + 1
    if is_high[0]:
        starts = np.insert(starts, 0, 0)
        
    ends = np.where(diff_high == -1)[0]
    if is_high[-1]:
        ends = np.append(ends, len(is_high) - 1)
        
    falling_peaks_5v = []
    for s, e in zip(starts, ends):
        cluster_indices = np.arange(s, e + 1)
        cluster_sync = sync_vals[cluster_indices]
        max_val = np.max(cluster_sync)
        peak_idx_relative = np.where(cluster_sync >= max_val - 0.05)[0][-1]
        falling_peaks_5v.append(cluster_indices[peak_idx_relative])
        
    return np.array(falling_peaks_5v)


def run_sta_lta_picker(signal, sampling_rate_hz, sta_us=0.5, lta_us=15.0, p_thresh=3.5, s_thresh=2.5, min_s_delay_us=2.0):
    """
    Strictly Causal STA/LTA Automatic Arrival Time Picker.
    Signal MUST be pre-filtered prior to executing this function.
    """
    dt = 1.0 / sampling_rate_hz
    dt_us = dt * 1e6
    num_samples = len(signal)
    time_us = np.arange(num_samples) * dt_us
    
    n_sta = max(1, int((sta_us * 1e-6) / dt))
    n_lta = max(1, int((lta_us * 1e-6) / dt))
    
    # Characteristic Function (Absolute Amplitude)
    cf = np.abs(signal)
    
    # Fast Causal Moving Average via Cumulative Sum
    cs = np.pad(np.cumsum(cf), (1, 0))
    
    def causal_ma(n_win):
        idx = np.arange(1, num_samples + 1)
        start_idx = np.maximum(0, idx - n_win)
        counts = idx - start_idx
        return (cs[idx] - cs[start_idx]) / counts

    sta = causal_ma(n_sta)
    lta = causal_ma(n_lta)
    
    lta_safe = np.where(lta == 0, 1e-12, lta)
    ratio = sta / lta_safe
    
    p_pick_us = None
    s_pick_us = None
    
    p_triggers = np.where(ratio >= p_thresh)[0]
    if len(p_triggers) > 0:
        p_idx = p_triggers[0]
        p_pick_us = time_us[p_idx]
        
        min_s_idx = p_idx + int((min_s_delay_us * 1e-6) / dt)
        if min_s_idx < num_samples:
            s_triggers = np.where(ratio[min_s_idx:] >= s_thresh)[0]
            if len(s_triggers) > 0:
                s_idx = min_s_idx + s_triggers[0]
                s_pick_us = time_us[s_idx]
                
    return p_pick_us, s_pick_us, sta, lta, ratio


def extract_cc_template(signal, pick_index, half_width_samples):
    """Extract an odd-length, demeaned template around a master arrival."""
    half_width_samples = max(1, int(half_width_samples))
    start = int(pick_index) - half_width_samples
    stop = int(pick_index) + half_width_samples + 1
    if start < 0 or stop > len(signal):
        raise ValueError('The template window extends beyond the waveform limits.')

    template = np.asarray(signal[start:stop], dtype=float)
    template = template - np.mean(template)
    if np.linalg.norm(template) == 0:
        raise ValueError('The selected template has zero amplitude.')
    return template


def normalized_cc_pick(signal, template, reference_index, search_half_samples):
    """Find the template centre with maximum normalized cross-correlation."""
    signal = np.asarray(signal, dtype=float)
    template = np.asarray(template, dtype=float)
    template_half = len(template) // 2
    search_half_samples = max(0, int(search_half_samples))

    first_centre = max(template_half, int(reference_index) - search_half_samples)
    last_centre = min(
        len(signal) - template_half - 1,
        int(reference_index) + search_half_samples
    )
    if first_centre > last_centre:
        return None, np.nan

    template_norm = np.linalg.norm(template)
    best_index = None
    best_coefficient = -np.inf

    for centre in range(first_centre, last_centre + 1):
        segment = signal[centre - template_half:centre + template_half + 1]
        segment = segment - np.mean(segment)
        denominator = template_norm * np.linalg.norm(segment)
        if denominator == 0:
            continue
        coefficient = float(np.dot(template, segment) / denominator)
        if coefficient > best_coefficient:
            best_coefficient = coefficient
            best_index = centre

    return best_index, best_coefficient


# ==========================================
# 2. POPUP BATCH VALIDATION WINDOW
# ==========================================
class STALTAValidationDialog(QDialog):
    """Pop-up window displaying stacked traces with auto-picked arrivals snapped to wave traces."""
    def __init__(self, waveforms_df, sampling_rate_hz, file_picks, file_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"STA/LTA Auto-Picker Inspection - {file_name}")
        self.setGeometry(100, 100, 1000, 800)
        
        num_traces, num_samples = waveforms_df.shape
        dt_us = (1.0 / sampling_rate_hz) * 1e6
        time_us = np.arange(num_samples) * dt_us
        
        layout = QVBoxLayout(self)
        fig, ax = plt.subplots(figsize=(10, max(6, num_traces * 0.4)))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        for i in range(num_traces):
            raw_sig = waveforms_df.iloc[i].values
            sig = parent.apply_signal_processing(raw_sig) if parent else raw_sig - np.mean(raw_sig)
            max_amp = np.max(np.abs(sig)) if np.max(np.abs(sig)) > 0 else 1.0
            norm_sig = (sig / max_amp) * 0.8
            
            offset = i + 1
            trace_y = norm_sig + offset
            ax.plot(time_us, trace_y, color='#2B5C8F', linewidth=0.8)
            
            p_auto = file_picks.get(i, {}).get('P_auto', None)
            s_auto = file_picks.get(i, {}).get('S_auto', None)
            
            # Snap dots onto the actual waveform trace using linear interpolation
            if p_auto is not None:
                p_y = np.interp(p_auto, time_us, trace_y)
                ax.plot(p_auto, p_y, 'ro', markersize=6, label='Auto P' if i == 0 else "")
            if s_auto is not None:
                s_y = np.interp(s_auto, time_us, trace_y)
                ax.plot(s_auto, s_y, 'go', markersize=6, label='Auto S' if i == 0 else "")
                
        ax.set_yticks(range(1, num_traces + 1))
        ax.set_yticklabels([f"Trace {i+1}" for i in range(num_traces)])
        ax.set_xlabel("Time (µs)", fontweight='bold')
        ax.set_title(f"Auto-Picked Arrivals Validation ({num_traces} Pre-Filtered Waveforms)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        canvas.draw()


# ==========================================
# 2B. CROSS-CORRELATION BATCH VALIDATION
# ==========================================
class CrossCorrelationValidationDialog(QDialog):
    """Stacked traces with P/S cross-correlation arrivals and CC values."""
    def __init__(self, waveforms_df, sampling_rate_hz, file_picks, file_name,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Cross-Correlation Inspection - {file_name}")
        self.setGeometry(90, 90, 1050, 820)

        num_traces, num_samples = waveforms_df.shape
        time_us = np.arange(num_samples) * (1.0 / sampling_rate_hz) * 1e6

        layout = QVBoxLayout(self)
        fig, ax = plt.subplots(figsize=(10, max(6, num_traces * 0.4)))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        for trace_idx in range(num_traces):
            raw_signal = waveforms_df.iloc[trace_idx].to_numpy(dtype=float)
            signal = (
                parent.apply_signal_processing(raw_signal)
                if parent is not None
                else raw_signal - np.mean(raw_signal)
            )
            amplitude = np.max(np.abs(signal))
            if not np.isfinite(amplitude) or amplitude == 0:
                amplitude = 1.0
            trace_y = (signal / amplitude) * 0.8 + trace_idx + 1
            ax.plot(time_us, trace_y, color='#2B5C8F', linewidth=0.8)

            picks = file_picks.get(trace_idx, {})
            p_cc = picks.get('P_cc')
            s_cc = picks.get('S_cc')
            if p_cc is not None:
                p_y = np.interp(p_cc, time_us, trace_y)
                ax.plot(p_cc, p_y, 'ro', markersize=6,
                        label='CC P' if trace_idx == 0 else '')
            if s_cc is not None:
                s_y = np.interp(s_cc, time_us, trace_y)
                ax.plot(s_cc, s_y, 'go', markersize=6,
                        label='CC S' if trace_idx == 0 else '')

        label_step = max(1, int(np.ceil(num_traces / 30)))
        tick_indices = np.arange(0, num_traces, label_step)
        if num_traces and tick_indices[-1] != num_traces - 1:
            tick_indices = np.append(tick_indices, num_traces - 1)
        ax.set_yticks(tick_indices + 1)
        ax.set_yticklabels([f"Trace {i + 1}" for i in tick_indices])
        ax.set_xlabel("Time (µs)", fontweight='bold')
        ax.set_title(
            f"Cross-Correlation Arrivals ({num_traces} Pre-Filtered Waveforms)",
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')
        fig.tight_layout()
        canvas.draw()


# ==========================================
# 2C. CLICKABLE ALL-WAVEFORM OVERVIEW
# ==========================================
class WaveformOverviewDialog(QDialog):
    """Stacked waveform overview for choosing the trace used to start analysis."""
    def __init__(self, waveforms_df, sampling_rate_hz, file_name,
                 current_trace_idx=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Waveform Overview - {file_name}")
        self.setGeometry(80, 80, 1100, 850)
        self.selected_trace_idx = None

        self.num_traces, num_samples = waveforms_df.shape
        time_us = np.arange(num_samples) * (1.0 / sampling_rate_hz) * 1e6

        layout = QVBoxLayout(self)
        instruction = QLabel(
            "Click a waveform to close this window and start the analysis from "
            "that trace. Use the toolbar first if you need to zoom or pan."
        )
        instruction.setStyleSheet(
            "font-weight: bold; color: #2B5C8F; padding: 6px;"
        )
        layout.addWidget(instruction)

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        for trace_idx in range(self.num_traces):
            raw_signal = waveforms_df.iloc[trace_idx].to_numpy(dtype=float)
            signal = (
                parent.apply_signal_processing(raw_signal)
                if parent is not None
                else raw_signal - np.nanmean(raw_signal)
            )

            amplitude = np.nanmax(np.abs(signal))
            if not np.isfinite(amplitude) or amplitude == 0:
                amplitude = 1.0
            normalized_signal = (signal / amplitude) * 0.78

            offset = trace_idx + 1
            is_current = trace_idx == current_trace_idx
            self.ax.plot(
                time_us,
                normalized_signal + offset,
                color='#D9792B' if is_current else '#2B5C8F',
                linewidth=1.5 if is_current else 0.75,
                zorder=3 if is_current else 2,
            )

        # Limit the number of labels without omitting any waveform.
        label_step = max(1, int(np.ceil(self.num_traces / 30)))
        tick_indices = np.arange(0, self.num_traces, label_step)
        if self.num_traces and tick_indices[-1] != self.num_traces - 1:
            tick_indices = np.append(tick_indices, self.num_traces - 1)

        self.ax.set_yticks(tick_indices + 1)
        self.ax.set_yticklabels([f"Trace {i + 1}" for i in tick_indices])
        self.ax.set_ylim(0.1, self.num_traces + 0.9)
        self.ax.set_xlabel("Time (µs)", fontweight='bold')
        self.ax.set_ylabel("Waveform trace")
        self.ax.set_title(
            f"All Waveforms — {file_name} ({self.num_traces} traces)\n"
            "Orange = currently active trace",
            fontweight='bold'
        )
        self.ax.grid(True, axis='x', alpha=0.25)
        self.fig.tight_layout()
        self.canvas.mpl_connect('button_press_event', self.select_trace)
        self.canvas.draw()

    def select_trace(self, event):
        if event.inaxes != self.ax or event.ydata is None or self.toolbar.mode != '':
            return

        trace_idx = int(round(event.ydata)) - 1
        if 0 <= trace_idx < self.num_traces:
            self.selected_trace_idx = trace_idx
            self.accept()


# ==========================================
# 3. MAIN APPLICATION GUI
# ==========================================
class WaveformAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sync-Coupled Acoustic Waveform Analyzer")
        self.setGeometry(50, 50, 1350, 950)
        
        # Core Data Structures
        self.mech_df = None
        self.peaks_5v = None
        self.interval_records = []
        self.current_interval_idx = -1
        
        self.current_metadata = {}
        self.current_waveforms_df = None
        self.num_traces = 0
        self.num_samples = 0
        self.current_trace_idx = 0
        self.sampling_rate_hz = 10e6
        
        self.show_theo_windows = False
        self.all_picks = {}
        self.cc_master_trace_idx = None
        self.cc_master_p_us = None
        self.cc_master_s_us = None
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Stylesheet
        self.setStyleSheet("""
            QWidget { background-color: #2D2D2D; color: #FFFFFF; }
            QTabWidget::pane { border: 1px solid #444; background: #2D2D2D; }
            QTabBar::tab { background: #3E3E3E; color: white; padding: 6px 14px; border: 1px solid #555; }
            QTabBar::tab:selected { background: #1E5128; font-weight: bold; }
            QPushButton { background-color: #4A4A4A; color: white; border: 1px solid #666; padding: 4px 10px; border-radius: 3px; }
            QPushButton:hover { background-color: #5A5A5A; }
            QPushButton:checked { background-color: #2E7D32; font-weight: bold; }
            QLineEdit, QComboBox { background-color: #1E1E1E; color: white; border: 1px solid #555; padding: 2px 4px; }
            QGroupBox { border: 1px solid #555; margin-top: 5px; padding-top: 8px; font-weight: bold; color: #88C0D0; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { background: #88C0D0; width: 14px; margin: -4px 0; border-radius: 7px; }
        """)
        
        self.tabs = QTabWidget()
        
        # Build the manual, STA/LTA and cross-correlation analysis tabs.
        tab1_widget = self.build_tab1_manual()
        tab2_widget = self.build_tab2_stalta()
        tab3_widget = self.build_tab3_cross_correlation()
        
        self.tabs.addTab(tab1_widget, "Waveform & Sync Manual Picker")
        self.tabs.addTab(tab2_widget, "STA/LTA Threshold Calibration")
        self.tabs.addTab(tab3_widget, "Cross-Correlation Picker")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tabs)
        
        self.plot_blank_sync()
        self.plot_blank_waveform()
        self.plot_blank_stalta()
        self.plot_blank_cc()

    # ==========================================
    # TAB 1: MANUAL PICKER UI BUILDER
    # ==========================================
    def build_tab1_manual(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top Panel: Sync Signal Plot
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig_sync, self.ax_sync = plt.subplots(figsize=(10, 2.2))
        self.canvas_sync = FigureCanvas(self.fig_sync)
        self.toolbar_sync = NavigationToolbar(self.canvas_sync, self)
        
        top_layout.addWidget(self.toolbar_sync)
        top_layout.addWidget(self.canvas_sync)
        
        # Middle Controls
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        
        row1 = QHBoxLayout()
        self.btn_load_mech = QPushButton("1. Load Mechanical File")
        self.btn_load_mech.clicked.connect(self.load_mechanical_file)
        
        self.btn_select_dir = QPushButton("2. Select Waveforms Folder")
        self.btn_select_dir.clicked.connect(self.associate_waveforms_directory)
        self.btn_select_dir.setEnabled(False)
        
        lbl_interval = QLabel("Interval / File:")
        self.combo_intervals = QComboBox()
        self.combo_intervals.setMinimumWidth(240)
        self.combo_intervals.currentIndexChanged.connect(self.on_interval_changed)

        self.btn_waveform_overview = QPushButton("View All Waveforms")
        self.btn_waveform_overview.setStyleSheet(
            "background-color: #2B5C8F; font-weight: bold;"
        )
        self.btn_waveform_overview.clicked.connect(self.show_waveform_overview)
        self.btn_waveform_overview.setEnabled(False)
        
        row1.addWidget(self.btn_load_mech)
        row1.addWidget(self.btn_select_dir)
        row1.addSpacing(10)
        row1.addWidget(lbl_interval)
        row1.addWidget(self.combo_intervals)
        row1.addWidget(self.btn_waveform_overview)
        row1.addStretch()
        
        # Row 2: Theoretical Velocity Windows
        row2 = QHBoxLayout()
        group_theo = QGroupBox("Theoretical Velocity Windows")
        layout_theo = QHBoxLayout(group_theo)
        
        self.txt_len_mm = QLineEdit("25.0")
        self.txt_len_mm.setFixedWidth(70)
        self.txt_vp = QLineEdit("5000")
        self.txt_vp.setFixedWidth(70)
        self.txt_vs = QLineEdit("3000")
        self.txt_vs.setFixedWidth(70)
        self.txt_tol = QLineEdit("1.0")
        self.txt_tol.setFixedWidth(70)
        
        self.btn_toggle_theo = QPushButton("Show Theo Windows")
        self.btn_toggle_theo.setCheckable(True)
        self.btn_toggle_theo.clicked.connect(self.toggle_theo_windows)
        
        layout_theo.addWidget(QLabel("L (mm):"))
        layout_theo.addWidget(self.txt_len_mm)
        layout_theo.addSpacing(10)
        layout_theo.addWidget(QLabel("Vp (m/s):"))
        layout_theo.addWidget(self.txt_vp)
        layout_theo.addSpacing(10)
        layout_theo.addWidget(QLabel("Vs (m/s):"))
        layout_theo.addWidget(self.txt_vs)
        layout_theo.addSpacing(10)
        layout_theo.addWidget(QLabel("± (µs):"))
        layout_theo.addWidget(self.txt_tol)
        layout_theo.addSpacing(15)
        layout_theo.addWidget(self.btn_toggle_theo)
        layout_theo.addStretch()
        
        row2.addWidget(group_theo)
        
        # Row 3: Manual Controls & Filters
        row3 = QHBoxLayout()
        self.txt_cutoff = QLineEdit("1.0")
        self.txt_cutoff.setFixedWidth(50)
        self.btn_apply_filter = QPushButton("Apply Filter")
        self.btn_apply_filter.clicked.connect(lambda: self.update_all_plots(preserve_view=True))
        
        self.rb_pwave = QRadioButton("P-wave (Red)")
        self.rb_swave = QRadioButton("S-wave (Green)")
        self.rb_pwave.setChecked(True)
        self.rb_pwave.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.rb_swave.setStyleSheet("color: #55FF55; font-weight: bold;")
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.rb_pwave)
        self.mode_group.addButton(self.rb_swave)
        
        self.lbl_picks_readout = QLabel("Manual Picks -> [P: - | S: -]")
        self.lbl_picks_readout.setStyleSheet("color: #88C0D0; font-weight: bold; margin-left: 10px;")
        
        self.btn_prev_tab1 = QPushButton("<- Prev Trace")
        self.btn_prev_tab1.clicked.connect(self.prev_waveform)
        self.btn_next_tab1 = QPushButton("Next Trace ->")
        self.btn_next_tab1.clicked.connect(self.next_waveform)
        self.btn_save = QPushButton("Save Synched CSV")
        self.btn_save.clicked.connect(self.save_synched_picks)
        
        row3.addWidget(QLabel("Filter Cutoff (MHz):"))
        row3.addWidget(self.txt_cutoff)
        row3.addWidget(self.btn_apply_filter)
        row3.addSpacing(15)
        row3.addWidget(QLabel("Manual Pick:"))
        row3.addWidget(self.rb_pwave)
        row3.addWidget(self.rb_swave)
        row3.addWidget(self.lbl_picks_readout)
        row3.addStretch()
        row3.addWidget(self.btn_prev_tab1)
        row3.addWidget(self.btn_next_tab1)
        row3.addWidget(self.btn_save)
        
        ctrl_layout.addLayout(row1)
        ctrl_layout.addLayout(row2)
        ctrl_layout.addLayout(row3)
        
        # Bottom Waveform Plot
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        bot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig_wave, self.ax_wave = plt.subplots(figsize=(10, 3.5))
        self.canvas_wave = FigureCanvas(self.fig_wave)
        self.toolbar_wave = NavigationToolbar(self.canvas_wave, self)
        
        bot_layout.addWidget(ctrl_widget)
        bot_layout.addWidget(self.toolbar_wave)
        bot_layout.addWidget(self.canvas_wave)
        
        self.canvas_wave.mpl_connect('button_press_event', self.on_waveform_click)
        
        splitter.addWidget(top_widget)
        splitter.addWidget(bot_widget)
        splitter.setSizes([240, 660])
        
        layout.addWidget(splitter)
        return widget

    # ==========================================
    # TAB 2: INTERACTIVE STA/LTA CALIBRATION
    # ==========================================
    def build_tab2_stalta(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Control Side Panel (Left)
        side_panel = QWidget()
        side_panel.setFixedWidth(320)
        side_layout = QVBoxLayout(side_panel)
        
        # Trace Switcher inside Tab 2
        group_trace_nav = QGroupBox("Trace Inspection Navigation")
        nav_box = QHBoxLayout(group_trace_nav)
        
        self.btn_prev_tab2 = QPushButton("<- Prev")
        self.btn_prev_tab2.clicked.connect(self.prev_waveform)
        self.lbl_trace_status = QLabel("Trace 0/0")
        self.lbl_trace_status.setStyleSheet("color: #88C0D0; font-weight: bold;")
        self.lbl_trace_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_next_tab2 = QPushButton("Next ->")
        self.btn_next_tab2.clicked.connect(self.next_waveform)
        
        nav_box.addWidget(self.btn_prev_tab2)
        nav_box.addWidget(self.lbl_trace_status)
        nav_box.addWidget(self.btn_next_tab2)
        
        group_sliders = QGroupBox("STA/LTA Threshold Controls")
        slider_box = QVBoxLayout(group_sliders)
        
        # STA Slider (0.1 to 10.0 us)
        slider_box.addWidget(QLabel("STA Window (µs):"))
        self.lbl_val_sta = QLabel("0.5 µs")
        self.lbl_val_sta.setStyleSheet("color: #88C0D0; font-weight: bold;")
        self.slider_sta = QSlider(Qt.Orientation.Horizontal)
        self.slider_sta.setRange(1, 100)
        self.slider_sta.setValue(5)
        self.slider_sta.valueChanged.connect(self.on_stalta_slider_changed)
        
        h_sta = QHBoxLayout()
        h_sta.addWidget(self.slider_sta)
        h_sta.addWidget(self.lbl_val_sta)
        slider_box.addLayout(h_sta)
        
        # LTA Slider (1.0 to 50.0 us)
        slider_box.addWidget(QLabel("LTA Window (µs):"))
        self.lbl_val_lta = QLabel("15.0 µs")
        self.lbl_val_lta.setStyleSheet("color: #88C0D0; font-weight: bold;")
        self.slider_lta = QSlider(Qt.Orientation.Horizontal)
        self.slider_lta.setRange(10, 500)
        self.slider_lta.setValue(150)
        self.slider_lta.valueChanged.connect(self.on_stalta_slider_changed)
        
        h_lta = QHBoxLayout()
        h_lta.addWidget(self.slider_lta)
        h_lta.addWidget(self.lbl_val_lta)
        slider_box.addLayout(h_lta)
        
        # P Threshold Slider (1.0 to 20.0)
        slider_box.addWidget(QLabel("P Trigger Threshold:"))
        self.lbl_val_pth = QLabel("3.5")
        self.lbl_val_pth.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.slider_pth = QSlider(Qt.Orientation.Horizontal)
        self.slider_pth.setRange(10, 200)
        self.slider_pth.setValue(35)
        self.slider_pth.valueChanged.connect(self.on_stalta_slider_changed)
        
        h_pth = QHBoxLayout()
        h_pth.addWidget(self.slider_pth)
        h_pth.addWidget(self.lbl_val_pth)
        slider_box.addLayout(h_pth)
        
        # S Threshold Slider (1.0 to 20.0)
        slider_box.addWidget(QLabel("S Trigger Threshold:"))
        self.lbl_val_sth = QLabel("2.5")
        self.lbl_val_sth.setStyleSheet("color: #55FF55; font-weight: bold;")
        self.slider_sth = QSlider(Qt.Orientation.Horizontal)
        self.slider_sth.setRange(10, 200)
        self.slider_sth.setValue(25)
        self.slider_sth.valueChanged.connect(self.on_stalta_slider_changed)
        
        h_sth = QHBoxLayout()
        h_sth.addWidget(self.slider_sth)
        h_sth.addWidget(self.lbl_val_sth)
        slider_box.addLayout(h_sth)
        
        # Batch Execution Button
        self.btn_apply_all_tab2 = QPushButton("Apply Parameters to All Traces")
        self.btn_apply_all_tab2.setStyleSheet("background-color: #1E5128; font-weight: bold; padding: 8px;")
        self.btn_apply_all_tab2.clicked.connect(self.run_stalta_on_current_file)
        
        side_layout.addWidget(group_trace_nav)
        side_layout.addWidget(group_sliders)
        side_layout.addSpacing(15)
        side_layout.addWidget(self.btn_apply_all_tab2)
        side_layout.addStretch()
        
        # Right Canvas (3 Stacked Subplots)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig_stalta, (self.ax_stalta_top, self.ax_stalta_mid, self.ax_stalta_bot) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True
        )
        self.canvas_stalta = FigureCanvas(self.fig_stalta)
        self.toolbar_stalta = NavigationToolbar(self.canvas_stalta, self)
        
        right_layout.addWidget(self.toolbar_stalta)
        right_layout.addWidget(self.canvas_stalta)
        
        layout.addWidget(side_panel)
        layout.addWidget(right_panel)
        return widget

    # ==========================================
    # TAB 3: CROSS-CORRELATION PICKER
    # ==========================================
    def build_tab3_cross_correlation(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        side_panel = QWidget()
        side_panel.setFixedWidth(340)
        side_layout = QVBoxLayout(side_panel)

        group_navigation = QGroupBox("Trace and Master Selection")
        navigation_layout = QVBoxLayout(group_navigation)
        navigation_row = QHBoxLayout()
        self.btn_prev_tab3 = QPushButton("<- Prev")
        self.btn_prev_tab3.clicked.connect(self.prev_waveform)
        self.lbl_cc_trace_status = QLabel("Trace 0/0")
        self.lbl_cc_trace_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cc_trace_status.setStyleSheet(
            "color: #88C0D0; font-weight: bold;"
        )
        self.btn_next_tab3 = QPushButton("Next ->")
        self.btn_next_tab3.clicked.connect(self.next_waveform)
        navigation_row.addWidget(self.btn_prev_tab3)
        navigation_row.addWidget(self.lbl_cc_trace_status)
        navigation_row.addWidget(self.btn_next_tab3)
        navigation_layout.addLayout(navigation_row)

        self.btn_set_cc_master = QPushButton("Use Current Trace as Master")
        self.btn_set_cc_master.setStyleSheet(
            "background-color: #2B5C8F; font-weight: bold; padding: 7px;"
        )
        self.btn_set_cc_master.clicked.connect(self.set_current_trace_as_cc_master)
        self.lbl_cc_master = QLabel("Master: not selected")
        self.lbl_cc_master.setStyleSheet("color: #D9792B; font-weight: bold;")
        navigation_layout.addWidget(self.btn_set_cc_master)
        navigation_layout.addWidget(self.lbl_cc_master)

        group_master_picks = QGroupBox("Master P/S Arrivals")
        master_pick_layout = QVBoxLayout(group_master_picks)
        master_pick_layout.addWidget(QLabel(
            "Choose P or S, then click its arrival on the master waveform."
        ))
        pick_mode_row = QHBoxLayout()
        self.rb_cc_pwave = QRadioButton("Pick master P")
        self.rb_cc_swave = QRadioButton("Pick master S")
        self.rb_cc_pwave.setChecked(True)
        self.rb_cc_pwave.setStyleSheet("color: #FF5555; font-weight: bold;")
        self.rb_cc_swave.setStyleSheet("color: #55FF55; font-weight: bold;")
        self.cc_pick_mode_group = QButtonGroup()
        self.cc_pick_mode_group.addButton(self.rb_cc_pwave)
        self.cc_pick_mode_group.addButton(self.rb_cc_swave)
        pick_mode_row.addWidget(self.rb_cc_pwave)
        pick_mode_row.addWidget(self.rb_cc_swave)
        master_pick_layout.addLayout(pick_mode_row)
        self.lbl_cc_master_picks = QLabel("Master picks — P: - | S: -")
        self.lbl_cc_master_picks.setStyleSheet(
            "color: #88C0D0; font-weight: bold;"
        )
        master_pick_layout.addWidget(self.lbl_cc_master_picks)

        group_parameters = QGroupBox("Template-Matching Parameters")
        parameter_layout = QVBoxLayout(group_parameters)

        def add_parameter_row(label_text, default_value):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            field = QLineEdit(default_value)
            field.setFixedWidth(75)
            row.addStretch()
            row.addWidget(field)
            parameter_layout.addLayout(row)
            return field

        self.txt_cc_p_halfwidth = add_parameter_row("P template half-width (µs):", "1.0")
        self.txt_cc_s_halfwidth = add_parameter_row("S template half-width (µs):", "1.5")
        self.txt_cc_search_halfwidth = add_parameter_row("Search ± (µs):", "5.0")
        self.txt_cc_min_coefficient = add_parameter_row("Minimum CC coefficient:", "0.50")

        self.btn_apply_cc_all = QPushButton("Apply Cross-Correlation to All Traces")
        self.btn_apply_cc_all.setStyleSheet(
            "background-color: #7A4E00; font-weight: bold; padding: 8px;"
        )
        self.btn_apply_cc_all.clicked.connect(self.run_cc_on_current_file)

        side_layout.addWidget(group_navigation)
        side_layout.addWidget(group_master_picks)
        side_layout.addWidget(group_parameters)
        side_layout.addSpacing(10)
        side_layout.addWidget(self.btn_apply_cc_all)
        side_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_cc, self.ax_cc = plt.subplots(figsize=(10, 7))
        self.canvas_cc = FigureCanvas(self.fig_cc)
        self.toolbar_cc = NavigationToolbar(self.canvas_cc, self)
        right_layout.addWidget(self.toolbar_cc)
        right_layout.addWidget(self.canvas_cc)
        self.canvas_cc.mpl_connect('button_press_event', self.on_cc_master_click)

        layout.addWidget(side_panel)
        layout.addWidget(right_panel)
        return widget

    # ==========================================
    # 4. BLANK PLOT INITIALIZERS
    # ==========================================
    def plot_blank_sync(self):
        self.ax_sync.clear()
        self.ax_sync.set_title("Sync Signal & Corrected 5V Falling Edge Peaks")
        self.ax_sync.set_xlabel("Time (s)")
        self.ax_sync.set_ylabel("Sync (V)")
        self.ax_sync.grid(True, linestyle='--', alpha=0.5)
        self.fig_sync.tight_layout()
        self.canvas_sync.draw()

    def plot_blank_waveform(self):
        self.ax_wave.clear()
        self.ax_wave.set_title("Waveform Trace Arrival Picker")
        self.ax_wave.set_xlabel("Time (µs)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)
        self.fig_wave.tight_layout()
        self.canvas_wave.draw()

    def plot_blank_stalta(self):
        for ax in [self.ax_stalta_top, self.ax_stalta_mid, self.ax_stalta_bot]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.ax_stalta_top.set_ylabel("Filtered Signal")
        self.ax_stalta_mid.set_ylabel("Abs Amp / STA / LTA")
        self.ax_stalta_bot.set_ylabel("STA/LTA Ratio")
        self.ax_stalta_bot.set_xlabel("Time (µs)")
        self.fig_stalta.tight_layout()
        self.canvas_stalta.draw()

    def plot_blank_cc(self):
        self.ax_cc.clear()
        self.ax_cc.set_title("Select a waveform interval and define a master trace")
        self.ax_cc.set_xlabel("Time (µs)")
        self.ax_cc.set_ylabel("Filtered amplitude")
        self.ax_cc.grid(True, alpha=0.3)
        self.fig_cc.tight_layout()
        self.canvas_cc.draw()

    # ==========================================
    # 5. DATA LOADING & SIGNAL PROCESSING
    # ==========================================
    def load_mechanical_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Mechanical Data File", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
            
        try:
            self.mech_df = pd.read_csv(file_path, sep='\t', skiprows=[1])
            if 'sync' not in self.mech_df.columns or 'time_s' not in self.mech_df.columns:
                raise ValueError("Dataset missing 'sync' or 'time_s' columns.")
                
            self.peaks_5v = find_5v_falling_peaks(self.mech_df)
            self.plot_sync_data()
            self.btn_select_dir.setEnabled(True)
            QMessageBox.information(self, "Success", f"Detected {len(self.peaks_5v)} 5V falling-edge peaks.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mechanical file:\n{str(e)}")

    def plot_sync_data(self):
        self.ax_sync.clear()
        time_vals = self.mech_df['time_s'].values
        sync_vals = self.mech_df['sync'].values
        
        self.ax_sync.plot(time_vals, sync_vals, color='#1f77b4', linewidth=1.0, label='Sync Signal')
        
        peak_times = time_vals[self.peaks_5v]
        peak_syncs = sync_vals[self.peaks_5v]
        self.ax_sync.plot(peak_times, peak_syncs, 'ro', markersize=5, label=f'5V Falling Peaks (N={len(self.peaks_5v)})')
        
        for rec in self.interval_records:
            is_active = (rec['interval_num'] - 1) == self.current_interval_idx
            alpha_val = 0.45 if is_active else 0.2
            self.ax_sync.axvspan(rec['start_time_s'], rec['end_time_s'], color='#FFE5B4', alpha=alpha_val)
            
        self.ax_sync.set_title("Sync Signal & Corrected 5V Falling Edge Peaks", fontweight='bold')
        self.ax_sync.set_xlabel("Time (s)")
        self.ax_sync.set_ylabel("Sync Voltage (V)")
        self.ax_sync.grid(True, linestyle='--', alpha=0.5)
        self.ax_sync.legend(loc='upper right')
        self.fig_sync.tight_layout()
        self.canvas_sync.draw()

    def associate_waveforms_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Waveforms Directory")
        if not dir_path:
            return
            
        tsv_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.tsv')])
        num_intervals = len(self.peaks_5v) // 2
        
        if len(tsv_files) < num_intervals:
            QMessageBox.warning(self, "Warning", f"Found {len(tsv_files)} .tsv files, expected {num_intervals}.")
            
        self.interval_records = []
        self.combo_intervals.blockSignals(True)
        self.combo_intervals.clear()
        
        for i in range(0, len(self.peaks_5v) - 1, 2):
            interval_num = (i // 2) + 1
            file_idx = i // 2
            if file_idx >= len(tsv_files):
                break
                
            file_name = tsv_files[file_idx]
            file_path = os.path.join(dir_path, file_name)
            
            s_idx, e_idx = self.peaks_5v[i], self.peaks_5v[i+1]
            t_start, t_end = self.mech_df.loc[s_idx, 'time_s'], self.mech_df.loc[e_idx, 'time_s']
            
            record = {
                'interval_num': interval_num,
                'file_name': file_name,
                'file_path': file_path,
                'start_idx': s_idx,
                'end_idx': e_idx,
                'start_time_s': t_start,
                'end_time_s': t_end,
            }
            self.interval_records.append(record)
            
            display_str = f"Interval {interval_num}: {file_name} ({t_start:.1f}s - {t_end:.1f}s)"
            self.combo_intervals.addItem(display_str)
            
            if file_name not in self.all_picks:
                self.all_picks[file_name] = {}
                
        self.combo_intervals.blockSignals(False)
        self.plot_sync_data()
        
        if self.interval_records:
            self.combo_intervals.setCurrentIndex(0)
            self.load_interval_waveform(0)

    def on_interval_changed(self, index):
        if 0 <= index < len(self.interval_records):
            self.load_interval_waveform(index)

    def load_interval_waveform(self, interval_idx):
        self.current_interval_idx = interval_idx
        record = self.interval_records[interval_idx]
        
        try:
            self.current_metadata, self.current_waveforms_df = read_bscan_tsv(record['file_path'])
            self.num_traces, self.num_samples = self.current_waveforms_df.shape
            self.current_trace_idx = 0
            self.cc_master_trace_idx = None
            self.cc_master_p_us = None
            self.cc_master_s_us = None
            self.lbl_cc_master.setText("Master: not selected")
            self.lbl_cc_master_picks.setText("Master picks — P: - | S: -")
            
            for k, v in self.current_metadata.items():
                if 'rate' in k.lower() or 'fs' in k.lower():
                    try:
                        self.sampling_rate_hz = float(''.join(c for c in v if c.isdigit() or c == '.')) * 1e6
                    except ValueError:
                        pass
                        
            fname = record['file_name']
            if fname not in self.all_picks:
                self.all_picks[fname] = {}
            for t in range(self.num_traces):
                if t not in self.all_picks[fname]:
                    self.all_picks[fname][t] = {
                        'P_manual': None, 'S_manual': None,
                        'P_auto': None, 'S_auto': None,
                        'P_cc': None, 'S_cc': None,
                        'P_cc_coeff': None, 'S_cc_coeff': None,
                    }
                else:
                    for key in ('P_cc', 'S_cc', 'P_cc_coeff', 'S_cc_coeff'):
                        self.all_picks[fname][t].setdefault(key, None)
                    
            self.plot_sync_data()
            self.update_all_plots(preserve_view=False)
            self.btn_waveform_overview.setEnabled(True)
            
        except Exception as e:
            self.btn_waveform_overview.setEnabled(False)
            QMessageBox.critical(self, "Error", f"Failed to load TSV waveform file:\n{str(e)}")

    def show_waveform_overview(self):
        """Show all traces and activate the waveform selected with the mouse."""
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            QMessageBox.warning(self, "Warning", "No waveform interval is loaded.")
            return

        file_name = self.interval_records[self.current_interval_idx]['file_name']
        dialog = WaveformOverviewDialog(
            self.current_waveforms_df,
            self.sampling_rate_hz,
            file_name,
            current_trace_idx=self.current_trace_idx,
            parent=self,
        )
        dialog.exec()

        if dialog.selected_trace_idx is not None:
            self.current_trace_idx = dialog.selected_trace_idx
            self.update_all_plots(preserve_view=False)

    def apply_signal_processing(self, raw_signal):
        """Pre-filtering with cutoff frequency applied before STA/LTA and manual display."""
        detrended = raw_signal - np.mean(raw_signal)
        try:
            cutoff_mhz = float(self.txt_cutoff.text())
            cutoff_hz = cutoff_mhz * 1e6
            nyquist = 0.5 * self.sampling_rate_hz
            if 0 < cutoff_hz < nyquist:
                b, a = butter(4, cutoff_hz / nyquist, btype='low')
                return filtfilt(b, a, detrended)
        except Exception:
            pass
        return detrended

    def on_tab_changed(self, index):
        self.update_all_plots(preserve_view=True)

    def update_all_plots(self, preserve_view=False):
        if self.num_traces > 0:
            self.lbl_trace_status.setText(f"Trace {self.current_trace_idx + 1}/{self.num_traces}")
            self.lbl_cc_trace_status.setText(
                f"Trace {self.current_trace_idx + 1}/{self.num_traces}"
            )
        else:
            self.lbl_trace_status.setText("Trace 0/0")
            self.lbl_cc_trace_status.setText("Trace 0/0")
            
        self.update_waveform_plot_tab1(preserve_view=preserve_view)
        self.update_stalta_plot_tab2(preserve_view=preserve_view)
        self.update_cc_plot_tab3(preserve_view=preserve_view)

    # ==========================================
    # 6. TAB 1 WAVEFORM PLOT RENDER (MANUAL ONLY)
    # ==========================================
    def toggle_theo_windows(self):
        self.show_theo_windows = self.btn_toggle_theo.isChecked()
        self.update_waveform_plot_tab1(preserve_view=True)

    def update_waveform_plot_tab1(self, preserve_view=False):
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            return
            
        cur_xlim = self.ax_wave.get_xlim() if preserve_view else None
        cur_ylim = self.ax_wave.get_ylim() if preserve_view else None
        is_default_view = (cur_xlim == (0.0, 1.0)) if cur_xlim else True
        
        self.ax_wave.clear()
        
        dt = 1.0 / self.sampling_rate_hz
        time_us = np.arange(self.num_samples) * dt * 1e6
        
        raw_waveform = self.current_waveforms_df.iloc[self.current_trace_idx].values
        processed = self.apply_signal_processing(raw_waveform)
        
        record = self.interval_records[self.current_interval_idx]
        fname = record['file_name']
        
        frac = (self.current_trace_idx / (self.num_traces - 1)) if self.num_traces > 1 else 0.0
        trace_mech_time = record['start_time_s'] + frac * (record['end_time_s'] - record['start_time_s'])
        trace_mech_row = int(round(record['start_idx'] + frac * (record['end_idx'] - record['start_idx'])))
        
        if self.show_theo_windows:
            try:
                L_m = float(self.txt_len_mm.text()) * 1e-3
                vp_ms = float(self.txt_vp.text())
                vs_ms = float(self.txt_vs.text())
                tol_us = float(self.txt_tol.text())
                
                t_p_us = (L_m / vp_ms) * 1e6 if vp_ms > 0 else 0
                t_s_us = (L_m / vs_ms) * 1e6 if vs_ms > 0 else 0
                
                self.ax_wave.axvspan(t_p_us - tol_us, t_p_us + tol_us, color='red', alpha=0.15, label='Theo P')
                self.ax_wave.axvspan(t_s_us - tol_us, t_s_us + tol_us, color='green', alpha=0.15, label='Theo S')
            except Exception:
                pass
        
        self.ax_wave.plot(time_us, processed, color='navy', linewidth=1.2, label='Filtered Signal')
        
        title_str = (f"File: {fname} | Trace {self.current_trace_idx + 1}/{self.num_traces} | "
                     f"Synched Mech Time: {trace_mech_time:.2f} s (Row {trace_mech_row})")
        self.ax_wave.set_title(title_str, fontsize=10, fontweight='bold')
        self.ax_wave.set_xlabel("Time (µs)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)
        
        picks = self.all_picks[fname][self.current_trace_idx]
        pm, sm = picks['P_manual'], picks['S_manual']
        
        if pm is not None:
            self.ax_wave.axvline(x=pm, color='#FF3333', linestyle='--', linewidth=1.8, label=f'Man P: {pm:.2f} µs')
        if sm is not None:
            self.ax_wave.axvline(x=sm, color='#33FF33', linestyle='--', linewidth=1.8, label=f'Man S: {sm:.2f} µs')
            
        self.ax_wave.legend(loc='upper right')
        
        if preserve_view and not is_default_view:
            self.ax_wave.set_xlim(cur_xlim)
            self.ax_wave.set_ylim(cur_ylim)
            
        self.fig_wave.tight_layout()
        self.canvas_wave.draw()
        
        pm_str = f"{pm:.2f}" if pm is not None else "-"
        sm_str = f"{sm:.2f}" if sm is not None else "-"
        self.lbl_picks_readout.setText(f"Manual Picks -> [P: {pm_str} | S: {sm_str}] µs")

    # ==========================================
    # 7. TAB 2 INTERACTIVE REAL-TIME RENDER
    # ==========================================
    def on_stalta_slider_changed(self):
        sta_us = self.slider_sta.value() / 10.0
        lta_us = self.slider_lta.value() / 10.0
        pth = self.slider_pth.value() / 10.0
        sth = self.slider_sth.value() / 10.0
        
        self.lbl_val_sta.setText(f"{sta_us:.1f} µs")
        self.lbl_val_lta.setText(f"{lta_us:.1f} µs")
        self.lbl_val_pth.setText(f"{pth:.1f}")
        self.lbl_val_sth.setText(f"{sth:.1f}")
        
        self.update_stalta_plot_tab2(preserve_view=True)

    def update_stalta_plot_tab2(self, preserve_view=False):
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            return
            
        sta_us = self.slider_sta.value() / 10.0
        lta_us = self.slider_lta.value() / 10.0
        p_thresh = self.slider_pth.value() / 10.0
        s_thresh = self.slider_sth.value() / 10.0
        
        cur_xlim = self.ax_stalta_top.get_xlim() if preserve_view else None
        cur_ylim_top = self.ax_stalta_top.get_ylim() if preserve_view else None
        cur_ylim_mid = self.ax_stalta_mid.get_ylim() if preserve_view else None
        cur_ylim_bot = self.ax_stalta_bot.get_ylim() if preserve_view else None
        is_default_view = (cur_xlim == (0.0, 1.0)) if cur_xlim else True
        
        fname = self.interval_records[self.current_interval_idx]['file_name']
        raw_sig = self.current_waveforms_df.iloc[self.current_trace_idx].values
        
        proc_sig = self.apply_signal_processing(raw_sig)
        
        p_us, s_us, sta_arr, lta_arr, ratio_arr = run_sta_lta_picker(
            proc_sig, self.sampling_rate_hz,
            sta_us=sta_us, lta_us=lta_us,
            p_thresh=p_thresh, s_thresh=s_thresh
        )
        
        self.all_picks[fname][self.current_trace_idx]['P_auto'] = p_us
        self.all_picks[fname][self.current_trace_idx]['S_auto'] = s_us
        
        dt_us = (1.0 / self.sampling_rate_hz) * 1e6
        time_us = np.arange(len(proc_sig)) * dt_us
        
        self.ax_stalta_top.clear()
        self.ax_stalta_mid.clear()
        self.ax_stalta_bot.clear()
        
        # Top Subplot: Pre-Filtered Acoustic Waveform
        self.ax_stalta_top.plot(time_us, proc_sig, color='navy', linewidth=1.0, label='Filtered Signal')
        if p_us is not None:
            self.ax_stalta_top.axvline(x=p_us, color='magenta', linestyle='--', linewidth=1.8, label=f'Auto P: {p_us:.2f} µs')
        if s_us is not None:
            self.ax_stalta_top.axvline(x=s_us, color='cyan', linestyle='--', linewidth=1.8, label=f'Auto S: {s_us:.2f} µs')
        self.ax_stalta_top.set_ylabel("Filtered Amp")
        self.ax_stalta_top.set_title(f"STA/LTA Calibration - Trace {self.current_trace_idx + 1}/{self.num_traces}", fontweight='bold')
        self.ax_stalta_top.grid(True, alpha=0.3)
        self.ax_stalta_top.legend(loc='upper right')
        
        # Middle Subplot: Absolute Pre-Filtered Amplitude & Moving Envelopes
        abs_sig = np.abs(proc_sig)
        self.ax_stalta_mid.plot(time_us, abs_sig, color='#A0A0A0', linewidth=0.8, alpha=0.6, label='Abs Filtered Amp')
        self.ax_stalta_mid.plot(time_us, sta_arr, color='red', linewidth=1.2, label='STA Envelope')
        self.ax_stalta_mid.plot(time_us, lta_arr, color='blue', linewidth=1.2, label='LTA Envelope')
        self.ax_stalta_mid.set_ylabel("Amplitude (a.u)")
        self.ax_stalta_mid.grid(True, alpha=0.3)
        self.ax_stalta_mid.legend(loc='upper right')
        
        # Bottom Subplot: STA/LTA Ratio & Cursor-Following Threshold Lines
        self.ax_stalta_bot.plot(time_us, ratio_arr, color='black', linewidth=1.2, label='STA/LTA Ratio')
        self.ax_stalta_bot.axhline(y=p_thresh, color='red', linestyle='--', linewidth=1.5, label=f'P Thresh ({p_thresh:.1f})')
        self.ax_stalta_bot.axhline(y=s_thresh, color='green', linestyle='--', linewidth=1.5, label=f'S Thresh ({s_thresh:.1f})')
        self.ax_stalta_bot.set_ylabel("STA/LTA Ratio")
        self.ax_stalta_bot.set_xlabel("Time (µs)")
        self.ax_stalta_bot.grid(True, alpha=0.3)
        self.ax_stalta_bot.legend(loc='upper right')
        
        if preserve_view and not is_default_view:
            self.ax_stalta_top.set_xlim(cur_xlim)
            self.ax_stalta_top.set_ylim(cur_ylim_top)
            self.ax_stalta_mid.set_ylim(cur_ylim_mid)
            self.ax_stalta_bot.set_ylim(cur_ylim_bot)
            
        self.fig_stalta.tight_layout()
        self.canvas_stalta.draw()

    # ==========================================
    # 8. TAB 3 CROSS-CORRELATION METHODS
    # ==========================================
    def set_current_trace_as_cc_master(self):
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            QMessageBox.warning(self, "Warning", "No waveform interval is loaded.")
            return

        self.cc_master_trace_idx = self.current_trace_idx
        fname = self.interval_records[self.current_interval_idx]['file_name']
        master_picks = self.all_picks[fname][self.cc_master_trace_idx]

        # Reuse manual arrivals when they have already been identified in Tab 1.
        self.cc_master_p_us = master_picks.get('P_manual')
        self.cc_master_s_us = master_picks.get('S_manual')
        self.lbl_cc_master.setText(
            f"Master: Trace {self.cc_master_trace_idx + 1}/{self.num_traces}"
        )
        self.update_cc_master_labels()
        self.update_cc_plot_tab3(preserve_view=False)

    def update_cc_master_labels(self):
        p_text = f"{self.cc_master_p_us:.3f} µs" if self.cc_master_p_us is not None else "-"
        s_text = f"{self.cc_master_s_us:.3f} µs" if self.cc_master_s_us is not None else "-"
        self.lbl_cc_master_picks.setText(f"Master picks — P: {p_text} | S: {s_text}")

    def on_cc_master_click(self, event):
        if event.inaxes != self.ax_cc or self.toolbar_cc.mode != '' or event.xdata is None:
            return
        if self.cc_master_trace_idx is None:
            QMessageBox.warning(
                self, "Select Master", "Use the current trace as the master before picking P and S."
            )
            return

        if self.rb_cc_pwave.isChecked():
            self.cc_master_p_us = float(event.xdata)
        else:
            self.cc_master_s_us = float(event.xdata)
        self.update_cc_master_labels()
        self.update_cc_plot_tab3(preserve_view=True)

    def update_cc_plot_tab3(self, preserve_view=False):
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            return

        plot_trace_idx = (
            self.cc_master_trace_idx
            if self.cc_master_trace_idx is not None
            else self.current_trace_idx
        )
        current_xlim = self.ax_cc.get_xlim() if preserve_view else None
        current_ylim = self.ax_cc.get_ylim() if preserve_view else None
        default_view = current_xlim == (0.0, 1.0) if current_xlim else True

        signal = self.apply_signal_processing(
            self.current_waveforms_df.iloc[plot_trace_idx].to_numpy(dtype=float)
        )
        time_us = np.arange(len(signal)) * (1.0 / self.sampling_rate_hz) * 1e6

        self.ax_cc.clear()
        self.ax_cc.plot(time_us, signal, color='navy', linewidth=1.0,
                        label='Pre-filtered waveform')

        try:
            p_halfwidth = float(self.txt_cc_p_halfwidth.text())
            s_halfwidth = float(self.txt_cc_s_halfwidth.text())
        except ValueError:
            p_halfwidth, s_halfwidth = 1.0, 1.5

        if self.cc_master_p_us is not None:
            self.ax_cc.axvline(self.cc_master_p_us, color='red', linestyle='--',
                               linewidth=1.8, label=f'Master P: {self.cc_master_p_us:.3f} µs')
            self.ax_cc.axvspan(
                self.cc_master_p_us - p_halfwidth,
                self.cc_master_p_us + p_halfwidth,
                color='red', alpha=0.12, label='P template'
            )
        if self.cc_master_s_us is not None:
            self.ax_cc.axvline(self.cc_master_s_us, color='green', linestyle='--',
                               linewidth=1.8, label=f'Master S: {self.cc_master_s_us:.3f} µs')
            self.ax_cc.axvspan(
                self.cc_master_s_us - s_halfwidth,
                self.cc_master_s_us + s_halfwidth,
                color='green', alpha=0.12, label='S template'
            )

        state = "master" if self.cc_master_trace_idx is not None else "candidate master"
        self.ax_cc.set_title(
            f"Cross-Correlation {state.title()} — Trace {plot_trace_idx + 1}/{self.num_traces}",
            fontweight='bold'
        )
        self.ax_cc.set_xlabel("Time (µs)")
        self.ax_cc.set_ylabel("Filtered amplitude")
        self.ax_cc.grid(True, alpha=0.3)
        self.ax_cc.legend(loc='upper right')

        if preserve_view and not default_view:
            self.ax_cc.set_xlim(current_xlim)
            self.ax_cc.set_ylim(current_ylim)
        self.fig_cc.tight_layout()
        self.canvas_cc.draw()

    def run_cc_on_current_file(self):
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            QMessageBox.warning(self, "Warning", "No waveform interval is loaded.")
            return
        if self.cc_master_trace_idx is None:
            QMessageBox.warning(self, "Missing Master", "Select a master trace first.")
            return
        if self.cc_master_p_us is None or self.cc_master_s_us is None:
            QMessageBox.warning(
                self, "Missing Arrivals", "Pick both P and S arrivals on the master waveform."
            )
            return

        try:
            p_half_us = float(self.txt_cc_p_halfwidth.text())
            s_half_us = float(self.txt_cc_s_halfwidth.text())
            search_half_us = float(self.txt_cc_search_halfwidth.text())
            minimum_cc = float(self.txt_cc_min_coefficient.text())
            if p_half_us <= 0 or s_half_us <= 0 or search_half_us < 0:
                raise ValueError
            if not 0.0 <= minimum_cc <= 1.0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(
                self, "Invalid Parameters",
                "Template widths must be positive, search width non-negative, "
                "and minimum CC between 0 and 1."
            )
            return

        samples_per_us = self.sampling_rate_hz / 1e6
        p_reference_index = int(round(self.cc_master_p_us * samples_per_us))
        s_reference_index = int(round(self.cc_master_s_us * samples_per_us))
        p_half_samples = max(1, int(round(p_half_us * samples_per_us)))
        s_half_samples = max(1, int(round(s_half_us * samples_per_us)))
        search_half_samples = max(0, int(round(search_half_us * samples_per_us)))

        master_signal = self.apply_signal_processing(
            self.current_waveforms_df.iloc[self.cc_master_trace_idx].to_numpy(dtype=float)
        )
        try:
            p_template = extract_cc_template(
                master_signal, p_reference_index, p_half_samples
            )
            s_template = extract_cc_template(
                master_signal, s_reference_index, s_half_samples
            )
        except ValueError as error:
            QMessageBox.critical(self, "Template Error", str(error))
            return

        fname = self.interval_records[self.current_interval_idx]['file_name']
        for trace_idx in range(self.num_traces):
            signal = self.apply_signal_processing(
                self.current_waveforms_df.iloc[trace_idx].to_numpy(dtype=float)
            )
            p_index, p_coefficient = normalized_cc_pick(
                signal, p_template, p_reference_index, search_half_samples
            )
            s_index, s_coefficient = normalized_cc_pick(
                signal, s_template, s_reference_index, search_half_samples
            )

            picks = self.all_picks[fname][trace_idx]
            picks['P_cc_coeff'] = (
                float(p_coefficient) if np.isfinite(p_coefficient) else None
            )
            picks['S_cc_coeff'] = (
                float(s_coefficient) if np.isfinite(s_coefficient) else None
            )
            picks['P_cc'] = (
                p_index / samples_per_us
                if p_index is not None and p_coefficient >= minimum_cc else None
            )
            picks['S_cc'] = (
                s_index / samples_per_us
                if s_index is not None and s_coefficient >= minimum_cc else None
            )

        self.update_all_plots(preserve_view=True)
        dialog = CrossCorrelationValidationDialog(
            self.current_waveforms_df,
            self.sampling_rate_hz,
            self.all_picks[fname],
            fname,
            parent=self,
        )
        dialog.exec()

    # ==========================================
    # 9. BATCH EXECUTION & MANUAL CONTROLS
    # ==========================================
    def run_stalta_on_current_file(self):
        """Runs STA/LTA using active slider parameters across all traces in active file."""
        if self.current_waveforms_df is None or self.current_interval_idx < 0:
            QMessageBox.warning(self, "Warning", "No active waveform file loaded.")
            return
            
        sta_us = self.slider_sta.value() / 10.0
        lta_us = self.slider_lta.value() / 10.0
        p_thresh = self.slider_pth.value() / 10.0
        s_thresh = self.slider_sth.value() / 10.0
        
        fname = self.interval_records[self.current_interval_idx]['file_name']
        
        for k in range(self.num_traces):
            raw_sig = self.current_waveforms_df.iloc[k].values
            proc_sig = self.apply_signal_processing(raw_sig)
            p_us, s_us, _, _, _ = run_sta_lta_picker(
                proc_sig, self.sampling_rate_hz,
                sta_us=sta_us, lta_us=lta_us,
                p_thresh=p_thresh, s_thresh=s_thresh
            )
            self.all_picks[fname][k]['P_auto'] = p_us
            self.all_picks[fname][k]['S_auto'] = s_us
            
        self.update_all_plots(preserve_view=True)
        
        dlg = STALTAValidationDialog(
            self.current_waveforms_df,
            self.sampling_rate_hz,
            self.all_picks[fname],
            fname,
            parent=self
        )
        dlg.exec()

    def on_waveform_click(self, event):
        if event.inaxes != self.ax_wave or self.toolbar_wave.mode != '':
            return
            
        fname = self.interval_records[self.current_interval_idx]['file_name']
        picked_time = event.xdata
        
        if self.rb_pwave.isChecked():
            self.all_picks[fname][self.current_trace_idx]['P_manual'] = picked_time
        else:
            self.all_picks[fname][self.current_trace_idx]['S_manual'] = picked_time
            
        self.update_waveform_plot_tab1(preserve_view=True)

    def prev_waveform(self):
        if self.current_waveforms_df is not None and self.current_trace_idx > 0:
            self.current_trace_idx -= 1
            self.update_all_plots(preserve_view=False)

    def next_waveform(self):
        if self.current_waveforms_df is not None and self.current_trace_idx < self.num_traces - 1:
            self.current_trace_idx += 1
            self.update_all_plots(preserve_view=False)

    def save_synched_picks(self):
        if not self.interval_records:
            QMessageBox.warning(self, "Warning", "No associated records to save.")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Synched Waveform Picks", "synched_waveform_picks.csv", "CSV Files (*.csv)"
        )
        if not output_path:
            return
            
        synched_rows = []
        for record in self.interval_records:
            fname = record['file_name']
            file_path = record['file_path']
            
            _, w_df = read_bscan_tsv(file_path)
            num_t, _ = w_df.shape
            
            start_t, end_t = record['start_time_s'], record['end_time_s']
            start_idx, end_idx = record['start_idx'], record['end_idx']
            
            for k in range(num_t):
                frac = (k / (num_t - 1)) if num_t > 1 else 0.0
                mech_time = start_t + frac * (end_t - start_t)
                mech_row = int(round(start_idx + frac * (end_idx - start_idx)))
                
                picks = self.all_picks.get(fname, {}).get(k, {
                    'P_manual': None, 'S_manual': None,
                    'P_auto': None, 'S_auto': None,
                    'P_cc': None, 'S_cc': None,
                    'P_cc_coeff': None, 'S_cc_coeff': None,
                })
                
                synched_rows.append({
                    'Interval_Num': record['interval_num'],
                    'File_Name': fname,
                    'Trace_Index': k,
                    'Mech_Time_s': mech_time,
                    'Mech_Row_Index': mech_row,
                    'P_Manual_us': picks['P_manual'],
                    'S_Manual_us': picks['S_manual'],
                    'P_Auto_us': picks['P_auto'],
                    'S_Auto_us': picks['S_auto'],
                    'P_CC_us': picks.get('P_cc'),
                    'S_CC_us': picks.get('S_cc'),
                    'P_CC_Coefficient': picks.get('P_cc_coeff'),
                    'S_CC_Coefficient': picks.get('S_cc_coeff'),
                })
                
        out_df = pd.DataFrame(synched_rows)
        out_df.to_csv(output_path, index=False)
        QMessageBox.information(self, "Success", f"Synchronized picks saved to:\n{output_path}")


# ==========================================
# 9. MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WaveformAnalyzerGUI()
    window.show()
    sys.exit(app.exec())
