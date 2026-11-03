#!/usr/bin/env python3
"""
Simplified Petrophysics & Triaxial Elasticity Explorer
=====================================================
Single-window teaching GUI featuring:
- Top Left: Differential stress vs. corrected axial strain with dynamic E-fit.
- Top Right: Derivative (tangent modulus) vs. differential stress with adjustable smoothing.
- Bottom: Live tracking plot of saved Young's Modulus (E) and R² values.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QCheckBox, QDoubleSpinBox, QSpinBox,
        QGroupBox, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
        QHeaderView
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QLabel, QCheckBox, QDoubleSpinBox, QSpinBox,
            QGroupBox, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
            QHeaderView
        )
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("PyQt6 or PySide6 along with Matplotlib, NumPy, and Pandas are required.")


COLORS = {
    "navy": "#18324A",
    "teal": "#087E8B",
    "orange": "#E07A3F",
    "blue": "#4C78A8",
    "green": "#4D8B31",
    "grey": "#737B80",
    "light": "#E8ECEF",
    "red": "#B33A3A",
    "purple": "#7B4173",
}

RAW_COLUMNS = [
    "v_lvdt_mm", "h_lvdt_mm", "v_load_kn", "h_load_kn",
    "pc_raw_mpa", "pc_displ_mm", "ppa_raw_mpa", "ppa_displ_mm",
    "ppb_raw_mpa", "ppb_displ_mm", "time_s", "record",
    "sync_channel", "sync_level",
]


class ExperimentData:
    """Handles raw/reduced triaxial test data import and zero reductions."""
    def __init__(self):
        self.raw_df = None
        self.reduced_df = None
        
        self.diameter_mm = 38.0
        self.length_mm = 76.0
        self.pc_zero_rec = 1642
        self.ppa_zero_rec = 2890
        self.ppb_zero_rec = 2922
        self.axial_zero_rec = 6827
        self.load_offset_kn = 0.0212

    @property
    def area_m2(self) -> float:
        return np.pi * (self.diameter_mm / 1000.0) ** 2 / 4.0

    def load_file(self, path: Path):
        if path.suffix.lower() == ".csv":
            self.reduced_df = pd.read_csv(path)
        else:
            self.raw_df = pd.read_csv(
                path, sep=r"\s+", skiprows=2, names=RAW_COLUMNS,
                usecols=range(14), engine="python"
            )
            self._reduce_raw()

    def _zero_from_rec(self, values: pd.Series, records: pd.Series, record: int) -> pd.Series:
        match = values.loc[records == record]
        if match.empty:
            return values - values.iloc[0]
        res = values - float(match.iloc[0])
        res.loc[records < record] = 0.0
        return res

    def _reduce_raw(self):
        d = self.raw_df.copy()
        d["pc_mpa"] = self._zero_from_rec(d.pc_raw_mpa, d.record, self.pc_zero_rec)
        d["ppa_mpa"] = self._zero_from_rec(d.ppa_raw_mpa, d.record, self.ppa_zero_rec)
        d["ppb_mpa"] = self._zero_from_rec(d.ppb_raw_mpa, d.record, self.ppb_zero_rec)
        d["pore_pressure_mpa"] = 0.5 * (d.ppa_mpa + d.ppb_mpa)

        signed_disp = -d.v_lvdt_mm
        d["piston_displacement_mm"] = self._zero_from_rec(
            signed_disp, d.record, self.axial_zero_rec
        )
        load = self._zero_from_rec(d.v_load_kn, d.record, self.axial_zero_rec)
        load.loc[d.record >= self.axial_zero_rec] += self.load_offset_kn
        d["axial_load_kn"] = load
        d["differential_stress_mpa"] = d.axial_load_kn / 1000.0 / self.area_m2
        d.loc[d.record < self.axial_zero_rec, "differential_stress_mpa"] = 0.0
        self.reduced_df = d


class MainCanvas(FigureCanvas):
    """Canvas layout containing top-left, top-right, and bottom-spanning axes."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(11, 8), dpi=100, facecolor="white")
        super().__init__(self.fig)
        self.setParent(parent)

        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)
        self.ax_stress_strain = self.fig.add_subplot(gs[0, 0])
        self.ax_derivative = self.fig.add_subplot(gs[0, 1])
        self.ax_history = self.fig.add_subplot(gs[1, :])

        self.axes = [self.ax_stress_strain, self.ax_derivative, self.ax_history]
        self.style_axes()

    def style_axes(self):
        for ax in self.axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color=COLORS["light"], linewidth=0.8, zorder=0)
            ax.tick_params(colors=COLORS["navy"], labelsize=9)
            ax.xaxis.label.set_color(COLORS["navy"])
            ax.yaxis.label.set_color(COLORS["navy"])


class MainWindow(QMainWindow):
    def __init__(self, data_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("Triaxial Elasticity & Modulus Fitting Explorer")
        self.resize(1300, 820)

        self.data_mgr = ExperimentData()
        self.saved_results = []

        self.current_e_gpa = None
        self.current_r2 = None

        if data_path and data_path.exists():
            self.data_mgr.load_file(data_path)

        self._init_ui()
        if self.data_mgr.reduced_df is not None:
            self.update_plots()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Sidebar Panel
        sidebar = QWidget()
        sidebar.setFixedWidth(330)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(5, 5, 5, 5)

        btn_load = QPushButton("Open Data File (Raw / Reduced CSV)")
        btn_load.setStyleSheet("padding: 8px; font-weight: bold; background-color: #087E8B; color: white;")
        btn_load.clicked.connect(self._open_file)
        side_layout.addWidget(btn_load)

        # Machine Compliance Group
        grp_compliance = QGroupBox("1. Frame Compliance")
        lay_comp = QVBoxLayout()
        self.chk_frame_corr = QCheckBox("Enable Frame Correction")
        self.chk_frame_corr.setChecked(True)
        self.chk_frame_corr.toggled.connect(self.update_plots)
        lay_comp.addWidget(self.chk_frame_corr)

        lay_comp.addWidget(QLabel("K_frame (MPa/mm):"))
        self.spin_k_frame = QDoubleSpinBox()
        self.spin_k_frame.setRange(100.0, 3000.0)
        self.spin_k_frame.setValue(819.11)
        self.spin_k_frame.setSingleStep(25.0)
        self.spin_k_frame.valueChanged.connect(self.update_plots)
        lay_comp.addWidget(self.spin_k_frame)
        grp_compliance.setLayout(lay_comp)
        side_layout.addWidget(grp_compliance)

        # Young's Modulus Fitting Group
        grp_e = QGroupBox("2. Dynamic Slope (Young's Modulus)")
        lay_e = QGridLayout()
        lay_e.addWidget(QLabel("q_min (MPa):"), 0, 0)
        self.spin_q_min = QDoubleSpinBox()
        self.spin_q_min.setRange(0.0, 120.0)
        self.spin_q_min.setValue(20.0)
        self.spin_q_min.setSingleStep(5.0)
        self.spin_q_min.valueChanged.connect(self.update_plots)
        lay_e.addWidget(self.spin_q_min, 0, 1)

        lay_e.addWidget(QLabel("q_max (MPa):"), 1, 0)
        self.spin_q_max = QDoubleSpinBox()
        self.spin_q_max.setRange(5.0, 150.0)
        self.spin_q_max.setValue(60.0)
        self.spin_q_max.setSingleStep(5.0)
        self.spin_q_max.valueChanged.connect(self.update_plots)
        lay_e.addWidget(self.spin_q_max, 1, 1)

        self.lbl_e_res = QLabel("E: -- GPa\nR²: --")
        self.lbl_e_res.setStyleSheet("font-weight: bold; color: #087E8B; font-size: 13px; margin-top: 4px;")
        lay_e.addWidget(self.lbl_e_res, 2, 0, 1, 2)
        grp_e.setLayout(lay_e)
        side_layout.addWidget(grp_e)

        # Derivative / Tangent Modulus Group
        grp_deriv = QGroupBox("3. Derivative Smoothing")
        lay_deriv = QVBoxLayout()
        lay_deriv.addWidget(QLabel("Smoothing Window Size (points):"))
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 101)
        self.spin_smooth.setValue(15)
        self.spin_smooth.setSingleStep(2)
        self.spin_smooth.valueChanged.connect(self.update_plots)
        lay_deriv.addWidget(self.spin_smooth)
        grp_deriv.setLayout(lay_deriv)
        side_layout.addWidget(grp_deriv)

        # Save & Tracking Controls Group
        grp_save = QGroupBox("4. Save & Track Values")
        lay_save = QVBoxLayout()
        btn_save = QPushButton("Save Current E & R² Value")
        btn_save.setStyleSheet("padding: 8px; font-weight: bold; background-color: #18324A; color: white;")
        btn_save.clicked.connect(self._save_current_value)
        lay_save.addWidget(btn_save)

        btn_clear = QPushButton("Clear Saved Points")
        btn_clear.clicked.connect(self._clear_saved_values)
        lay_save.addWidget(btn_clear)

        self.lbl_saved_count = QLabel("Saved points: 0")
        lay_save.addWidget(self.lbl_saved_count)
        grp_save.setLayout(lay_save)
        side_layout.addWidget(grp_save)

        side_layout.addStretch()

        # Canvas Setup
        self.canvas = MainCanvas(self)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.canvas, stretch=1)

    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Triaxial Data File", "", "All Supported (*.csv b849*);;CSV Files (*.csv)"
        )
        if file_path:
            self.data_mgr.load_file(Path(file_path))
            self.update_plots()

    def _get_processed_data(self):
        d = self.data_mgr.reduced_df.copy()
        test = d.loc[d.record >= self.data_mgr.axial_zero_rec].copy()

        k_frame = self.spin_k_frame.value()
        q = test["differential_stress_mpa"].to_numpy()

        if self.chk_frame_corr.isChecked():
            frame_ext = q / k_frame
        else:
            frame_ext = 0.0

        shortening = test["piston_displacement_mm"].to_numpy() - frame_ext
        axial_strain = shortening / self.data_mgr.length_mm
        test["axial_strain"] = axial_strain
        return test

    def update_plots(self):
        if self.data_mgr.reduced_df is None:
            return

        test = self._get_processed_data()
        
        peak_idx = int(np.nanargmax(test["differential_stress_mpa"].to_numpy()))
        branch = test.iloc[: peak_idx + 1].copy()

        strain = branch["axial_strain"].to_numpy()
        q_val = branch["differential_stress_mpa"].to_numpy()

        q_min = self.spin_q_min.value()
        q_max = self.spin_q_max.value()

        # Top Left Plot: Stress vs. Strain
        ax_tl = self.canvas.ax_stress_strain
        ax_tl.clear()
        ax_tl.plot(100 * strain, q_val, color=COLORS["navy"], lw=1.8, label="Data")

        mask = branch["differential_stress_mpa"].between(q_min, q_max)
        x_fit = branch.loc[mask, "axial_strain"].to_numpy()
        y_fit = branch.loc[mask, "differential_stress_mpa"].to_numpy()

        if x_fit.size >= 3:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            pred = slope * x_fit + intercept
            ss_res = np.sum((y_fit - pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            e_gpa = slope / 1000.0

            self.current_e_gpa = e_gpa
            self.current_r2 = r2

            ax_tl.plot(100 * x_fit, pred, color=COLORS["teal"], lw=3, label=f"Fit (E={e_gpa:.2f} GPa)")
            ax_tl.fill_between(100 * x_fit, q_min, q_max, color=COLORS["teal"], alpha=0.15)
            self.lbl_e_res.setText(f"E: {e_gpa:.2f} GPa\nR²: {r2:.4f}")
        else:
            self.current_e_gpa = None
            self.current_r2 = None
            self.lbl_e_res.setText("E: Range too small\nR²: --")

        corr_str = "Corrected" if self.chk_frame_corr.isChecked() else "Uncorrected"
        ax_tl.set_title(f"Differential Stress vs. Axial Strain ({corr_str})", fontsize=11, fontweight="bold")
        ax_tl.set_xlabel("Axial Strain (%)")
        ax_tl.set_ylabel("Differential Stress, q (MPa)")
        ax_tl.legend(frameon=False, loc="lower right", fontsize=8)

        # Top Right Plot: Derivative
        ax_tr = self.canvas.ax_derivative
        ax_tr.clear()

        if len(q_val) > 5:
            dq = np.gradient(q_val)
            de = np.gradient(strain)
            with np.errstate(divide='ignore', invalid='ignore'):
                tangent_e = np.where(de > 1e-7, (dq / de) / 1000.0, np.nan)

            w_size = self.spin_smooth.value()
            smooth_e = pd.Series(tangent_e).rolling(window=w_size, min_periods=1, center=True).mean().to_numpy()

            ax_tr.plot(q_val, tangent_e, color=COLORS["grey"], alpha=0.35, lw=0.8, label="Raw Derivative")
            ax_tr.plot(q_val, smooth_e, color=COLORS["purple"], lw=2.0, label=f"Smoothed (w={w_size})")
            ax_tr.axvspan(q_min, q_max, color=COLORS["teal"], alpha=0.12, label="Selected Fitting Range")

            valid_max = np.nanmax(smooth_e) if not np.all(np.isnan(smooth_e)) else 30
            ax_tr.set_ylim(0, max(10, valid_max * 1.25))

        ax_tr.set_title("Derivative (Tangent Modulus dq/dε)", fontsize=11, fontweight="bold")
        ax_tr.set_xlabel("Differential Stress, q (MPa)")
        ax_tr.set_ylabel("Tangent Modulus (GPa)")
        ax_tr.legend(frameon=False, loc="upper right", fontsize=8)

        # Bottom Plot: Tracking History
        self._update_bottom_plot()

        self.canvas.style_axes()
        self.canvas.draw_idle()

    def _update_bottom_plot(self):
        ax_b = self.canvas.ax_history
        ax_b.clear()

        if not self.saved_results:
            ax_b.set_title("Saved Young's Modulus History (Click 'Save Current E & R² Value')", fontsize=11, fontweight="bold")
            ax_b.set_xlabel("Saved Index / Midpoint Stress q (MPa)")
            ax_b.set_ylabel("Young's Modulus E (GPa)")
            return

        indices = [r["index"] for r in self.saved_results]
        e_vals = [r["e_gpa"] for r in self.saved_results]

        ax_b.plot(indices, e_vals, "-o", color=COLORS["teal"], lw=2, ms=7, zorder=3, label="Saved E (GPa)")
        
        for r in self.saved_results:
            ax_b.annotate(
                f"E={r['e_gpa']:.2f}\nR²={r['r2']:.3f}\n[{r['q_min']:.0f}-{r['q_max']:.0f}]",
                (r["index"], r["e_gpa"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color=COLORS["navy"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["teal"], alpha=0.8)
            )

        ax_b.set_title("Saved Young's Modulus (E) and Fit Quality Tracking", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Saved Evaluation Count (#)")
        ax_b.set_ylabel("Young's Modulus E (GPa)")
        
        y_max = max(e_vals) * 1.35 if e_vals else 20
        ax_b.set_ylim(0, y_max)
        ax_b.set_xticks(indices)

    def _save_current_value(self):
        if self.current_e_gpa is None or self.current_r2 is None:
            return

        q_min = self.spin_q_min.value()
        q_max = self.spin_q_max.value()
        q_mid = 0.5 * (q_min + q_max)

        record = {
            "index": len(self.saved_results) + 1,
            "e_gpa": self.current_e_gpa,
            "r2": self.current_r2,
            "q_min": q_min,
            "q_max": q_max,
            "q_mid": q_mid
        }
        self.saved_results.append(record)
        self.lbl_saved_count.setText(f"Saved points: {len(self.saved_results)}")
        self.update_plots()

    def _clear_saved_values(self):
        self.saved_results.clear()
        self.lbl_saved_count.setText("Saved points: 0")
        self.update_plots()


def main():
    app = QApplication(sys.argv)

    default_csv = Path("b849_reduced.csv")
    data_path = default_csv if default_csv.exists() else None
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    win = MainWindow(data_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()