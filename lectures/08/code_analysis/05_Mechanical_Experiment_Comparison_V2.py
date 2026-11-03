#!/usr/bin/env python3
"""Interactive comparison of reduced mechanical UCS experiments.

The program reads tab-separated ``*_acousto_mech.txt`` files whose first row
contains column names and second row contains units. Files may be added from
different directories. Any subset of experiments can be plotted, with an
optional second Y axis and a two-click linear-slope measurement tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = [
    "#0B7F86", "#D77A2D", "#3D6E91", "#C4493D", "#755A96",
    "#3C8D5A", "#8B6545", "#B65D89", "#64757D", "#A59A2A",
]
NO_RIGHT_AXIS = "(None)"


def read_acousto_mech_file(path: str | Path) -> dict:
    """Read one analyzer-ready mechanical file and its units row."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as handle:
        columns = handle.readline().rstrip("\r\n").split("\t")
        units = handle.readline().rstrip("\r\n").split("\t")
    if len(columns) != len(units):
        raise ValueError(
            f"units row has {len(units)} entries but header has {len(columns)}"
        )

    frame = pd.read_csv(path, sep="\t", skiprows=[1], low_memory=False)
    if list(frame.columns) != columns:
        raise ValueError("parsed columns do not match the header")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty:
        raise ValueError("file contains no numerical records")

    name = path.stem
    for suffix in ("_acousto_mech", "-acousto_mech"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return {
        "base_name": name,
        "name": name,
        "path": path,
        "data": frame,
        "units": dict(zip(columns, units)),
    }


def common_numeric_columns(experiments: list[dict], include_raw=False) -> list[str]:
    """Return numerical columns shared by all active experiments."""
    if not experiments:
        return []
    common = None
    first_order = []
    for experiment in experiments:
        frame = experiment["data"]
        columns = [
            column for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and frame[column].notna().any()
            and (include_raw or not column.endswith("_raw"))
        ]
        if common is None:
            common, first_order = set(columns), columns
        else:
            common &= set(columns)
    return [column for column in first_order if column in (common or set())]


def preferred_column(columns, choices, fallback):
    for choice in choices:
        if choice in columns:
            return choice
    return columns[min(fallback, len(columns) - 1)]


def run_gui(initial_paths: list[str] | None = None) -> int:
    """Create and run the PyQt6 comparison window."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
        QMainWindow, QMenu, QMessageBox, QPushButton, QToolButton, QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure

    class MechanicalComparisonWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("UCS mechanical experiment comparison")
            self.resize(1500, 900)
            self.experiments = []
            self.active_paths = set()
            self.lines = []
            self.annotations = {}
            self.experiment_actions = []
            self.all_action = None
            self.slope_start = None
            self.slope_graphics = []

            central = QWidget()
            main = QVBoxLayout(central)
            main.setContentsMargins(8, 8, 8, 8)

            file_row = QHBoxLayout()
            self.add_button = QPushButton("Add experiments…")
            self.add_button.setToolTip(
                "Add files without removing those already loaded; repeat to "
                "load files from another directory."
            )
            self.clear_files_button = QPushButton("Clear all")
            file_row.addWidget(self.add_button)
            file_row.addWidget(self.clear_files_button)
            file_row.addSpacing(10)
            file_row.addWidget(QLabel("Active experiments:"))
            self.experiment_button = QToolButton()
            self.experiment_button.setText("No experiments")
            self.experiment_button.setMinimumWidth(190)
            self.experiment_button.setPopupMode(
                QToolButton.ToolButtonPopupMode.InstantPopup
            )
            self.experiment_menu = QMenu(self.experiment_button)
            self.experiment_button.setMenu(self.experiment_menu)
            file_row.addWidget(self.experiment_button)
            file_row.addSpacing(10)
            self.raw_checkbox = QCheckBox("Include raw ADC columns")
            file_row.addWidget(self.raw_checkbox)
            file_row.addStretch()
            self.file_label = QLabel("No experiments loaded")
            file_row.addWidget(self.file_label)

            plot_row = QHBoxLayout()
            plot_row.addWidget(QLabel("X:"))
            self.x_combo = QComboBox()
            self.x_combo.setMinimumWidth(200)
            plot_row.addWidget(self.x_combo)
            plot_row.addWidget(QLabel("Left Y:"))
            self.y_combo = QComboBox()
            self.y_combo.setMinimumWidth(220)
            plot_row.addWidget(self.y_combo)
            plot_row.addWidget(QLabel("Right Y:"))
            self.y2_combo = QComboBox()
            self.y2_combo.setMinimumWidth(220)
            self.y2_combo.addItem(NO_RIGHT_AXIS)
            plot_row.addWidget(self.y2_combo)
            plot_row.addWidget(QLabel("Interaction:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Inspect / hover", "Measure slope (2 clicks)"])
            self.mode_combo.setMinimumWidth(185)
            plot_row.addWidget(self.mode_combo)
            self.clear_slope_button = QPushButton("Clear slope")
            self.reset_button = QPushButton("Reset view")
            plot_row.addWidget(self.clear_slope_button)
            plot_row.addWidget(self.reset_button)
            plot_row.addStretch()
            self.slope_label = QLabel("Slope: —")
            self.slope_label.setMinimumWidth(225)
            plot_row.addWidget(self.slope_label)

            self.figure = Figure(figsize=(12.8, 7.2))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.left_axis = self.figure.add_subplot(111)
            self.right_axis = self.left_axis.twinx()
            self.right_axis.set_visible(False)

            main.addLayout(file_row)
            main.addLayout(plot_row)
            main.addWidget(self.toolbar)
            main.addWidget(self.canvas, stretch=1)
            self.setCentralWidget(central)

            self.add_button.clicked.connect(self.choose_files)
            self.clear_files_button.clicked.connect(self.clear_files)
            self.raw_checkbox.toggled.connect(self.refresh_choices)
            self.x_combo.currentTextChanged.connect(self.redraw)
            self.y_combo.currentTextChanged.connect(self.redraw)
            self.y2_combo.currentTextChanged.connect(self.redraw)
            self.mode_combo.currentTextChanged.connect(self.change_mode)
            self.clear_slope_button.clicked.connect(self.clear_slope)
            self.reset_button.clicked.connect(self.reset_view)
            self.canvas.mpl_connect("motion_notify_event", self.show_hover)
            self.canvas.mpl_connect("button_press_event", self.plot_clicked)
            self.draw_message("Add one or more *_acousto_mech.txt files")

        def draw_message(self, message):
            self.left_axis.clear()
            self.right_axis.clear()
            self.right_axis.set_visible(False)
            self.left_axis.text(
                0.5, 0.5, message, transform=self.left_axis.transAxes,
                ha="center", va="center", fontsize=17, color="#64757D",
            )
            self.left_axis.set_axis_off()
            self.canvas.draw_idle()

        # ----- Loading and activation -------------------------------------
        def choose_files(self, _checked=False):
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Add mechanical experiment files", "",
                "Acousto-mechanical text files (*.txt);;All files (*)",
            )
            if paths:
                self.load_paths(paths)

        def load_paths(self, paths):
            known = {experiment["path"] for experiment in self.experiments}
            added, errors, skipped = [], [], []
            for supplied in paths:
                path = Path(supplied).expanduser().resolve()
                if path in known:
                    skipped.append(path.name)
                    continue
                try:
                    experiment = read_acousto_mech_file(path)
                    added.append(experiment)
                    known.add(path)
                except Exception as error:
                    errors.append(f"{path.name}: {error}")
            if errors:
                QMessageBox.warning(
                    self, "Some files could not be loaded", "\n\n".join(errors)
                )
            if not added:
                if skipped:
                    self.statusBar().showMessage("Already loaded: " + ", ".join(skipped))
                return

            self.experiments.extend(added)
            self.active_paths.update(item["path"] for item in added)
            self.unique_names()
            self.rebuild_experiment_menu()
            self.file_label.setText(f"{len(self.experiments)} experiment(s) loaded")
            self.refresh_choices()
            message = "Added: " + ", ".join(item["path"].name for item in added)
            if skipped:
                message += " | Already loaded: " + ", ".join(skipped)
            self.statusBar().showMessage(message)

        def unique_names(self):
            groups = {}
            for experiment in self.experiments:
                groups.setdefault(experiment["base_name"], []).append(experiment)
            used = set()
            for base, group in groups.items():
                for number, experiment in enumerate(group, 1):
                    name = base if len(group) == 1 else f"{base} — {experiment['path'].parent.name}"
                    if name in used:
                        name = f"{name} ({number})"
                    experiment["name"] = name
                    used.add(name)

        def rebuild_experiment_menu(self):
            self.experiment_menu.clear()
            self.experiment_actions = []
            self.all_action = QAction("All experiments", self)
            self.all_action.setCheckable(True)
            self.all_action.setChecked(
                bool(self.experiments)
                and all(item["path"] in self.active_paths for item in self.experiments)
            )
            self.all_action.toggled.connect(self.toggle_all)
            self.experiment_menu.addAction(self.all_action)
            self.experiment_menu.addSeparator()
            for index, experiment in enumerate(self.experiments):
                action = QAction(experiment["name"], self)
                action.setCheckable(True)
                action.setChecked(experiment["path"] in self.active_paths)
                action.setToolTip(str(experiment["path"]))
                action.toggled.connect(
                    lambda checked, i=index: self.toggle_one(i, checked)
                )
                self.experiment_menu.addAction(action)
                self.experiment_actions.append(action)
            self.update_experiment_button()

        def toggle_all(self, checked):
            for action in self.experiment_actions:
                action.blockSignals(True)
                action.setChecked(checked)
                action.blockSignals(False)
            self.active_paths = (
                {item["path"] for item in self.experiments} if checked else set()
            )
            self.update_experiment_button()
            self.refresh_choices()

        def toggle_one(self, index, checked):
            path = self.experiments[index]["path"]
            self.active_paths.add(path) if checked else self.active_paths.discard(path)
            all_checked = bool(self.experiment_actions) and all(
                action.isChecked() for action in self.experiment_actions
            )
            self.all_action.blockSignals(True)
            self.all_action.setChecked(all_checked)
            self.all_action.blockSignals(False)
            self.update_experiment_button()
            self.refresh_choices()

        def active_experiments(self):
            return [
                item for item in self.experiments if item["path"] in self.active_paths
            ]

        def update_experiment_button(self):
            active, total = self.active_experiments(), len(self.experiments)
            if not total:
                text = "No experiments"
            elif len(active) == total:
                text = f"All ({total})"
            elif not active:
                text = f"None (0/{total})"
            elif len(active) == 1:
                text = active[0]["name"]
            else:
                text = f"Selected ({len(active)}/{total})"
            self.experiment_button.setText(text)

        def clear_files(self, _checked=False):
            self.experiments.clear()
            self.active_paths.clear()
            self.rebuild_experiment_menu()
            self.file_label.setText("No experiments loaded")
            for combo in (self.x_combo, self.y_combo, self.y2_combo):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            self.y2_combo.addItem(NO_RIGHT_AXIS)
            self.clear_slope(redraw=False)
            self.draw_message("Add one or more *_acousto_mech.txt files")

        # ----- Plot choices and rendering ---------------------------------
        def refresh_choices(self, *_args):
            active = self.active_experiments()
            if not active:
                self.clear_slope(redraw=False)
                self.draw_message("No experiments are active")
                return
            columns = common_numeric_columns(active, self.raw_checkbox.isChecked())
            if len(columns) < 2:
                QMessageBox.critical(
                    self, "No common variables",
                    "The active files do not share at least two numerical columns."
                )
                return

            old_x, old_y, old_y2 = (
                self.x_combo.currentText(), self.y_combo.currentText(),
                self.y2_combo.currentText(),
            )
            default_x = preferred_column(
                columns,
                ("axial_strain_percent", "axial_strain", "time_s", "elapsed_time_s", "record"),
                0,
            )
            default_y = preferred_column(
                columns,
                ("axial_stress_MPa", "axial_force_zeroed_kN", "vertical_load_calibrated"),
                1,
            )
            for combo in (self.x_combo, self.y_combo, self.y2_combo):
                combo.blockSignals(True)
                combo.clear()
            self.x_combo.addItems(columns)
            self.y_combo.addItems(columns)
            self.y2_combo.addItem(NO_RIGHT_AXIS)
            self.y2_combo.addItems(columns)
            self.x_combo.setCurrentText(old_x if old_x in columns else default_x)
            self.y_combo.setCurrentText(old_y if old_y in columns else default_y)
            self.y2_combo.setCurrentText(old_y2 if old_y2 in columns else NO_RIGHT_AXIS)
            for combo in (self.x_combo, self.y_combo, self.y2_combo):
                combo.blockSignals(False)
            self.redraw()

        def common_unit(self, column):
            units = {
                str(item["units"].get(column, "")).strip()
                for item in self.active_experiments()
                if str(item["units"].get(column, "")).strip()
            }
            return next(iter(units)) if len(units) == 1 else ("mixed units" if units else "")

        def axis_label(self, column):
            unit = self.common_unit(column)
            return f"{column} [{unit}]" if unit else column

        def redraw(self, *_args):
            active = self.active_experiments()
            x_name, y_name, y2_name = (
                self.x_combo.currentText(), self.y_combo.currentText(),
                self.y2_combo.currentText(),
            )
            if not active or not x_name or not y_name:
                return
            self.clear_slope(redraw=False)
            self.left_axis.clear()
            self.right_axis.clear()
            self.left_axis.set_axis_on()
            use_right = bool(y2_name and y2_name != NO_RIGHT_AXIS)
            self.right_axis.set_visible(use_right)
            self.lines = []

            for index, experiment in enumerate(active):
                frame = experiment["data"]
                color = COLORS[index % len(COLORS)]
                x = frame[x_name].to_numpy(float)
                for axis, y_column, side, style, width in [
                    (self.left_axis, y_name, "left", "-", 1.9),
                    (self.right_axis, y2_name, "right", "--", 1.7),
                ]:
                    if side == "right" and not use_right:
                        continue
                    y = frame[y_column].to_numpy(float)
                    valid = np.isfinite(x) & np.isfinite(y)
                    if not valid.any():
                        continue
                    line, = axis.plot(
                        x[valid], y[valid], color=color, linewidth=width,
                        linestyle=style, alpha=0.92,
                        label=experiment["name"] if side == "left" else "_nolegend_",
                    )
                    line.set_picker(7)
                    self.lines.append({
                        "line": line, "experiment": experiment, "side": side,
                        "x_column": x_name, "y_column": y_column,
                    })

            self.left_axis.set_xlabel(self.axis_label(x_name), fontsize=12)
            self.left_axis.set_ylabel(self.axis_label(y_name), fontsize=12)
            title = f"Mechanical comparison: {y_name} versus {x_name}"
            if use_right:
                self.right_axis.set_ylabel(self.axis_label(y2_name), fontsize=12, color="#8B4A1D")
                self.right_axis.tick_params(axis="y", colors="#8B4A1D")
                title += f" | dashed: {y2_name} (right axis)"
            self.left_axis.set_title(title, color="#102A3A", fontsize=14, pad=15)
            self.left_axis.grid(alpha=0.23)
            self.left_axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=min(4, max(1, len(active))), fontsize=9, frameon=True,
            )
            self.annotations = {}
            for axis in [self.left_axis] + ([self.right_axis] if use_right else []):
                annotation = axis.annotate(
                    "", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor="#0B7F86", alpha=0.96),
                    arrowprops=dict(arrowstyle="->", color="#0B7F86"), zorder=40,
                )
                annotation.set_visible(False)
                self.annotations[axis] = annotation
            self.figure.subplots_adjust(
                left=0.085, right=0.90 if use_right else 0.97,
                bottom=0.11, top=0.88,
            )
            self.canvas.draw_idle()

        def reset_view(self, _checked=False):
            for axis in (self.left_axis, self.right_axis):
                if axis.get_visible():
                    axis.relim()
                    axis.autoscale_view()
            self.canvas.draw_idle()

        # ----- Hover and curve selection ----------------------------------
        def nearest_point(self, event):
            candidates = []
            for info in self.lines:
                line = info["line"]
                contains, detail = line.contains(event)
                x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                for point in detail.get("ind", []) if contains else []:
                    point = int(point)
                    px, py = line.get_transform().transform((x[point], y[point]))
                    candidates.append((np.hypot(px - event.x, py - event.y), info, point))
            if not candidates:
                return None
            _, info, point = min(candidates, key=lambda item: item[0])
            return info, point

        def show_hover(self, event):
            previously_visible = any(item.get_visible() for item in self.annotations.values())
            for item in self.annotations.values():
                item.set_visible(False)
            picked = self.nearest_point(event)
            if picked:
                info, point = picked
                line = info["line"]
                x, y = line.get_xdata()[point], line.get_ydata()[point]
                annotation = self.annotations.get(line.axes)
                if annotation:
                    annotation.xy = (x, y)
                    annotation.set_text(
                        f"{info['experiment']['name']} — {info['side']} Y\n"
                        f"x: {x:.6g}\ny: {y:.6g}"
                    )
                    annotation.set_visible(True)
            now_visible = any(item.get_visible() for item in self.annotations.values())
            if previously_visible or now_visible:
                self.canvas.draw_idle()

        # ----- Two-click slope measurement --------------------------------
        def change_mode(self, text):
            self.clear_slope()
            message = (
                "Slope mode: click start and end points on the same curve."
                if text.startswith("Measure")
                else "Inspect mode: move the cursor over a curve."
            )
            self.statusBar().showMessage(message)

        def clear_slope(self, _checked=False, redraw=True):
            self.slope_start = None
            for graphic in self.slope_graphics:
                try:
                    graphic.remove()
                except (ValueError, AttributeError):
                    pass
            self.slope_graphics.clear()
            self.slope_label.setText("Slope: —")
            if redraw:
                self.canvas.draw_idle()

        def add_point_marker(self, info, point):
            line = info["line"]
            marker, = line.axes.plot(
                [line.get_xdata()[point]], [line.get_ydata()[point]], marker="o",
                markersize=8, markerfacecolor="white",
                markeredgecolor=line.get_color(), markeredgewidth=2,
                linestyle="none", zorder=50,
            )
            self.slope_graphics.append(marker)

        def plot_clicked(self, event):
            if not self.mode_combo.currentText().startswith("Measure") or event.button != 1:
                return
            if getattr(self.toolbar, "mode", ""):
                self.statusBar().showMessage(
                    "Turn off the toolbar zoom/pan tool before measuring a slope."
                )
                return
            picked = self.nearest_point(event)
            if not picked:
                self.statusBar().showMessage("Click closer to a plotted curve.")
                return
            info, point = picked

            if self.slope_start is None:
                self.clear_slope(redraw=False)
                self.slope_start = (info, point)
                self.add_point_marker(info, point)
                self.slope_label.setText("Slope: select end point")
                self.statusBar().showMessage(
                    f"Start selected on {info['experiment']['name']} ({info['side']} Y)."
                )
                self.canvas.draw_idle()
                return

            start_info, start_point = self.slope_start
            if info["line"] is not start_info["line"]:
                self.clear_slope(redraw=False)
                self.slope_start = (info, point)
                self.add_point_marker(info, point)
                self.slope_label.setText("Slope: select end point")
                self.statusBar().showMessage(
                    "Different curve selected; this point is now the new start."
                )
                self.canvas.draw_idle()
                return

            line = info["line"]
            x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
            first, last = sorted((start_point, point))
            xs, ys = x[first:last + 1], y[first:last + 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[valid], ys[valid]
            if len(xs) < 2 or np.unique(xs).size < 2:
                self.statusBar().showMessage(
                    "Select an interval containing at least two distinct X values."
                )
                return

            slope, intercept = np.polyfit(xs, ys, 1)
            fitted = slope * xs + intercept
            ss_res = float(np.sum((ys - fitted) ** 2))
            ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            self.add_point_marker(info, point)
            highlight, = line.axes.plot(
                xs, ys, color=line.get_color(), linewidth=4, alpha=0.55, zorder=45
            )
            order = np.argsort(xs)
            fit_line, = line.axes.plot(
                xs[order], fitted[order], color="#111111", linewidth=2,
                linestyle=":", zorder=51,
            )
            experiment = info["experiment"]
            x_unit = str(experiment["units"].get(info["x_column"], "")).strip()
            y_unit = str(experiment["units"].get(info["y_column"], "")).strip()
            slope_unit = (
                f" {y_unit}/{x_unit}"
                if x_unit not in ("", "–", "-") and y_unit not in ("", "–", "-")
                else ""
            )
            text = f"m = {slope:.6g}{slope_unit}\nR² = {r2:.4f}\nn = {len(xs)}"
            note = line.axes.annotate(
                text, xy=(xs[-1], fitted[-1]), xytext=(12, -52),
                textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=line.get_color(), alpha=0.96),
                arrowprops=dict(arrowstyle="->", color=line.get_color()), zorder=55,
            )
            self.slope_graphics.extend([highlight, fit_line, note])
            self.slope_label.setText(f"Slope: {slope:.6g}{slope_unit}")
            self.statusBar().showMessage(
                f"{experiment['name']} | {info['y_column']} vs {info['x_column']} | "
                f"slope={slope:.6g}{slope_unit}, R²={r2:.4f}, n={len(xs)}"
            )
            self.slope_start = None
            self.canvas.draw_idle()

    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv)
    window = MechanicalComparisonWindow()
    window.show()
    if initial_paths:
        window.load_paths(initial_paths)
    else:
        QTimer.singleShot(100, window.choose_files)
    app._mechanical_comparison_window = window
    return app.exec() if owns_application else 0


if __name__ == "__main__":
    raise SystemExit(run_gui(sys.argv[1:]))
