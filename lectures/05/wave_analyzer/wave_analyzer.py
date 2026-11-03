import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import butter, filtfilt


class WaveformPickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasonic Waveform P & S Arrival Picker")
        self.root.geometry("1080x780")

        # Data tracking variables
        self.folder_path = ""
        self.file_list = []
        self.current_file_idx = 0
        self.picks_data = {}  # Format: {filename: {'P': val, 'S': val}}

        self.p_arrival = None
        self.s_arrival = None
        self.p_line = None
        self.s_line = None

        # Crosshair lines
        self.ch_vline = None
        self.ch_hline = None

        # --- UI LAYOUT ---
        # Top Controls Frame
        control_frame = tk.Frame(self.root, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = tk.Button(
            control_frame,
            text="📁 Select Folder",
            command=self.select_directory,
            font=("Arial", 10, "bold"),
        )
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.lbl_file_info = tk.Label(
            control_frame, text="No directory selected", font=("Arial", 10)
        )
        self.lbl_file_info.pack(side=tk.LEFT, padx=10)

        # Adjustable Filter Cutoff Entry
        tk.Label(control_frame, text="Filter Cutoff (MHz):", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=(20, 2)
        )
        self.ent_cutoff = tk.Entry(control_frame, width=6)
        self.ent_cutoff.insert(0, "1.0")
        self.ent_cutoff.pack(side=tk.LEFT, padx=2)

        self.btn_refilter = tk.Button(
            control_frame, text="Apply Filter", command=self.load_waveform
        )
        self.btn_refilter.pack(side=tk.LEFT, padx=5)

        # Mode Selection & Next/Save Frame
        pick_frame = tk.Frame(self.root, pady=5)
        pick_frame.pack(side=tk.TOP, fill=tk.X)

        self.pick_mode = tk.StringVar(value="P")

        tk.Label(pick_frame, text="Active Pick Mode:").pack(
            side=tk.LEFT, padx=(10, 2)
        )
        rb_p = tk.Radiobutton(
            pick_frame,
            text="P-wave (Red)",
            variable=self.pick_mode,
            value="P",
            fg="red",
            font=("Arial", 10, "bold"),
        )
        rb_s = tk.Radiobutton(
            pick_frame,
            text="S-wave (Green)",
            variable=self.pick_mode,
            value="S",
            fg="green",
            font=("Arial", 10, "bold"),
        )
        rb_p.pack(side=tk.LEFT, padx=5)
        rb_s.pack(side=tk.LEFT, padx=5)

        self.lbl_picks = tk.Label(
            pick_frame,
            text="P: None µs | S: None µs",
            font=("Arial", 10, "bold"),
            fg="navy",
        )
        self.lbl_picks.pack(side=tk.LEFT, padx=20)

        self.btn_next = tk.Button(
            pick_frame,
            text="Next Waveform ➔",
            command=self.next_waveform,
            state=tk.DISABLED,
        )
        self.btn_next.pack(side=tk.RIGHT, padx=10)

        self.btn_save = tk.Button(
            pick_frame,
            text="💾 Save All Picks (.txt)",
            command=self.save_results,
            state=tk.DISABLED,
            bg="#2ecc71",
            fg="black",
        )
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # Container Frame for Navigation Toolbar + Plot Canvas
        plot_container = tk.Frame(self.root)
        plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Matplotlib Figure Setup
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.config(cursor="crosshair")

        # Matplotlib Navigation Toolbar (Zoom / Pan)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_container)
        self.toolbar.update()
        
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Event listeners for picking and dynamic crosshair movement
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def apply_lowpass_filter(self, raw_time, amp_vals, cutoff_freq_hz=1e6):
        """Applies a zero-phase 4th-order Butterworth low-pass filter with safe Nyquist handling."""
        if len(raw_time) < 4:
            return amp_vals

        dt = np.mean(np.diff(raw_time))
        if dt <= 0:
            return amp_vals

        dt_sec = dt if dt < 1e-3 else dt * 1e-6
        fs = 1.0 / dt_sec  # Sampling frequency in Hz
        nyquist = 0.5 * fs

        if cutoff_freq_hz >= nyquist:
            cutoff_freq_hz = 0.8 * nyquist

        normal_cutoff = cutoff_freq_hz / nyquist
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return amp_vals

        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, amp_vals)

    def select_directory(self):
        """Opens directory dialog and loads all valid files regardless of extension or lack thereof."""
        folder = filedialog.askdirectory()
        if not folder:
            return

        # Accepts any valid file (with or without extensions), excluding hidden OS files and output summary
        files = [
            f
            for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f))
            and not f.startswith(".")
            and f != "waveform_picks_summary.txt"
        ]

        if not files:
            messagebox.showerror(
                "Error", "No valid waveform files found in the selected folder."
            )
            return

        self.folder_path = folder
        self.file_list = files
        self.current_file_idx = 0
        self.picks_data = {f: {"P": None, "S": None} for f in self.file_list}

        self.btn_next.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

        self.load_waveform()

    def load_waveform(self):
        """Loads data (Col 0=Amplitude, Col 1=Time), filters, and plots horizontal waveform with small markers."""
        if not self.file_list:
            return

        current_file = self.file_list[self.current_file_idx]
        file_path = os.path.join(self.folder_path, current_file)

        self.lbl_file_info.config(
            text=f"File {self.current_file_idx + 1} of {len(self.file_list)}: {current_file}"
        )

        try:
            data = pd.read_csv(file_path, sep=None, engine="python", header=None)
            data = data.apply(pd.to_numeric, errors="coerce").dropna()

            # Column 0 = Amplitude, Column 1 = Time
            y_amp = data.iloc[:, 0].values
            raw_time = data.iloc[:, 1].values
        except Exception as e:
            messagebox.showerror("Read Error", f"Could not parse {current_file}.\n{e}")
            return

        try:
            cutoff_mhz = float(self.ent_cutoff.get())
            cutoff_hz = cutoff_mhz * 1e6
        except ValueError:
            cutoff_hz = 1e6

        if np.max(raw_time) < 1.0:
            time_us = raw_time * 1e6
        else:
            time_us = raw_time

        filtered_amp = self.apply_lowpass_filter(raw_time, y_amp, cutoff_freq_hz=cutoff_hz)

        # Plot: X = Time (µs), Y = Amplitude
        self.ax.clear()
        self.ax.plot(
            time_us, 
            filtered_amp, 
            color="#1f77b4", 
            linewidth=1.0, 
            marker="o", 
            markersize=2.5, 
            label="Filtered Signal"
        )
        self.ax.set_title(f"Waveform: {current_file} ({cutoff_hz/1e6:.2f} MHz Low-Pass Filtered)", fontsize=11)
        self.ax.set_xlabel("Time (µs)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # Interactive crosshair lines
        self.ch_vline = self.ax.axvline(
            x=time_us[0], color="gray", linestyle=":", linewidth=0.8, visible=False
        )
        self.ch_hline = self.ax.axhline(
            y=0, color="gray", linestyle=":", linewidth=0.8, visible=False
        )

        # Restore existing picks if returning to this file
        self.p_arrival = self.picks_data[current_file]["P"]
        self.s_arrival = self.picks_data[current_file]["S"]

        self.p_line = None
        self.s_line = None

        self.update_pick_lines()
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Updates crosshair lines as mouse moves inside plot area."""
        if self.ch_vline is None or self.ch_hline is None:
            return

        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.ch_vline.set_xdata([event.xdata, event.xdata])
            self.ch_hline.set_ydata([event.ydata, event.ydata])
            self.ch_vline.set_visible(True)
            self.ch_hline.set_visible(True)
            self.canvas.draw_idle()
        elif self.ch_vline.get_visible():
            self.ch_vline.set_visible(False)
            self.ch_hline.set_visible(False)
            self.canvas.draw_idle()

    def on_plot_click(self, event):
        """Captures arrival time click along X-axis (Time)."""
        if event.inaxes != self.ax or event.xdata is None or self.toolbar.mode != "":
            return

        clicked_time = event.xdata
        current_file = self.file_list[self.current_file_idx]

        if self.pick_mode.get() == "P":
            self.p_arrival = clicked_time
            self.picks_data[current_file]["P"] = clicked_time
        else:
            self.s_arrival = clicked_time
            self.picks_data[current_file]["S"] = clicked_time

        self.update_pick_lines()

    def update_pick_lines(self):
        """Redraws vertical pick lines (Time on X-axis)."""
        if self.p_arrival is not None:
            if self.p_line in self.ax.lines:
                self.p_line.remove()
            self.p_line = self.ax.axvline(
                x=self.p_arrival, color="red", linestyle="--", linewidth=1.8, label="P-wave"
            )

        if self.s_arrival is not None:
            if self.s_line in self.ax.lines:
                self.s_line.remove()
            self.s_line = self.ax.axvline(
                x=self.s_arrival, color="green", linestyle="--", linewidth=1.8, label="S-wave"
            )

        p_str = f"{self.p_arrival:.2f} µs" if self.p_arrival is not None else "None"
        s_str = f"{self.s_arrival:.2f} µs" if self.s_arrival is not None else "None"
        self.lbl_picks.config(text=f"P Arrival: {p_str} | S Arrival: {s_str}")

        self.canvas.draw_idle()

    def next_waveform(self):
        """Advances to the next waveform file in sequence."""
        if self.current_file_idx < len(self.file_list) - 1:
            self.current_file_idx += 1
            self.load_waveform()
        else:
            messagebox.showinfo(
                "End of Directory", "You have reached the last waveform file."
            )

    def save_results(self):
        """Saves all picked arrival times to a summary TXT file."""
        if not self.folder_path:
            return

        save_path = os.path.join(self.folder_path, "waveform_picks_summary.txt")

        try:
            with open(save_path, "w") as f:
                f.write("Filename\tP_arrival_us\tS_arrival_us\n")
                for filename, picks in self.picks_data.items():
                    p_val = f"{picks['P']:.4f}" if picks["P"] is not None else "NaN"
                    s_val = f"{picks['S']:.4f}" if picks["S"] is not None else "NaN"
                    f.write(f"{filename}\t{p_val}\t{s_val}\n")

            messagebox.showinfo(
                "File Saved", f"Picks saved successfully to:\n{save_path}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write file:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformPickerApp(root)
    root.mainloop()