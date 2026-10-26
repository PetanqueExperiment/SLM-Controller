"""Last-sent hologram preview, its Fourier intensity, and temperature label."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from temperature_log import DailyTemperatureLog


class LastSentPreviewPanel(QtWidgets.QWidget):
    """Owns this panel's widgets and handlers. Shared phase state is on ``self.gui``."""

    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self.gui = gui
        self._ft_auto_levels = True
        self._ft_levels = None  # (lo, hi) while manual
        self._ft_level_edits_guard = False
        self._ft_view_auto = True  # False after user zooms/pans; Home restores
        self._build()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.root)

    def _build(self):
        """Side-by-side: last hologram sent (left) and |FFT|² (right)."""
        self.root = pg.LayoutWidget()
        self.last_sent_preview_layout = self.root

        self.label_ft_preview = QtWidgets.QLabel("Fourier transform of hologram (|FFT|²)")
        self.label_ft_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.label_last_sent_preview = QtWidgets.QLabel("Last hologram sent to SLM")
        self.label_last_sent_preview.setAlignment(QtCore.Qt.AlignCenter)

        self.label_slm_temperature = QtWidgets.QLabel("SLM temperature: —")
        self.label_slm_temperature.setAlignment(QtCore.Qt.AlignCenter)
        _temp_font = self.label_slm_temperature.font()
        _temp_font.setBold(True)
        self.label_slm_temperature.setFont(_temp_font)

        try:
            _grey_cmap = pg.colormap.get("grey", source="matplotlib", skipCache=True)
        except Exception:
            _grey_cmap = pg.colormap.get("grey")

        # --- Fourier intensity of the sent hologram ---
        self.ft_plot = pg.PlotWidget()
        self.ft_plot.setAspectLocked(True)
        self.ft_plot.hideButtons()
        self.ft_plot.setMenuEnabled(False)
        for _ax in ("left", "bottom", "right", "top"):
            self.ft_plot.plotItem.hideAxis(_ax)
        self.ft_img_item = pg.ImageItem()
        self.ft_plot.addItem(self.ft_img_item)
        _ft_vb = self.ft_plot.getViewBox()
        _ft_vb.setMouseEnabled(True, True)
        # Rubber-band zoom (matplotlib-like); switch to Pan via the Pan button
        if hasattr(_ft_vb, "setMouseMode"):
            _ft_vb.setMouseMode(_ft_vb.RectMode)
        if hasattr(_ft_vb, "disableAutoRange"):
            _ft_vb.disableAutoRange()
        else:
            _ft_vb.enableAutoRange(enable=False)
        self.ft_plot.plotItem.setClipToView(False)
        # User zoom/pan → stop auto-fitting the view on each hologram update
        if hasattr(_ft_vb, "sigRangeChangedManually"):
            _ft_vb.sigRangeChangedManually.connect(self._on_ft_view_changed_manually)
        elif hasattr(_ft_vb, "sigRangeChanged"):
            _ft_vb.sigRangeChanged.connect(self._on_ft_view_changed_manually)
        _ft_cb_kw = dict(colorMap=_grey_cmap, interactive=True)
        try:
            self.ft_colorbar = pg.ColorBarItem(colorMapMenu=False, **_ft_cb_kw)
        except TypeError:
            self.ft_colorbar = pg.ColorBarItem(**_ft_cb_kw)
        self.ft_colorbar.setImageItem(self.ft_img_item, insert_in=self.ft_plot.plotItem)
        self._connect_ft_colorbar_levels()
        self.ft_plot.setMinimumSize(200, 220)
        self.ft_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        # Manual / auto level controls for the FT colorbar
        self.ft_levels_row = QtWidgets.QWidget()
        _ft_lvl = QtWidgets.QHBoxLayout(self.ft_levels_row)
        _ft_lvl.setContentsMargins(0, 0, 0, 0)
        _ft_lvl.setSpacing(4)
        _ft_lvl.addWidget(QtWidgets.QLabel("min"))
        self.ft_vmin_edit = QtWidgets.QLineEdit()
        self.ft_vmin_edit.setPlaceholderText("min")
        self.ft_vmin_edit.setMaximumWidth(100)
        _ft_lvl.addWidget(self.ft_vmin_edit)
        _ft_lvl.addWidget(QtWidgets.QLabel("max"))
        self.ft_vmax_edit = QtWidgets.QLineEdit()
        self.ft_vmax_edit.setPlaceholderText("max")
        self.ft_vmax_edit.setMaximumWidth(100)
        _ft_lvl.addWidget(self.ft_vmax_edit)
        self.ft_auto_levels_btn = QtWidgets.QPushButton("Auto")
        self.ft_auto_levels_btn.setToolTip("Reset color limits from off-center percentiles")
        self.ft_auto_levels_btn.setFixedWidth(48)
        _ft_lvl.addWidget(self.ft_auto_levels_btn)
        _ft_lvl.addSpacing(8)
        self.ft_zoom_btn = QtWidgets.QPushButton("Zoom")
        self.ft_zoom_btn.setCheckable(True)
        self.ft_zoom_btn.setChecked(True)
        self.ft_zoom_btn.setToolTip("Rectangle zoom (drag on the FT plot), like matplotlib")
        self.ft_zoom_btn.setFixedWidth(48)
        self.ft_pan_btn = QtWidgets.QPushButton("Pan")
        self.ft_pan_btn.setCheckable(True)
        self.ft_pan_btn.setToolTip("Pan the FT view (drag on the plot)")
        self.ft_pan_btn.setFixedWidth(40)
        self.ft_home_btn = QtWidgets.QPushButton("Home")
        self.ft_home_btn.setToolTip("Reset FT view to the full image")
        self.ft_home_btn.setFixedWidth(48)
        _ft_lvl.addWidget(self.ft_zoom_btn)
        _ft_lvl.addWidget(self.ft_pan_btn)
        _ft_lvl.addWidget(self.ft_home_btn)
        _ft_lvl.addStretch(1)
        self.ft_vmin_edit.returnPressed.connect(self._apply_ft_levels_from_edits)
        self.ft_vmax_edit.returnPressed.connect(self._apply_ft_levels_from_edits)
        self.ft_auto_levels_btn.clicked.connect(self._auto_ft_levels_clicked)
        self.ft_zoom_btn.clicked.connect(lambda _checked: self._set_ft_mouse_mode("zoom"))
        self.ft_pan_btn.clicked.connect(lambda _checked: self._set_ft_mouse_mode("pan"))
        self.ft_home_btn.clicked.connect(self._ft_home_view)

        # --- Hologram image ---
        self.last_sent_plot = pg.PlotWidget()
        self.last_sent_plot.setAspectLocked(True)
        self.last_sent_plot.hideButtons()
        self.last_sent_plot.setMenuEnabled(False)
        for _ax in ("left", "bottom", "right", "top"):
            self.last_sent_plot.plotItem.hideAxis(_ax)
        self.last_sent_img_item = pg.ImageItem()
        self.last_sent_plot.addItem(self.last_sent_img_item)
        _vb = self.last_sent_plot.getViewBox()
        _vb.setMouseEnabled(False, False)
        if hasattr(_vb, "disableAutoRange"):
            _vb.disableAutoRange()
        else:
            _vb.enableAutoRange(enable=False)
        self.last_sent_plot.plotItem.setClipToView(False)
        _cb_kw = dict(
            values=(0, 255),
            colorMap=_grey_cmap,
            interactive=False,
            limits=(0, 255),
        )
        try:
            self.last_sent_colorbar = pg.ColorBarItem(colorMapMenu=False, **_cb_kw)
        except TypeError:
            self.last_sent_colorbar = pg.ColorBarItem(**_cb_kw)
        self.last_sent_colorbar.setImageItem(
            self.last_sent_img_item,
            insert_in=self.last_sent_plot.plotItem,
        )
        self.last_sent_plot.setMinimumSize(200, 220)
        self.last_sent_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        self.last_sent_preview_layout.addWidget(self.label_last_sent_preview, row=0, col=0)
        self.last_sent_preview_layout.addWidget(self.label_ft_preview, row=0, col=1)
        self.last_sent_preview_layout.addWidget(self.last_sent_plot, row=1, col=0)
        self.last_sent_preview_layout.addWidget(self.ft_plot, row=1, col=1)
        self.last_sent_preview_layout.addWidget(self.ft_levels_row, row=2, col=1)
        self.last_sent_preview_layout.addWidget(self.label_slm_temperature, row=3, col=0, colspan=2)

    def _start_slm_temperature_monitor(self):
        """Poll Read_SLM_temperature once per second and log each sample."""
        self.gui._temperature_log = DailyTemperatureLog()
        self.gui._temp_log_error_reported = False
        self.gui._slm_temp_timer = QtCore.QTimer(self)
        self.gui._slm_temp_timer.setInterval(1000)
        self.gui._slm_temp_timer.timeout.connect(self._refresh_slm_temperature)
        self._refresh_slm_temperature()
        self.gui._slm_temp_timer.start()

    def _refresh_slm_temperature(self):
        comm = getattr(getattr(self.gui, "SLM", None), "communication", None)
        if comm is None:
            self.label_slm_temperature.setText("SLM temperature: —")
            return
        try:
            temp_c = comm.read_temperature()
        except Exception:
            self.label_slm_temperature.setText("SLM temperature: unavailable")
            return
        self.label_slm_temperature.setText(f"SLM temperature: {temp_c:.1f} °C")
        try:
            self.gui._temperature_log.append(temp_c)
        except Exception as exc:
            if not self.gui._temp_log_error_reported:
                self.gui._temp_log_error_reported = True
                print(f"Temperature log write failed: {exc}")

    def _fit_view_to_image(self, plot, arr):
        """Lock a plot view to the full array so resize does not crop the image."""
        arr = np.asarray(arr)
        if arr.ndim != 2:
            return
        ny, nx = arr.shape
        plot.getViewBox().setRange(xRange=(0, nx), yRange=(0, ny), padding=0)

    def _fit_last_sent_view_to_image(self):
        if not hasattr(self.gui, "totalPhase"):
            return
        self._fit_view_to_image(self.last_sent_plot, self.gui.totalPhase)

    def _on_ft_view_changed_manually(self, *args):
        self._ft_view_auto = False

    def _set_ft_mouse_mode(self, mode):
        """mode: 'zoom' (rectangle) or 'pan'."""
        vb = self.ft_plot.getViewBox()
        if mode == "pan":
            self.ft_pan_btn.setChecked(True)
            self.ft_zoom_btn.setChecked(False)
            if hasattr(vb, "setMouseMode"):
                vb.setMouseMode(vb.PanMode)
        else:
            self.ft_zoom_btn.setChecked(True)
            self.ft_pan_btn.setChecked(False)
            if hasattr(vb, "setMouseMode"):
                vb.setMouseMode(vb.RectMode)

    def _ft_home_view(self):
        """Reset FT view to the full |FFT|² image (matplotlib Home)."""
        self._ft_view_auto = True
        img = self.ft_img_item.image
        if img is not None:
            self._fit_view_to_image(self.ft_plot, img)
        elif hasattr(self.gui, "totalPhase"):
            self._fit_view_to_image(self.ft_plot, self.gui.totalPhase)

    def _hologram_fft_intensity(self, hologram_u8):
        """
        Optical far-field intensity from the uint8 SLM hologram:

            phi = 2π · U / P ,   I = |FFT{ exp(i φ) }|²  (fftshifted)
        """
        arr = np.asarray(hologram_u8)
        if arr.ndim != 2:
            return None
        p = float(getattr(self.gui, "param_P", 255) or 255)
        if p <= 0:
            p = 255.0
        phase = arr.astype(np.float64) * (2.0 * np.pi / p)
        field = np.exp(1j * phase)
        ft = np.fft.fftshift(np.fft.fft2(field))
        return np.abs(ft) ** 2

    @staticmethod
    def _off_center_mask(shape, radius_frac=0.03):
        """Boolean mask True away from a disk around the FFT origin (zero order)."""
        ny, nx = shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.ogrid[:ny, :nx]
        r0 = radius_frac * min(ny, nx)
        return (yy - cy) ** 2 + (xx - cx) ** 2 > r0 ** 2

    def _compute_ft_auto_levels(self, intensity):
        """Percentile limits from off-center pixels (avoids zero-order dominance)."""
        mask = self._off_center_mask(intensity.shape)
        vals = intensity[mask] if mask.any() else intensity.ravel()
        if vals.size < 16:
            vals = intensity.ravel()
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(intensity)), float(np.max(intensity))
            if hi <= lo:
                hi = lo + 1.0
        return max(0.0, lo), hi

    def _connect_ft_colorbar_levels(self):
        """Keep spin/line edits and stored levels in sync when the bar is dragged."""
        for sig_name in ("sigLevelsChangeFinished", "sigLevelsChanged"):
            sig = getattr(self.ft_colorbar, sig_name, None)
            if sig is not None:
                sig.connect(self._on_ft_colorbar_levels)
                return

    def _on_ft_colorbar_levels(self, *args):
        levels = None
        if args:
            try:
                levels = (float(args[0][0]), float(args[0][1]))
            except Exception:
                levels = None
        if levels is None:
            levels = self._read_ft_levels_from_widgets()
        if levels is None:
            return
        lo, hi = levels
        if hi <= lo:
            return
        self._ft_auto_levels = False
        self._ft_levels = (lo, hi)
        self._sync_ft_level_edits(lo, hi)

    def _read_ft_levels_from_widgets(self):
        try:
            levels = self.ft_colorbar.levels()
            return float(levels[0]), float(levels[1])
        except Exception:
            pass
        try:
            levels = self.ft_img_item.levels
            return float(levels[0]), float(levels[1])
        except Exception:
            return None

    def _sync_ft_level_edits(self, lo, hi):
        self._ft_level_edits_guard = True
        try:
            self.ft_vmin_edit.setText("%.6g" % lo)
            self.ft_vmax_edit.setText("%.6g" % hi)
        finally:
            self._ft_level_edits_guard = False

    def _apply_ft_levels(self, lo, hi):
        if hi <= lo:
            return
        self._ft_auto_levels = False
        self._ft_levels = (float(lo), float(hi))
        self.ft_img_item.setLevels([lo, hi])
        try:
            self.ft_colorbar.setLevels((lo, hi), update_items=True)
        except TypeError:
            self.ft_colorbar.setLevels((lo, hi))
        self._sync_ft_level_edits(lo, hi)

    def _apply_ft_levels_from_edits(self):
        if self._ft_level_edits_guard:
            return
        try:
            lo = float(self.ft_vmin_edit.text().strip())
            hi = float(self.ft_vmax_edit.text().strip())
        except ValueError:
            print("FT colorbar: enter numeric min/max")
            return
        if hi <= lo:
            print("FT colorbar: max must be greater than min")
            return
        self._apply_ft_levels(lo, hi)

    def _auto_ft_levels_clicked(self):
        self._ft_auto_levels = True
        self._ft_levels = None
        if hasattr(self.gui, "totalPhase"):
            self._update_ft_preview(self.gui.totalPhase)

    def _update_ft_preview(self, hologram_u8):
        """Show |FFT|² on a linear color scale (manual limits preserved until Auto)."""
        intensity = self._hologram_fft_intensity(hologram_u8)
        if intensity is None:
            return

        if self._ft_auto_levels or self._ft_levels is None:
            lo, hi = self._compute_ft_auto_levels(intensity)
            self._ft_levels = (lo, hi)
        else:
            lo, hi = self._ft_levels

        self.ft_img_item.setAutoLevels(False)
        self.ft_img_item.setImage(intensity, autoLevels=False, levels=[lo, hi])
        try:
            self.ft_colorbar.setLevels((lo, hi), update_items=True)
        except TypeError:
            self.ft_colorbar.setLevels((lo, hi))
        self._sync_ft_level_edits(lo, hi)
        if self._ft_view_auto:
            self._fit_view_to_image(self.ft_plot, intensity)

    def _update_last_sent_preview(self):
        """Show the buffer last passed to the SLM and its Fourier intensity."""
        arr = np.asarray(self.gui.totalPhase)
        self.last_sent_img_item.setAutoLevels(False)
        self.last_sent_img_item.setImage(arr, autoLevels=False, levels=[0, 255])
        self.last_sent_colorbar.setLevels((0, 255), update_items=True)
        self._fit_last_sent_view_to_image()
        self._update_ft_preview(arr)

    def eventFilter(self, watched, event):
        """Keep image previews scaled to the full frame when a plot is resized."""
        if event.type() != QtCore.QEvent.Resize:
            return super().eventFilter(watched, event)
        if watched is self.last_sent_plot:
            QtCore.QTimer.singleShot(0, self._fit_last_sent_view_to_image)
        elif watched is self.ft_plot and self._ft_view_auto:
            img = self.ft_img_item.image
            if img is not None:
                QtCore.QTimer.singleShot(0, lambda: self._fit_view_to_image(self.ft_plot, img))
        return super().eventFilter(watched, event)
