"""
This module links user interactions to the responses of the ifcview software.
"""

import os
from contextlib import contextmanager
import h5py
import hdf5plugin  # noqa: F401  (registers HDF5 compression plugins on import)
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui, run
import ifcview.lib.rendering as re
import ifcview.lib.utilities as util
from ifcview.lib.rendering import GuiRendering, FilePicker, FileSaver


class GuiInteraction(GuiRendering):
    """Wire user actions to GUI responses for the ifcview IFC viewer.

    Builds on :class:`GuiRendering` (which creates the widgets) and adds the
    behaviour: pick/open a file, list its experiments, scan an experiment for
    cells and browse them, look up an event by id, and display/save the
    selected channel image. Each method carries its own docstring; the central
    pieces of state are ``current_state`` (the tuple the display timer diffs
    against to decide when to re-render), ``image``/``image_norm`` (the shown
    image and its contrast-adjusted copy), and ``browse_cells`` (the cached
    per-experiment cell scan that paging slices through).
    """

    def __init__(self):
        super().__init__()
        # Captured at page-construction time (valid context) so deferred
        # handlers have a reliable client reference to await connection on,
        # without depending on ui.context being correct when they run.
        self.client = ui.context.client
        self.select_file_button.on("click", self.pick_file)
        self.save_image_button.on("click", self.save_image)
        self.reset_button.on("click", self.reset_min_max)
        self.tab_one.on("click", self.__select_tab_one)
        self.tab_two.on("click", self.__select_tab_two)
        self.open_button.on("click", self.open_image)
        self.close_file_button.on("click", self.close_file)
        self.event_input.on("keydown.enter", self.open_image)
        # Scanning is driven solely by the experiment select's value-change
        # event (fires for both user picks and load_file's programmatic
        # selection). The marker below collapses transient value changes into a
        # single scan per experiment (see on_experiment_change).
        self._last_scanned_experiment = None
        self.experiment_select.on_value_change(
            lambda e: self.on_experiment_change())
        self.prev_button.on("click", self.browse_prev)
        self.next_button.on("click", self.browse_next)
        self.channel_select.on("update:model-value",
                               lambda e: self.on_channel_change())
        # Mask overlay: toggling or switching run re-renders. The run dropdown
        # is always shown alongside the toggle but is interactive only while the
        # overlay is on; show_data diffs both values (see its state tuple) so the
        # change reaches display_image.
        self.mask_toggle.on("update:model-value",
                            lambda e: self.on_mask_toggle())
        # Arrow keys: left/right cycle through the browsed images, up/down
        # cycle through the channel layers. The keyboard ignores typing in
        # inputs/selects so the event-id field still works normally.
        self.keyboard = ui.keyboard(on_key=self.handle_key)
        self.current_state, self.image, self.image_norm = None, None, None
        # Coloured RGBA overlay for the currently displayed event (or None when
        # the overlay is off / unavailable); reused by the saved view.
        self.mask = None
        self.timer = ui.timer(re.UPDATE_RATE, lambda: self.show_data())
        self.selected_tab = 1
        self.last_folder = ""
        self.fig, self.ax = None, None
        self.browse_page = 0
        # Cells found by the single per-experiment scan (capped at
        # MAX_BROWSE_PAGES pages). Paging just slices this list -- no rescans.
        self.browse_cells = []
        self.browse_page_size = re.BROWSE_PAGE_SIZE
        self.channel_labels = []
        # Ref-count for the shared overlay so overlapping scans (rapid
        # experiment switches) keep it open until the last one finishes.
        self._overlay_depth = 0

    def __select_tab_one(self):
        self.selected_tab = 1

    def __select_tab_two(self):
        self.selected_tab = 2

    async def pick_file(self) -> None:
        """To pick a file when click the button 'Select file' """
        config_data = util.load_config()
        if config_data is None:
            self.last_folder = ""
        else:
            try:
                self.last_folder = config_data["last_folder"]
            except KeyError:
                self.last_folder = ""
        if (self.last_folder == "") or (not os.path.exists(self.last_folder)):
            file_path = await FilePicker("~",
                                         allowed_extensions=re.INPUT_EXT)
        else:
            file_path = await FilePicker(self.last_folder,
                                         allowed_extensions=re.INPUT_EXT)
        if file_path:
            self.last_folder = os.path.dirname(file_path)
            config_data = {'last_folder': self.last_folder}
            util.save_config(config_data)
            await self.load_file(file_path)

    async def load_file(self, file_path):
        """Open a hdf file: list its experiments and enable lookup/browse.

        The HDF reads run in a *separate process* (``run.cpu_bound``), not a
        thread: h5py holds the GIL while reading, so a thread-based scan would
        starve the asyncio event loop and the client would drop with "Connection
        lost" and reload. The modal spinner covering the per-experiment work
        (channel refresh + cell scan) is owned by ``on_experiment_change``,
        which the experiment selection below triggers; the quick top-level read
        here is left uncovered as it returns near-instantly.
        """
        file_path = file_path.replace("\\", "/")
        experiments = await run.cpu_bound(util.list_experiments, file_path)
        if isinstance(experiments, str):
            ui.notify("Error reading file: " + experiments)
            return
        if not experiments:
            ui.notify("No experiments (groups containing an 'events' group) "
                      "found in the file.")
            return
        self.reset()
        self.file_path_display.set_text(file_path)
        self.hdf_key_display.set_text("")
        self.lookup_panel.set_visibility(True)
        self.browse_panel.set_visibility(True)
        # Selecting the first experiment triggers on_experiment_change, which
        # refreshes the channels and runs the single cell scan. Reset the
        # de-dup marker and clear the value first so the scan still fires when
        # the same experiment name is selected again (e.g. reopening a file).
        self._last_scanned_experiment = None
        self.experiment_select.set_options(experiments, value=None)
        self.experiment_select.set_value(next(iter(experiments)))

    @contextmanager
    def loading_overlay(self, message):
        """Show the reusable modal spinner for the duration of a block.

        The overlay element is built once at page-construction time (see
        ``GuiRendering.init_gui``), so it always belongs to this page's client
        and shows reliably even when toggled from a deferred value-change
        handler. The dialog is always closed on exit, even if the block raises.
        It is ref-counted so two overlapping uses (e.g. quick experiment
        switches) keep it open until the outermost one exits.
        """
        self.loading_label.set_text(message)
        self._overlay_depth += 1
        self.loading_dialog.open()
        try:
            yield
        finally:
            self._overlay_depth -= 1
            if self._overlay_depth <= 0:
                self._overlay_depth = 0
                self.loading_dialog.close()

    def reset_browse_paging(self):
        """Return browsing to the first page and drop the cached cells."""
        self.browse_page = 0
        self.browse_cells = []

    def close_file(self):
        """Close the current file and hide the lookup/browse panels."""
        self.reset()
        self.file_path_display.set_text("")
        self.experiment_select.set_options([])
        self.channel_select.set_options({})
        self.mask_run_select.set_options({})
        self.mask_toggle.set_value(False)
        self.channel_labels = []
        self.browse_container.clear()
        self.browse_info.set_text("")
        self.reset_browse_paging()
        self._last_scanned_experiment = None
        self.lookup_panel.set_visibility(False)
        self.browse_panel.set_visibility(False)

    async def on_experiment_change(self):
        """Refresh channels and run the single cell scan for the selected
        experiment.

        Bound to the select's value-change event, which fires both when the
        user picks an experiment and when ``load_file`` selects the first one.
        The handler runs deferred, so several transient value changes (e.g. the
        clear->set in ``load_file``) can be queued together; the de-dup marker
        collapses them into one scan of the experiment finally selected. This is
        the ONLY place a cell scan is started -- so it happens exactly once per
        new experiment (and on file open, via ``load_file``'s selection).
        """
        experiment = self.experiment_select.value
        if not experiment or experiment == self._last_scanned_experiment:
            return
        self._last_scanned_experiment = experiment
        self.reset_browse_paging()
        # Buffer the WHOLE wait (channel refresh + cell scan, both worker reads)
        # behind one modal overlay so there is no gap with no feedback, and show
        # an in-list placeholder for good measure. The modal also blocks stray
        # clicks while the scan runs.
        self.browse_container.clear()
        self.browse_info.set_text("Scanning…")
        self.prev_button.disable()
        self.next_button.disable()
        with self.loading_overlay("Scanning experiment for cells…"):
            # Wait for the socket so the spinner actually paints before the
            # (potentially slow) worker reads begin; best-effort if already up.
            try:
                await self.client.connected()
            except Exception:
                pass
            try:
                await self.refresh_channels()
                await self.refresh_mask_runs()
                await self.scan_browse()
            except Exception as error:
                self.browse_cells = []
                self.render_browse_page()
                ui.notify("Could not scan experiment: {}".format(error))

    async def refresh_channels(self):
        """Populate the channel dropdown from the experiment's attributes.

        Builds the per-layer channel names (e.g. "Ch02 FITC PAC-1") and sets
        the colormap for the default channel (channel 1 -> grey).
        """
        file_path = self.file_path_display.text
        experiment = self.experiment_select.value
        if not file_path or not experiment:
            return
        labels = await run.cpu_bound(util.get_channel_labels, file_path,
                                     experiment)
        if self.experiment_select.value != experiment:
            return  # selection changed mid-read; a newer task owns the UI
        self.channel_labels = labels
        options = {i: lab for i, lab in enumerate(labels)}
        self.channel_select.set_options(options,
                                        value=0 if labels else None)
        if labels:
            self.cmap_list.set_value(util.channel_colormap(labels[0], 0))

    def on_channel_change(self):
        """Auto-set the colormap when the selected channel changes."""
        idx = self.channel_select.value
        if idx is None:
            return
        label = self.channel_labels[idx] if idx < len(self.channel_labels) \
            else ""
        self.cmap_list.set_value(util.channel_colormap(label, idx))

    def on_mask_toggle(self):
        """Enable/disable the run dropdown when the overlay toggle changes.

        The actual re-render is driven by ``show_data`` (the toggle value is
        part of its diffed state), so this only manages the run dropdown's
        interactivity.
        """
        self.update_mask_run_interactivity()

    def update_mask_run_interactivity(self):
        """Make the mask-run dropdown interactive only while the overlay is on.

        The dropdown stays visible alongside the toggle (its visibility is owned
        by ``enable_ui_image``); here we only enable it when the overlay is on
        and disable it otherwise.
        """
        if self.mask_toggle.value:
            self.mask_run_select.enable()
        else:
            self.mask_run_select.disable()

    async def refresh_mask_runs(self):
        """Populate the mask-run dropdown for the selected experiment.

        Lists the run subgroups under the experiment's ``masks`` group and
        selects the first by default. When none exist the toggle is reset off.
        The toggle/dropdown visibility is owned by ``enable_ui_image`` (shown
        only while an image with available masks is displayed).
        """
        file_path = self.file_path_display.text
        experiment = self.experiment_select.value
        if not file_path or not experiment:
            return
        runs = await run.cpu_bound(util.list_mask_runs, file_path, experiment)
        if self.experiment_select.value != experiment:
            return  # selection changed mid-read; a newer task owns the UI
        if runs:
            self.mask_run_select.set_options(runs, value=runs[0])
        else:
            self.mask_run_select.set_options({})
            self.mask_toggle.set_value(False)

    async def scan_browse(self):
        """Scan the current experiment ONCE for cells and cache them.

        Collects up to ``MAX_BROWSE_PAGES`` pages worth of cells in a single
        bounded scan (it stops as soon as that many cells are found, or once
        ``CELL_FILTER_SCAN_LIMIT`` events have been probed -- whichever comes
        first), so the preview stays fast. Paging then just slices the cache,
        with no further scanning. Noise/empty events are skipped by reading a
        sub-sampled brightfield layer (see ``util.list_cell_events_page``). The
        caller (``on_experiment_change``) owns the loading overlay around this.
        """
        file_path = self.file_path_display.text
        experiment = self.experiment_select.value
        if not file_path or not experiment:
            return
        max_cells = self.browse_page_size * re.MAX_BROWSE_PAGES
        names, _has_more, _next_id = await run.cpu_bound(
            util.list_cell_events_page, file_path, experiment,
            0, max_cells,
            re.CELL_FILTER_CHANNEL, re.CELL_CONTRAST_THRESHOLD,
            re.CELL_FILTER_SCAN_LIMIT)
        if self.experiment_select.value != experiment:
            return  # selection changed mid-scan; a newer task owns the UI
        self.browse_cells = names
        self.browse_page = 0
        self.render_browse_page()

    def render_browse_page(self):
        """Render the current page from the cached cell list (no scanning)."""
        experiment = self.experiment_select.value
        size = self.browse_page_size
        start = self.browse_page * size
        page_names = self.browse_cells[start:start + size]
        self.browse_container.clear()
        with self.browse_container:
            for name in page_names:
                key = util.event_key(experiment, name)
                ui.item(name, on_click=lambda k=key: self.open_key(k))
        if page_names:
            self.browse_info.set_text(f"Page {self.browse_page + 1}")
        else:
            self.browse_info.set_text("No cells found")
        if self.browse_page > 0:
            self.prev_button.enable()
        else:
            self.prev_button.disable()
        # More cached cells beyond this page? (Capped at MAX_BROWSE_PAGES.)
        if start + size < len(self.browse_cells):
            self.next_button.enable()
        else:
            self.next_button.disable()

    def browse_prev(self):
        """Show the previous cached page."""
        if self.browse_page > 0:
            self.browse_page -= 1
            self.render_browse_page()

    def browse_next(self):
        """Show the next cached page (no further scanning)."""
        if (self.browse_page + 1) * self.browse_page_size \
                < len(self.browse_cells):
            self.browse_page += 1
            self.render_browse_page()

    def handle_key(self, e):
        """Map arrow keys to image/channel cycling.

        Left/Right step through the browsed cell images; Up/Down step through
        the channel layers. Both wrap around.
        """
        if not e.action.keydown:
            return
        if e.key.arrow_left:
            self.cycle_image(-1)
        elif e.key.arrow_right:
            self.cycle_image(1)
        elif e.key.arrow_up:
            self.cycle_channel(-1)
        elif e.key.arrow_down:
            self.cycle_channel(1)

    def cycle_image(self, delta):
        """Open the image ``delta`` steps away in the browsed cell list."""
        cells = self.browse_cells
        if not cells:
            return
        experiment = self.experiment_select.value
        # The displayed key ends in the event id (e.g. ".../events/42").
        key = self.hdf_key_display.text
        name = key.rsplit("/", 1)[-1] if key else ""
        if name in cells:
            idx = (cells.index(name) + delta) % len(cells)
        else:
            idx = 0
        self.open_key(util.event_key(experiment, cells[idx]))

    def cycle_channel(self, delta):
        """Select the channel layer ``delta`` steps away (wraps around)."""
        n = len(self.channel_labels)
        if n == 0:
            return
        current = self.channel_select.value
        idx = (int(current) + delta) % n if current is not None else 0
        self.channel_select.set_value(idx)
        self.on_channel_change()

    async def open_image(self):
        """Resolve and open an event by experiment / event id. The channel
        dropdown in the toolbar selects which layer is shown."""
        file_path = self.file_path_display.text
        experiment = self.experiment_select.value
        event_id = self.event_input.value
        if not file_path or not experiment:
            ui.notify("Select a file and experiment first.")
            return
        if event_id is None or str(event_id).strip() == "":
            ui.notify("Enter an event ID.")
            return
        key = await run.cpu_bound(util.resolve_dataset_key, file_path,
                                 experiment, event_id)
        if key is None:
            ui.notify("No event '{}' found in {}.".format(event_id,
                                                          experiment))
            return
        self.open_key(key)

    def open_key(self, key):
        """Select a dataset key; the display timer renders it."""
        self.hdf_key_display.set_text(key)

    def disable_sliders(self):
        """Disable and reset values of the contrast sliders"""
        self.min_slider.set_value(0)
        self.min_slider.disable()
        self.max_slider.set_value(255)
        self.max_slider.disable()

    def enable_ui_image(self):
        """Enable UI-elements for displaying a 2d dataset as an image."""
        self.cmap_list.enable()
        self.min_slider.enable()
        self.max_slider.enable()
        self.main_plot.set_visibility(True)
        self.save_image_button.enable()
        self.histogram_plot.set_visibility(True)
        self.image_info_table.set_visibility(True)
        # The mask overlay toggle and its run dropdown are shown together
        # whenever an image is displayed and the experiment has at least one
        # mask run. (reset() hides them while no image is shown, so their
        # visibility has to be re-asserted here.) The run dropdown stays visible
        # regardless of the toggle but is only interactive when the overlay is
        # on.
        has_runs = bool(self.mask_run_select.options)
        self.mask_toggle.set_visibility(has_runs)
        self.mask_run_select.set_visibility(has_runs)
        self.update_mask_run_interactivity()

    def reset(self, keep_display=False):
        """Reset status of UI-elements"""
        if not keep_display:
            self.hdf_key_display.set_text("")
            self.file_path_display.set_text("")
            self.hdf_value_display.set_text("")
        self.cmap_list.value = re.CMAP_LIST[0]
        self.cmap_list.disable()
        self.channel_select.set_visibility(False)
        self.mask_toggle.set_visibility(False)
        self.mask_run_select.set_visibility(False)
        self.disable_sliders()
        self.image, self.image_norm, self.mask = None, None, None
        self.main_plot.set_visibility(True)
        self.save_image_button.disable()
        self.histogram_plot.set_visibility(False)
        self.image_info_table.set_visibility(False)
        self.panel_tabs.set_value(self.tab_one)
        self.selected_tab = 1

    def reset_min_max(self):
        """Reset minimum and maximum values of the contrast sliders"""
        self.min_slider.set_value(0)
        self.max_slider.set_value(255)

    def display_image(self, data_obj):
        """Display an event as an image.

        Handles a 2D grayscale dataset ``(H, W)`` and, for the cytometry
        events, a 3D channel stack ``(C, H, W)`` -- a single channel layer is
        shown, selected by the "Channel" field (1-based). Only the chosen
        layer is read from disk, so this stays cheap for large stacks.
        """
        self.enable_ui_image()
        shape = data_obj.shape
        if len(shape) == 3:
            n_layers = shape[0]
            self.channel_select.set_visibility(True)
            try:
                layer = int(self.channel_select.value)
            except (TypeError, ValueError):
                layer = 0
            layer = int(np.clip(layer, 0, n_layers - 1))
            self.image = data_obj[layer]
        else:
            self.channel_select.set_visibility(False)
            self.image = data_obj[:]
        min_val = int(self.min_slider.value)
        max_val = int(self.max_slider.value)
        if min_val > 0 or max_val < 255:
            if min_val >= max_val:
                min_val = np.clip(max_val - 1, 0, 254)
                self.min_slider.set_value(min_val)
            nmin, nmax = np.min(self.image), np.max(self.image)
            if nmax != nmin:
                self.image_norm = np.uint8(
                    255.0 * (self.image - nmin) / (nmax - nmin))
                self.image_norm = np.clip(self.image_norm, min_val, max_val)
            else:
                self.image_norm = np.zeros(self.image.shape, dtype=np.uint8)
        else:
            self.image_norm = np.copy(self.image)

        # Read the matching instance mask (same id, selected run) when the
        # overlay is on. Navigating by the open dataset object keeps this a
        # single O(1) lookup and works for both file layouts:
        #   data_obj            -> the event dataset (".../events/<id>")
        #   .parent             -> the events group
        #   .parent.parent      -> the experiment group (file root for a
        #                          single-experiment file)
        self.mask = None
        if self.mask_toggle.value and self.mask_run_select.value:
            try:
                experiment_group = data_obj.parent.parent
                masks_group = experiment_group.get("masks")
                run = self.mask_run_select.value
                name = data_obj.name.rsplit("/", 1)[-1]
                if masks_group is not None and run in masks_group \
                        and name in masks_group[run]:
                    self.mask = masks_group[run][name][()]
            except Exception:
                self.mask = None

        self.fig = self.main_plot.figure
        self.fig.clf()
        self.fig.set_dpi(self.dpi)
        self.ax = self.fig.gca()
        self.ax.imshow(self.image_norm, cmap=self.cmap_list.value)
        if self.mask is not None:
            self.ax.imshow(util.colorize_mask(self.mask, alpha=re.MASK_ALPHA))
        self.fig.tight_layout()
        self.main_plot.update()

        if self.selected_tab == 2:
            rows, columns = util.format_statistical_info(self.image)
            self.image_info_table.rows[:] = rows
            self.image_info_table.columns[:] = columns
            self.image_info_table.update()
            with self.histogram_plot:
                plt.clf()
                flat_data = self.image.ravel()
                num_bins = min(255, len(flat_data))
                hist, bin_edges = np.histogram(flat_data, bins=num_bins)
                plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist,
                         color='skyblue', edgecolor='black', alpha=0.65,
                         label=f"Num bins: {num_bins}")
                plt.title("Histogram")
                plt.xlabel("Grayscale")
                plt.ylabel("Frequency")
                plt.legend()
                self.histogram_plot.update()

    def __clear_plot(self):
        self.main_plot.figure.clf()
        self.main_plot.update()
        with self.histogram_plot:
            plt.clf()
            self.histogram_plot.update()

    def show_data(self):
        """Display data getting from a hdf file"""
        file_path1 = self.file_path_display.text
        hdf_key1 = self.hdf_key_display.text
        if (file_path1 != "") and (hdf_key1 != "") and (hdf_key1 is not None):
            new_state = (file_path1, hdf_key1, self.hdf_value_display.text,
                         self.cmap_list.value, self.min_slider.value,
                         self.max_slider.value, self.selected_tab,
                         self.channel_select.value,
                         self.mask_toggle.value, self.mask_run_select.value)
            if new_state != self.current_state:
                self.current_state = new_state
                try:
                    (data_type, value) = util.get_hdf_data(file_path1,
                                                           hdf_key1)
                    if data_type in ("string", "number", "boolean"):
                        self.hdf_value_display.set_text(str(value))
                        self.__clear_plot()
                        self.reset(keep_display=True)
                    elif data_type == "array":
                        self.hdf_value_display.set_text("Array shape: "
                                                        "" + str(value))
                        dim = len(value)
                        if dim == 2 or dim == 3:
                            hdf_obj = h5py.File(file_path1, "r")
                            self.display_image(hdf_obj[hdf_key1])
                            hdf_obj.close()
                        else:
                            ui.notify("Can't display {}-d array! Only 2D "
                                      "images or (channel, H, W) stacks are "
                                      "supported.".format(dim))
                            self.__clear_plot()
                            self.reset(keep_display=True)
                    else:
                        self.hdf_value_display.set_text(data_type)
                        self.__clear_plot()
                        self.reset(keep_display=True)
                except Exception as error:
                    self.reset(keep_display=True)
                    ui.notify("Error reading dataset: {}".format(error))
        else:
            self.hdf_value_display.set_text("")
            self.__clear_plot()
            self.reset(keep_display=True)

    def _compose_current_view(self):
        """Render exactly what is on screen to an ``(H, W, 3)`` uint8 RGB array.

        Reproduces the displayed view: the contrast-adjusted base image mapped
        through the active colormap (matplotlib's default min/max scaling, as in
        ``imshow``), with the coloured mask overlay alpha-composited on top when
        it is showing. Used by "Save current view" so the file matches the
        display rather than the raw channel data.
        """
        arr = np.asarray(self.image_norm, dtype=np.float64)
        amin, amax = arr.min(), arr.max()
        if amax > amin:
            norm = (arr - amin) / (amax - amin)
        else:
            norm = np.zeros_like(arr)
        rgb = plt.get_cmap(self.cmap_list.value)(norm)[..., :3]
        if self.mask is not None:
            rgba = util.colorize_mask(self.mask, alpha=re.MASK_ALPHA)
            a = rgba[..., 3:4]
            rgb = rgb * (1.0 - a) + rgba[..., :3] * a
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    async def save_image(self) -> None:
        """Save the current view (as shown) when 'Save current view' is
        clicked."""
        if self.last_folder and os.path.exists(self.last_folder):
            start_dir = self.last_folder
        else:
            start_dir = "~"
        file_path = await FileSaver(start_dir,
                                    title="File name (ext: .tif, .jpg, .png)")
        if not file_path or self.image_norm is None:
            return
        if os.path.splitext(file_path)[-1] not in (".tif", ".jpg", ".png"):
            ui.notify("Please use .tif, .jpg, or .png as file extension!")
            return
        overwriting = os.path.isfile(file_path)
        error = util.save_image(file_path, self._compose_current_view())
        if error is not None:
            ui.notify(error)
        elif overwriting:
            ui.notify("File {} is overwritten".format(file_path))
        else:
            ui.notify("File is saved at: {}".format(file_path))

    def shutdown(self):
        """Routine to close the app"""
        self.timer.cancel()
        ui.notify("The server has been stopped. You can close this tab!")
