"""
Module for constructing GUI components:
    -   Main window.
    -   Dialog for selecting a file.
    -   Dialog for saving a file.
"""

import os
import platform
import string
from pathlib import Path
from typing import List, Optional

from nicegui import ui
from nicegui.events import GenericEventArguments

import ifcview.lib.utilities as util

# ============================================================================
#                   Global parameters for the GUI
# ============================================================================

# First entry is the default (channel 1 / brightfield). Perceptually-uniform
# sequential colormaps (viridis family) are used both as the dropdown choices
# and for the per-channel auto-colormap (see util.channel_colormap).
CMAP_LIST = ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]
# Dropdown options: matplotlib name -> capitalised label shown to the user.
CMAP_OPTIONS = {name: name.capitalize() for name in CMAP_LIST}
FONT_STYLE = "font-size: 105%; font-weight: bold"
UPDATE_RATE = 0.2  # second
BROWSE_PAGE_SIZE = 10  # number of datasets listed per browse page
MAX_BROWSE_PAGES = 5  # pages of cells scanned/cached per experiment (cap)
# "Cells only" browse filter: keep events whose brightfield layer has enough
# contrast (std/mean). Empty/noise events have a wide, sparse histogram and
# thus low contrast; real cells show a concentrated bright/dark structure.
CELL_FILTER_CHANNEL = 0  # layer used to judge "is a cell" (0 = brightfield)
CELL_CONTRAST_THRESHOLD = 0.01  # min std/mean of that layer to count as a cell
CELL_FILTER_SCAN_LIMIT = 20000  # max events probed per page when filtering
MASK_ALPHA = 0.45  # opacity of the coloured instance-mask overlay
RATIO = 0.65  # Ratio for adjusting size between image/plot and screen
MAX_FIG_SIZE = [12.0, 9.0]
MAX_PLOT_SIZE = [9.0, 7.0]
INPUT_EXT = ["hdf", "nxs", "h5", "hdf5"]
HEADER_COLOR = "#3874c8"
HEADER_TITLE = "IMAGING FLOW CYTOMETRY HDF5 VIEWER"
LEFT_DRAWER_COLOR = "#d7e3f4"
TREE_BGR_COLOR = "#f8f8ff"


class GuiRendering:
    """
    A class to build the graphical user interface for an HDF image viewer.

    This class creates the UI elements (header, file picker, image plot,
    contrast sliders, colormap selector, and an image-information tab) used
    to view 2D image datasets stored in an HDF file.

    Attributes
    ----------
    fig_size : tuple
        Dimensions for the image figure in the UI.
    plot_size : tuple
        Dimensions for the histogram plot in the UI.
    select_file_button : UI button
        Button to trigger file selection.
    experiment_select : UI select
        Dropdown listing the top-level experiments of the opened file.
    event_input : UI input
        Field for entering the event id to look up.
    channel_select : UI select
        Dropdown of channel names (from the experiment's attributes) selecting
        which layer of the (C, H, W) stack to display.
    browse_container : UI list
        Paginated list of dataset names for the selected experiment.
    file_path_display : UI label
        Label to display selected file path.
    hdf_key_display : UI label
        Label to display HDF key.
    hdf_value_display : UI label
        Label to display HDF value.
    cmap_list : UI select
        Dropdown to select color map for the image.
    save_image_button : UI button
        Button to save current image.
    main_plot : UI matplotlib
        Matplotlib element to display the image.
    min_slider : UI slider
        Slider to adjust the minimum value for image contrast.
    max_slider : UI slider
        Slider to adjust the maximum value for image contrast.
    reset_button : UI button
        Button to reset contrast adjustments.
    histogram_plot : UI pyplot
        Pyplot element to display histogram of an image.
    image_info_table : UI table
        Table to display statistical information of an image.

    Methods
    -------
    init_gui()
        Initializes and constructs the GUI elements.
    """

    def __init__(self):
        super().__init__()
        # Initial parameters
        (sc_height, sc_width, dpi) = util.get_height_width_screen()
        hei_size = RATIO * sc_width / dpi
        wid_size = RATIO * sc_height / dpi
        self.dpi = dpi
        self.fig_size = (min(hei_size, MAX_FIG_SIZE[0]), min(wid_size, MAX_FIG_SIZE[1]))
        self.plot_size = (
            min(hei_size, MAX_PLOT_SIZE[0]),
            min(wid_size, MAX_PLOT_SIZE[1]),
        )
        self.select_file_button = None
        self.lookup_panel = None
        self.experiment_select = None
        self.event_input = None
        self.open_button = None
        self.close_file_button = None
        self.browse_panel = None
        self.browse_info = None
        self.browse_container = None
        self.prev_button = None
        self.next_button = None
        self.file_path_display = None
        self.hdf_key_display = None
        self.hdf_value_display = None
        self.channel_select = None
        self.cmap_list = None
        self.mask_toggle = None
        self.mask_run_select = None
        self.save_image_button = None
        self.main_plot = None
        self.min_slider = None
        self.max_slider = None
        self.reset_button = None
        self.histogram_plot = None
        self.image_info_table = None
        self.tab_one = None
        self.tab_two = None
        self.panel_tabs = None
        self.loading_dialog = None
        self.loading_label = None
        self.init_gui()

    def init_gui(self):
        """
        Initializes and constructs the various elements of the GUI.

        This method sets up the header, drawer (HDF tree), the image plot,
        contrast sliders, colormap selector, and the image-information tab.
        """
        # For the header
        with (
            ui.header()
            .style("background-color: " + HEADER_COLOR)
            .classes("items-center justify-between")
        ):
            ui.label(HEADER_TITLE).style(FONT_STYLE)

        # For the left drawer: file selection, image lookup, and browsing.
        with ui.left_drawer(fixed=True, bottom_corner=True).style(
            "background-color: " + LEFT_DRAWER_COLOR
        ):
            with ui.column().classes("w-full"):
                self.select_file_button = ui.button("Select file").props("icon=folder")

                # Lookup panel: jump straight to an image by ids.
                self.lookup_panel = ui.column().classes("w-full")
                with self.lookup_panel:
                    ui.label("Lookup image").style(FONT_STYLE)
                    self.experiment_select = ui.select([], label="Experiment").classes(
                        "w-full"
                    )
                    self.event_input = ui.input("Event ID").classes("w-full")
                    with ui.row().classes("w-full"):
                        self.open_button = ui.button("Open").props("icon=image")
                        self.close_file_button = ui.button("Close file").props(
                            "outline"
                        )
                self.lookup_panel.set_visibility(False)

                # Browse panel: page through the first datasets of a sample.
                self.browse_panel = ui.column().classes("w-full")
                with self.browse_panel:
                    ui.label("Quick Cell View").style(FONT_STYLE)
                    self.browse_info = ui.label("")
                    self.browse_container = (
                        ui.list().props("dense bordered").classes("w-full")
                    )
                    with ui.row().classes("items-center"):
                        self.prev_button = ui.button("Prev").props("outline")
                        self.next_button = ui.button("Next").props("outline")
                self.browse_panel.set_visibility(False)

        # Layout for the main page.
        with ui.column().classes("w-full no-wrap gap-1"):
            # For displaying file-path, key, and value of a hdf/nxs/h5 file
            with ui.row().classes("w-full no-wrap"):
                with ui.row().classes("w-1/3 items-center"):
                    ui.label("File path: ").style(FONT_STYLE)
                    self.file_path_display = ui.label("")
                with ui.row().classes("w-1/3 items-center"):
                    ui.label("Key: ").style(FONT_STYLE)
                    self.hdf_key_display = ui.label("")
                with ui.row().classes("w-1/3 items-center"):
                    ui.label("Value: ").style(FONT_STYLE)
                    self.hdf_value_display = ui.label("")
            ui.separator()

            # For ui-components used to interact with the image.
            with ui.row().classes("w-full justify-between items-center"):
                with ui.row().classes("items-center"):
                    ui.label("Channel: ").style(FONT_STYLE)
                    self.channel_select = ui.select(options={}, label=None).classes(
                        "w-56"
                    )
                with ui.row().classes("items-center"):
                    ui.label("Color map: ").style(FONT_STYLE)
                    self.cmap_list = ui.select(CMAP_OPTIONS, value=CMAP_LIST[0])
                # Mask overlay: a toggle plus a (normally hidden) run dropdown
                # that only appears once masks exist for the experiment.
                with ui.row().classes("items-center"):
                    self.mask_toggle = ui.switch("Mask overlay")
                    self.mask_run_select = ui.select(
                        options={}, label="Mask run"
                    ).classes("w-40")
                self.mask_toggle.set_visibility(False)
                self.mask_run_select.set_visibility(False)
                self.save_image_button = ui.button("Save current view")

            # Tabs for image visualization and image information
            tabs = ui.tabs().classes("w-full")
            with tabs:
                self.tab_one = ui.tab("Image").style(
                    "background-color: " + TREE_BGR_COLOR
                )
                self.tab_two = ui.tab("Image information").style(
                    "background-color: " + TREE_BGR_COLOR
                )
            self.panel_tabs = ui.tab_panels(tabs, value=self.tab_one).classes("w-full")
            with self.panel_tabs:
                # Tab 1 for displaying the image
                with ui.tab_panel(self.tab_one):
                    with ui.row().classes("w-full justify-center items-center"):
                        self.main_plot = ui.matplotlib(
                            figsize=self.fig_size, dpi=self.dpi
                        ).classes("mx-auto")

                    # Sliders for adjusting the contrast of an image.
                    with ui.row().classes(
                        "w-full justify-between no-wrap items-center"
                    ):
                        ui.label("Min: ").style(FONT_STYLE)
                        self.min_slider = (
                            ui.slider(min=0, max=254, value=0)
                            .props("label-always")
                            .on(
                                "update:model-value",
                                throttle=UPDATE_RATE,
                                leading_events=False,
                            )
                        )

                        ui.label("Max: ").style(FONT_STYLE)
                        self.max_slider = (
                            ui.slider(min=1, max=255, value=255)
                            .props("label-always")
                            .on(
                                "update:model-value",
                                throttle=UPDATE_RATE,
                                leading_events=False,
                            )
                        )
                        self.reset_button = ui.button("Reset")
                # Tab 2 for showing image information
                with ui.tab_panel(self.tab_two):
                    with ui.row().classes(
                        "w-full justify-between no-wrap items-center"
                    ):
                        self.histogram_plot = ui.pyplot(
                            figsize=self.plot_size,
                            close=False,
                        ).classes("w-full")
                        self.image_info_table = ui.table(
                            columns=[], rows=[], row_key="information"
                        )

        # Reusable modal "busy" overlay, built once here (at page-construction
        # time) so it always belongs to this page's client. It can then be
        # toggled reliably even from a deferred value-change handler -- creating
        # a fresh ui.dialog() inside such a handler attaches it to whatever slot
        # context happens to be current, which is why the scan previously ran
        # with no visible buffer. It is shown via GuiInteraction.loading_overlay
        # and never re-created; the label text is updated per use.
        with (
            ui.dialog().props("persistent") as self.loading_dialog,
            ui.card()
            .classes("items-center gap-4 w-72")
            .style("background-color: " + LEFT_DRAWER_COLOR),
        ):
            ui.spinner(size="lg")
            self.loading_label = ui.label("").classes("text-center w-full")


class FilePicker(ui.dialog):
    """
    A dialog class for file picking in the GUI. Codes are adapted from an
    example of NiceGUI:
    https://github.com/zauberzeug/nicegui/tree/main/examples/local_file_picker

    Allows users to browse and select files from the local filesystem where
    the application is running.

    Parameters
    ----------
    directory : str
        The starting directory for the file picker.
    upper_limit : str, optional
        The upper directory limit for browsing. None by default.
    show_hidden_files : bool, optional
        Flag to show hidden files. False by default.
    allowed_extensions : List of str, optional
        List of allowed file extensions for filtering. None by default.

    Methods
    -------
    check_extension(filename: str)
        Check if the given filename has an allowed extension.
    add_drives_toggle()
        Add a toggle for drive selection on Windows systems.
    update_grid()
        Update the file grid based on the current directory and filters.
    handle_double_click(GenericEventArguments)
        Handle double click events on the file grid.
    handle_ok()
        Handle the OK button, click to submit the selected file path.
    """

    def __init__(
        self,
        directory: str,
        *,
        upper_limit: Optional[str] = None,
        show_hidden_files: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.show_hidden_files = show_hidden_files
        self.allowed_extensions = allowed_extensions
        self.drives_toggle = None
        self.path = Path(directory).expanduser()
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(
                directory if upper_limit is ... else upper_limit
            ).expanduser()
        with self, ui.card():
            self.add_drives_toggle()
            self.grid = (
                ui.aggrid(
                    {
                        "columnDefs": [{"field": "name", "headerName": "File"}],
                        "rowSelection": "single",
                    },
                    html_columns=[0],
                )
                .classes("w-96")
                .on("cellClicked", self.handle_single_click)
                .on("cellDoubleClicked", self.handle_double_click)
            )
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=self.close).props("outline")
                ui.button("Ok", on_click=self.handle_ok)
        # Track the row picked by a single click so 'Ok' works without needing
        # the aggrid selection round-trip (which a fresh update_grid would wipe).
        self.selected_path = None
        self.update_grid()

    def check_extension(self, filename: str) -> bool:
        """Check if the filename has an allowed extension."""
        if self.allowed_extensions is None:
            return True
        else:
            return filename.split(".")[-1].lower() in self.allowed_extensions

    def add_drives_toggle(self):
        """Give a list of available drivers in a WinOS computer"""
        if platform.system() == "Windows":
            drives = [
                "%s:\\" % d for d in string.ascii_uppercase if os.path.exists("%s:" % d)
            ]
            if self.path != "" or self.path != ".":
                select_drive = os.path.splitdrive(self.path)[0] + "\\"
            else:
                self.path = Path(drives[0]).expanduser()
                select_drive = drives[0]
            self.drives_toggle = ui.toggle(
                drives, value=select_drive, on_change=self.__update_drive
            )

    def __update_drive(self):
        if self.drives_toggle:
            self.path = Path(self.drives_toggle.value).expanduser()
            self.update_grid()

    def update_grid(self) -> None:
        paths = list(self.path.glob("*"))
        if not self.show_hidden_files:
            paths = [p for p in paths if not p.name.startswith(".")]
        if self.allowed_extensions:
            paths = [p for p in paths if p.is_dir() or self.check_extension(p.name)]
        paths.sort(key=lambda p: p.name.lower())
        paths.sort(key=lambda p: not p.is_dir())

        self.grid.options["rowData"] = [
            {
                "name": f"📁 <strong>{p.name}</strong>" if p.is_dir() else p.name,
                "path": str(p),
            }
            for p in paths
        ]
        if (
            self.upper_limit is None
            and self.path != self.path.parent
            or self.upper_limit is not None
            and self.path != self.upper_limit
        ):
            self.grid.options["rowData"].insert(
                0,
                {
                    "name": "📁 <strong>..</strong>",
                    "path": str(self.path.parent),
                },
            )
        self.grid.update()

    def handle_single_click(self, e: GenericEventArguments) -> None:
        """Remember the row a single click landed on (for the 'Ok' button)."""
        self.selected_path = Path(e.args["data"]["path"])

    def handle_double_click(self, e: GenericEventArguments) -> None:
        """Open a folder, or pick a file, on double click."""
        self.path = Path(e.args["data"]["path"])
        if self.path.is_dir():
            self.selected_path = None
            self.update_grid()
        elif self.path:
            self.submit(str(self.path))

    def handle_ok(self):
        """Confirm the single-click selection: enter folders, submit files."""
        selected = self.selected_path
        if selected is None:
            ui.notify("Select a file or folder first.")
            return
        if selected.is_dir():
            self.path = selected
            self.selected_path = None
            self.update_grid()
            return
        if self.check_extension(selected.name):
            self.submit(str(selected))
        else:
            exts = ", ".join("." + e for e in (self.allowed_extensions or []))
            ui.notify(
                "Please select a file with the extension: " + (exts or "(any)") + "!"
            )


class FileSaver(ui.dialog):
    """
    A dialog class for saving files in the GUI.

    Allows users to specify a file name and directory for saving files.

    Parameters
    ----------
    directory : str
        Starting directory for the file saver.
    upper_limit : str, optional
        Upper directory limit for browsing. None by default.
    show_hidden_files : bool, optional
        Flag to show hidden files. False by default.
    title : str, optional
        Title for the file-name input-field. 'File name' by default.

    Methods
    -------
    add_drives_toggle()
        Add a toggle for drive selection on Windows systems.
    update_grid() -> None
        Update the file grid based on the current directory.
    handle_double_click(e: GenericEventArguments) -> None
        Handle double-click events on the file grid.
    handle_save()
        Handle the Save button, click to submit the specified file path.
    create_folder_dialog()
        Open a dialog for creating a new folder.
    create_folder(folder_name: str, dialog: ui.dialog)
        Create a new folder with the specified name.
    """

    def __init__(
        self,
        directory: str,
        *,
        upper_limit: Optional[str] = None,
        show_hidden_files: bool = False,
        title: Optional[str] = "File name",
    ) -> None:
        super().__init__()
        self.show_hidden_files = show_hidden_files
        self.drives_toggle = None
        self.path = Path(directory).expanduser()
        self.title = title
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(
                directory if upper_limit is ... else upper_limit
            ).expanduser()

        with self, ui.card():
            self.add_drives_toggle()
            self.grid = (
                ui.aggrid(
                    {
                        "columnDefs": [{"field": "name", "headerName": "File"}],
                        "rowSelection": "single",
                    },
                    html_columns=[0],
                )
                .classes("w-96")
                .on("cellDoubleClicked", self.handle_double_click)
            )
            # Input field for filename
            self.filename_input = (
                ui.input(self.title)
                .classes("w-full justify-between")
                .on("keydown.enter", self.handle_save)
            )
            with ui.row().classes("w-full justify-between"):
                ui.button("Create Folder", on_click=self.create_folder_dialog).props(
                    "outline"
                )
                ui.button("Cancel", on_click=self.close).props("outline")
                ui.button("Save", on_click=self.handle_save)
        self.update_grid()

    def add_drives_toggle(self):
        """Give a list of available drivers in a WinOS computer"""
        if platform.system() == "Windows":
            drives = [
                "%s:\\" % d for d in string.ascii_uppercase if os.path.exists("%s:" % d)
            ]
            if self.path != "" or self.path != ".":
                select_drive = os.path.splitdrive(self.path)[0] + "\\"
            else:
                self.path = Path(drives[0]).expanduser()
                select_drive = drives[0]
            self.drives_toggle = ui.toggle(
                drives, value=select_drive, on_change=self.__update_drive
            )

    def __update_drive(self):
        if self.drives_toggle:
            self.path = Path(self.drives_toggle.value).expanduser()
            self.update_grid()

    def update_grid(self) -> None:
        paths = list(self.path.glob("*"))
        if not self.show_hidden_files:
            paths = [p for p in paths if not p.name.startswith(".")]
        paths.sort(key=lambda p: p.name.lower())
        paths.sort(key=lambda p: not p.is_dir())

        self.grid.options["rowData"] = [
            {
                "name": f"📁 <strong>{p.name}</strong>" if p.is_dir() else p.name,
                "path": str(p),
            }
            for p in paths
        ]
        if (
            self.upper_limit is None
            and self.path != self.path.parent
            or self.upper_limit is not None
            and self.path != self.upper_limit
        ):
            self.grid.options["rowData"].insert(
                0, {"name": "📁 <strong>..</strong>", "path": str(self.path.parent)}
            )
        self.grid.update()

    def handle_double_click(self, e: GenericEventArguments) -> None:
        self.path = Path(e.args["data"]["path"])
        if self.path.is_dir():
            self.update_grid()
        else:
            self.filename_input.value = self.path.name
            self.path = self.path.parent

    def handle_save(self):
        filename = self.filename_input.value
        if not filename:
            ui.notify("File name cannot be empty!")
            return
        save_path = self.path / filename
        save_path_str = str(save_path).replace("\\", "/")
        self.submit(save_path_str)

    async def create_folder_dialog(self):
        """Open a dialog to get the name of the new folder and create it."""
        with ui.dialog().classes("w-100 h-100") as dialog, ui.card():
            with ui.column():
                folder_name_input = ui.input("Folder Name").classes(
                    "w-full justify-between"
                )
                with ui.row():
                    ui.button("Cancel", on_click=dialog.close).props("outline")
                    ui.button(
                        "Create",
                        on_click=lambda: self.create_folder(
                            folder_name_input.value, dialog
                        ),
                    )
        await dialog

    def create_folder(self, folder_name: str, dialog: ui.dialog):
        if not folder_name:
            ui.notify("Folder name cannot be empty!")
            return
        new_folder_path = self.path / folder_name
        if new_folder_path.exists():
            ui.notify(f"A folder named '{folder_name}' already exists!")
            return
        new_folder_path.mkdir(parents=True, exist_ok=True)
        self.update_grid()
        dialog.close()
