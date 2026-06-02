"""
Utility helpers for the viewer:

    -   Screen size, used to size the figures.
    -   List experiments (top-level groups) and scan an experiment for cells.
    -   Resolve a dataset key from an experiment and event id.
    -   Derive per-channel labels and colormaps from experiment attributes.
    -   Read a dataset's type/value and basic image statistics.
    -   Save the displayed image to a file.
    -   Save/load the path of the last-opened folder.
"""

import os
import json
import platform
import tkinter as tk
import h5py
import hdf5plugin  # noqa: F401  (registers HDF5 compression plugins on import)
import numpy as np
from PIL import Image


def get_height_width_screen():
    """
    Get the height and width of the current screen.

    Returns
    -------
    tuple
       A tuple of (screen height, screen width).
    """
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    dpi = root.winfo_fpixels('1i')
    root.destroy()
    return screen_height, screen_width, dpi


def list_top_level_groups(hdf_file):
    """
    List the top-level groups (experiments/samples) of an HDF5 file.

    Only the first level is read, so this stays cheap even for very large
    master files. The full file is never walked.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.

    Returns
    -------
    list or str
        A list of top-level group names, or a string describing an error.
    """
    try:
        with h5py.File(hdf_file, 'r') as f:
            return [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    except Exception as error:
        return str(error)


def event_is_cell(layer, threshold=0.01):
    """
    Cheaply decide whether one channel image looks like a real captured cell.

    Empty/noise events have a wide, sparse intensity histogram with no
    concentrated structure, so their relative contrast (std / mean) is low.
    A real cell shows a bright/dark blob against the background, giving a much
    higher std / mean. This is bit-depth independent and needs only the
    statistics of a single channel layer (no histogram).

    Parameters
    ----------
    layer : ndarray
        A single 2D channel image (e.g. the brightfield layer of an event).
    threshold : float
        Minimum coefficient of variation (std / mean) to count as a cell.

    Returns
    -------
    bool
        True if the layer looks like a real cell.
    """
    layer = np.asarray(layer, dtype=np.float64)
    mean = layer.mean()
    if mean <= 0:
        return False
    return (layer.std() / mean) >= threshold


def _layer_is_cell(dset, channel, threshold, max_samples):
    """Sub-sample one channel layer of an event and run the cell test.

    Reads at most ~``max_samples`` x ``max_samples`` strided pixels instead of
    the full image, so judging an event costs roughly constant work regardless
    of frame size. This keeps the browse scan cheap even over millions of
    events (and avoids starving the asyncio loop while it runs in a worker).
    """
    try:
        if dset.ndim == 3:
            ch = min(channel, dset.shape[0] - 1)
            h, w = dset.shape[1], dset.shape[2]
            sh, sw = max(1, h // max_samples), max(1, w // max_samples)
            layer = dset[ch, ::sh, ::sw]
        else:
            h, w = dset.shape[0], dset.shape[1]
            sh, sw = max(1, h // max_samples), max(1, w // max_samples)
            layer = dset[::sh, ::sw]
    except Exception:
        return False
    return event_is_cell(layer, threshold)


def list_cell_events_page(hdf_file, group_path, start_id=0, count=10,
                          channel=0, threshold=0.01, scan_limit=20000,
                          max_samples=32):
    """
    List a page of cell-looking events, resuming from a raw event-id cursor.

    Probes integer event ids in order (0-based) starting at ``start_id`` with
    O(1) link-existence checks, and for each existing event reads only a
    sub-sampled brightfield layer (see :func:`_layer_is_cell`), keeping it if
    :func:`event_is_cell` passes. Crucially it resumes from ``start_id`` and
    returns a cursor for the next page, so paging never rescans from the
    beginning -- the cost is O(events on this page), not O(events so far).
    Scanning is bounded by ``scan_limit`` so a rare-cell sample can't stall.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    group_path : str
        Path to the group (an experiment name like ``"C54"``).
    start_id : int
        Raw event id to start scanning from (the cursor of this page).
    count : int
        Number of cell events to return.
    channel : int
        Layer index used to judge "is a cell" (0 = brightfield).
    threshold : float
        Minimum std / mean of that layer to count as a cell.
    scan_limit : int
        Maximum number of ids probed in this call.
    max_samples : int
        Target sub-sample resolution per axis when reading a layer.

    Returns
    -------
    tuple
        ``(names, has_more, next_id)`` -- the page of passing event names,
        whether another passing event was found within the scan, and the raw
        event id to start the next page from (only meaningful if ``has_more``).
    """
    try:
        with h5py.File(hdf_file, 'r') as f:
            if group_path not in f:
                return [], False, start_id
            group = f[group_path]
            kept = []
            probes = 0
            event_id = max(0, int(start_id))
            while probes < scan_limit:
                name = str(event_id)
                if name in group and _layer_is_cell(group[name], channel,
                                                    threshold, max_samples):
                    if len(kept) >= count:
                        # One extra cell confirms a next page and fixes its
                        # resume cursor; stop before reading any further.
                        return kept, True, event_id
                    kept.append(name)
                event_id += 1
                probes += 1
            return kept, False, event_id
    except Exception:
        return [], False, start_id


def __event_id_candidates(event_id):
    """Yield plausible stored forms of an event id (exact, then zero-padded)."""
    event_id = str(event_id).strip()
    seen = {event_id}
    yield event_id
    if event_id.isdigit():
        value = str(int(event_id))
        for width in range(len(value), 9):
            padded = value.zfill(width)
            if padded not in seen:
                seen.add(padded)
                yield padded


def resolve_dataset_key(hdf_file, experiment, event_id):
    """
    Resolve an event dataset key from an experiment and event id.

    Each event is a single ``(channel, H, W)`` stack named by its event id;
    the channel is selected as a layer at display time, not part of the name.
    Because event ids may or may not be zero-padded, a few candidate forms are
    tried using O(1) link-existence checks (``name in group``), which never
    enumerate the (potentially millions of) sibling datasets.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    experiment : str
        Top-level group name (e.g. ``"C54"``).
    event_id : str or int
        Event identifier as entered by the user.

    Returns
    -------
    str or None
        The resolved key (e.g. ``"C54/42"``) if a matching dataset exists,
        otherwise None.
    """
    try:
        with h5py.File(hdf_file, 'r') as f:
            if experiment not in f:
                return None
            group = f[experiment]
            for candidate in __event_id_candidates(event_id):
                if candidate in group:
                    return f"{experiment}/{candidate}"
    except Exception:
        return None
    return None


def _parse_json_attr(value):
    """Parse an HDF5 attribute that stores a JSON list (e.g. channel_names).

    Attributes may come back as a JSON string, bytes, or an array; return a
    Python list, or None if it can't be parsed.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    try:
        return list(value)
    except TypeError:
        return None


def _probe_n_layers(group, scan_limit=10000):
    """Determine the channel-layer count from the FIRST existing event.

    Probes integer event ids 0, 1, 2, ... with O(1) link-existence checks and
    reads only the *shape* (header, not data) of the first event found. This
    deliberately avoids ``next(iter(group))``: iterating an HDF5 group forces
    h5py to read and alphabetically sort every link name, which on a master
    holding hundreds of thousands of events takes tens of seconds. Returns the
    layer count (1 for a 2D event), or None if no integer-named event is found
    within ``scan_limit`` probes (e.g. a non-integer-named group).
    """
    event_id = 0
    while event_id < scan_limit:
        name = str(event_id)
        if name in group:
            dset = group[name]
            return int(dset.shape[0]) if dset.ndim == 3 else 1
        event_id += 1
    return None


def get_channel_labels(hdf_file, experiment):
    """
    Return per-layer channel labels for an experiment from its attributes.

    Each event stack has one layer per *kept* channel. ``channel_names`` is
    indexed by ``physical_channels``, so ``kept_channels`` is mapped through
    ``physical_channels`` to the names (e.g. kept channel 6 -> "Ch06 SSC").
    Falls back to ``"Channel 1".."Channel N"`` (sized to the actual stack) when
    the attributes are missing or inconsistent.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    experiment : str
        Top-level group name (e.g. ``"C54"``).

    Returns
    -------
    list of str
        One label per channel layer (empty list if it can't be determined).
    """
    try:
        with h5py.File(hdf_file, "r") as f:
            if experiment not in f:
                return []
            grp = f[experiment]
            n_layers = _probe_n_layers(grp)
            names = _parse_json_attr(grp.attrs.get("channel_names"))
            physical = _parse_json_attr(grp.attrs.get("physical_channels"))
            kept = _parse_json_attr(grp.attrs.get("kept_channels"))
            if names and physical and kept and len(names) == len(physical):
                phys_to_name = {p: str(n) for p, n in zip(physical, names)}
                labels = [phys_to_name.get(c, "Ch{}".format(c)) for c in kept]
                if n_layers is None or len(labels) == n_layers:
                    return labels
            if n_layers:
                return ["Channel {}".format(i + 1) for i in range(n_layers)]
            return []
    except Exception:
        return []


# Auto colormap per channel: brightfield (channel 1) is always grey; the other
# channels get a perceptually-uniform (viridis-family) map matched (loosely) to
# the signal. Matching is by recognizable tokens in the channel label, with an
# index-based fallback so unknown channels still get a distinct map. All values
# here must stay within rendering.CMAP_LIST.
CHANNEL_CMAP_RULES = [
    ("bf", "gray"),        # brightfield
    ("bright", "gray"),
    ("fitc", "viridis"),   # FITC
    ("gfp", "viridis"),
    ("ssc", "cividis"),    # side scatter
    ("apc", "plasma"),
    ("cd235", "plasma"),
    ("-pe", "inferno"),    # phycoerythrin
    (" pe", "inferno"),
    ("pe-", "inferno"),
]
CHANNEL_CMAP_FALLBACK = ["gray", "viridis", "plasma", "inferno", "magma",
                         "cividis"]


def channel_colormap(label, index=0):
    """
    Pick an automatic colormap for a channel from its label.

    Channel 1 (``index == 0``, brightfield) is always grey. Other channels are
    matched by tokens in the label (e.g. "FITC" -> Greens), falling back to a
    distinct colour by index if the label isn't recognized.
    """
    if index == 0:
        return "gray"
    text = str(label).lower()
    for token, cmap in CHANNEL_CMAP_RULES:
        if token in text:
            return cmap
    return CHANNEL_CMAP_FALLBACK[index % len(CHANNEL_CMAP_FALLBACK)]


def get_hdf_data(file_path, dataset_path):
    """
    Get data type and value from a specified dataset in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    dataset_path : str
        Path to the dataset within the HDF5 file.

    Returns
    -------
    tuple
        A tuple containing the data type and the value of the dataset.
    """
    with h5py.File(file_path, 'r') as file:
        if dataset_path not in file:
            return "not path", None
        try:
            item = file[dataset_path]
            if isinstance(item, h5py.Group):
                return "group", None
            data_type, value = "unknown", None
            # Check the type and shape of a dataset
            if item.dtype.kind == 'S':  # Fixed-length bytes
                data = item[()]
                if item.size == 1:  # Single string or byte
                    if isinstance(data, bytes):
                        data_type, value = "string", data.decode('utf-8')
                    elif isinstance(data.flat[0], bytes):
                        data_type, value = "string", data.flat[0].decode(
                            'utf-8')
                else:
                    data_type, value = "array", [d.decode('utf-8') for d in
                                                 data]
            elif item.dtype.kind == 'U':  # Fixed-length Unicode
                data = item[()]
                if item.size == 1:  # Single string
                    data_type, value = "string", data
                else:
                    data_type, value = "array", list(data)
            elif h5py.check_dtype(vlen=item.dtype) in [str, bytes]:
                data = item[()]
                if isinstance(data, (str, bytes)):
                    data_type, value = "string", data if isinstance(data, str)\
                        else data.decode('utf-8')
                else:
                    joined_data = ''.join(
                        [d if isinstance(d, str) else d.decode('utf-8')
                         for d in data])
                    data_type, value = "string", joined_data
            elif item.dtype.kind in ['i', 'f', 'u']:
                if item.shape == () or item.size == 1:
                    data_type, value = "number", item[()]
                else:
                    data_type, value = "array", item.shape
            elif item.dtype.kind == 'b':  # Boolean type
                data_type, value = "boolean", int(item[()])
            return data_type, value
        except Exception as error:
            return str(error), None


def format_statistical_info(image):
    """
    Get statistical information of a 2d array and format the output as a
    Nicegui table object.

    Parameters
    ----------
    image : ndarray
        NumPy array to format.

    Returns
    -------
    tuple
        A tuple containing the rows and columns formatted for the table.
    """
    data_type = image.dtype.name
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))
    columns = [{"name": "information", "label": "Information",
                "field": "information"},
               {"name": "value", "label": "Value", "field": "value"}]
    # The "field" of each column must match a key in the rows, otherwise the
    # cell renders blank (this was why the Information column showed nothing).
    rows = [{"information": "Minimum", "value": round(min_val, 4)},
            {"information": "Maximum", "value": round(max_val, 4)},
            {"information": "Mean", "value": round(mean_val, 4)},
            {"information": "Standard deviation", "value": round(std_val, 4)},
            {"information": "Data type", "value": data_type}]
    return rows, columns


def save_image(file_path, mat):
    """
    Save a 2D array as an image file.

    Parameters
    ----------
    file_path : str
        Path where the image will be saved.
    mat : ndarray
        2D array to be saved as an image.

    Returns
    -------
    None or str
        Returns None if successful, or a string message if an error occurs.
    """
    file_ext = os.path.splitext(file_path)[-1]
    if not ((file_ext == ".tif") or (file_ext == ".tiff")):
        mat = np.uint8(
            255.0 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat)))
    else:
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32)
    image = Image.fromarray(mat)
    try:
        image.save(file_path)
    except Exception as error:
        return str(error)


def get_config_path():
    """
    Get path to save a config file depending on the OS system.
    """
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        return os.path.join(home, 'AppData', 'Roaming', 'ifcview',
                            'ifcview_config.json')
    elif platform.system() == "Darwin":
        return os.path.join(home, 'Library', 'Application Support', 'ifcview',
                            'ifcview_config.json')
    else:
        return os.path.join(home, '.ifcview', 'ifcview_config.json')


def save_config(data):
    """
    Save data (dictionary) to the config file (json format).
    """
    config_path = get_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(data, f)


def load_config():
    """
    Load the config file.
    """
    config_path = get_config_path()
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
