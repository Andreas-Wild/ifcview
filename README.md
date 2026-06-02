# ifcview

Minimal browser-based HDF5 viewer for **imaging flow cytometry (IFC)** data,
written in pure Python with [NiceGUI](https://nicegui.io/).

It is built to browse large per-experiment IFC datasets that have been
consolidated into a single HDF5 "master" file, without ever enumerating the
(potentially millions of) events in a group.

## Data model

The viewer expects a master `.h5` file where:

- each **top-level group** is an experiment / sample (e.g. `/C54`, `/C55`);
- each **dataset** is one event, named by its integer id (`"0"`, `"1"`, ...);
- each event is a `(channels, height, width)` `uint16` stack — the channel is a
  layer along axis 0, selected at display time;
- per-experiment metadata (`channel_names`, `physical_channels`,
  `kept_channels`, ...) lives in the group's HDF5 attributes and drives the
  channel labels and per-channel colormaps.

## Features

- Open a master `.h5` and pick an experiment from a dropdown.
- On selecting an experiment, a bounded scan finds the first few pages of real
  cells (skipping empty/noise events) for a quick **browse** preview.
- Look up any event directly by id, or click a browsed cell to display it.
- Per-channel display: pick a channel, auto colormap, adjustable contrast, plus
  an image-information tab (histogram + statistics).
- Save the displayed image to `.tif`, `.jpg`, or `.png`.
- Reads compressed datasets via [hdf5plugin](https://pypi.org/project/hdf5plugin/).

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). The Python version
is pinned in `.python-version` and dependencies are locked in `uv.lock`, so a
single command creates the environment and installs everything:

```bash
uv sync
```

## Run

```bash
uv run ifcview --port 8180
```

Then open the printed URL in a browser. Exit with `Ctrl + C`.

## Test data

`make_test_data.py` merges per-sample source files (`C54.h5`, `C55.h5`, ...)
into a single master `test.h5` for development.

## Credit & license

`ifcview` is a fork of [broh5](https://github.com/algotom/broh5) by
Nghia T. Vo, repurposed from a general/tomography HDF viewer into a minimal IFC
viewer. Licensed under Apache 2.0 (see `LICENSE`).
