# ifcview

Minimal browser-based HDF5 viewer for **imaging flow cytometry (IFC)** data,
written in pure Python with [NiceGUI](https://nicegui.io/).

It is built to browse large per-experiment IFC datasets that have been
consolidated into a single HDF5 "master" file, without ever enumerating the
(potentially millions of) events in a group.

## Data model

An **experiment** is any group that contains an `events` subgroup. Two file
layouts are supported and detected automatically:

- a **single-experiment file**, where the file root itself holds `events` (and,
  optionally, `masks`); the dropdown shows one entry, labelled with the file
  stem;
- a **master file**, where each top-level group (e.g. `/C54`, `/C55`) is an
  experiment holding its own `events`/`masks` subgroups.

Within an experiment:

- `events/<id>` is one event, named by its integer id (`"0"`, `"1"`, ...), stored
  as a `(channels, height, width)` `uint16` stack — the channel is a layer along
  axis 0, selected at display time;
- `masks/<run>/<id>` (optional) is the matching instance mask for that event: a
  `(height, width)` integer label image where `0` is background and `1..N` are
  distinct instances, with one mask per event (1:1). Multiple mask **runs**
  (e.g. `cyto3_best`) can coexist as sibling subgroups;
- experiment metadata (`channel_names`, `physical_channels`, `kept_channels`,
  ...) lives in the experiment group's HDF5 attributes — the **file root** attrs
  for a single-experiment file — and drives the channel labels and per-channel
  colormaps.

## Features

- Open a master `.h5` and pick an experiment from a dropdown.
- On selecting an experiment, a bounded scan finds the first few pages of real
  cells (skipping empty/noise events) for a quick **browse** preview.
- Look up any event directly by id, or click a browsed cell to display it.
- Per-channel display: pick a channel, auto colormap, adjustable contrast, plus
  an image-information tab (histogram + statistics).
- **Mask overlay**: toggle a coloured instance-mask overlay (distinct colour per
  instance, translucent) on top of the displayed image, with a mask-run dropdown
  when several runs are available. It is a direct per-event lookup, so it does
  not affect browsing performance.
- Save the current view (image + colormap + contrast + any mask overlay, exactly
  as shown) to `.tif`, `.jpg`, or `.png`.
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

or simply run with your .venv properly activated:

```bash
python -m ifcview.main 
```

Then open the printed URL in a browser. Exit with `Ctrl + C`.


## Credit & license

`ifcview` is a fork of [broh5](https://github.com/algotom/broh5) by
Nghia T. Vo, repurposed from a general/tomography HDF viewer into a minimal IFC
viewer. Licensed under Apache 2.0 (see `LICENSE`).
