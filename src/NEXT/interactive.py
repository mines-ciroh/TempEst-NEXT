#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:05:09 2026

@author: Daniel Philippus

This module supports interactive and command-line use.
"""

import urllib.request as request
import os
from NEXT import NEXT
import pickle
from NEXT import data
import pandas as pd
import argparse
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

next_url = "https://github.com/mines-ciroh/TempEst-NEXT/raw/refs/heads/master/coefs.pickle"

def get_model(cache=None, url=next_url, overwrite=False):
    """
    Retrieve and return a pre-trained model.

    Parameters
    ----------
    cache : str, optional
        If provided, retrieve from or write to a local file. The default is None.
    url : str, optional
        URL from which to retrieve model file. The default is next_url.
    overwrite : bool, optional
        Overwrite cached model if it exists. The default is False.

    Returns
    -------
    NEXT.NEXT
        The pre-trained model.
    """
    # Check if a stored model exists
    if not overwrite and cache is not None:
        if os.path.exists(cache):
            return NEXT.from_pickle(cache)
    # No stored model.
    req = request.urlopen(url)
    model = pickle.loads(req.read())
    req.close()
    if cache is not None:
        if overwrite or os.path.exists(cache):
            model.to_pickle(cache)
    return model


def iget(arg, query, options=None, default=None):
    """
    Interactively update parameter if needed.

    Parameters
    ----------
    arg : str
        Pre-specified value. Will only ask if arg is None.
    query : str
        What to ask the user for.
    options : [str], optional
        List of options for categorical questions. The default is None. Will
        be set to all lowercase.
    default : str, optional
        The default option if user enters nothing. The default is None. If
        None, it will just complain and ask again.

    Returns
    -------
    str
        New argument value.
    """
    oq = query
    if arg is not None:
        return arg
    if options is not None:
        options = [x.lower() for x in options]
        query = query + f" ({', '.join(options)})"
    if default is not None:
        query = query + f" [default: {default}]"
    arg = input(query + ': ')
    if not arg.strip():
        if default is not None:
            return default
        print("Error: a value must be provided.")
        return iget(None, oq, options, default)
    if options is not None:
        if arg.lower() in options:
            return arg
        print(f"Error: invalid answer {arg}")
        return iget(None, oq, options, default)
    return arg.lower()


# Query, options, default. We separate these out as a dict so they are usable
# for multiple interaction modes.
descs = {
    "site_type": ("Site type", list(data.geom_fns), None),
    "site": ("Numeric site ID (no prefix; e.g., 10343500)", None, None),
    "lat": ("Pour point latitude (Decimal degrees North, e.g. 40.623)", None, None),
    "lon": ("Pour point longitude (Decimal degrees East, e.g. -120.1)", None, None),
    "start": ("Start date (yyyy-mm-dd)", None, "2001-01-01"),
    "end": ("End date (yyyy-mm-dd)", None, "2020-12-31"),
    "weather": ("Weather data source", list(data.weather_fns), "gridmet"),
    "model": ("Pre-trained model URL (optional)", None,
              next_url),
    "modpath": ("Pre-trained model cache file (optional)", None, "./model.pickle"),
    "datafile": ("Location to cache input data (optional)", None, ""),
    "output": ("Output file location", None, None)
    }


def run(site_type=None, site=None, lat=None, lon=None, start=None, end=None,
        weather=None, model=None, modpath=None, datafile=None, output=None):
    # Validate inputs
    site_type = iget(site_type, *descs['site_type'])
    if site_type == 'coordinates':
        lat = iget(lat, *descs['lat'])
        lon = iget(lon, *descs['lon'])
        site = lon + ":" + lat
    else:
        site = iget(site, *descs['site'])
    start = iget(start, *descs['start'])
    end = iget(end, *descs['end'])
    weather = iget(weather, *descs['weather'])
    model = iget(model, *descs['model'])
    modpath = iget(modpath, *descs['modpath'])
    datafile = iget(datafile, *descs['datafile'])
    output = iget(output, *descs['output'])
    # Actually run the model, retrieve data, etc
    print("Retrieving model")
    nxt = get_model(modpath, model, False)
    print("Downloading prediction data")
    if datafile and os.path.exists(datafile):
        inpdata = pd.read_csv(datafile, parse_dates=['date'])
    else:
        inpdata = data.full_data(site, start, end, site_type=site_type,
                                 weather=weather)
        if datafile:
            inpdata.to_csv(datafile)
    print("Running model")
    predictions = nxt.run(inpdata, reset=True)
    print("Results preview")
    print(predictions)
    predictions.to_csv(output)
    print(f"Done; results are in {output}")
    sys.exit()




class GUI(tk.Frame):
    browse = ['modpath', 'datafile', 'output']
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()
        self.setup()
    def setup(self):
        self.entries = {}
        self.values = {}
        self.createEntries()
    def createEntries(self):
        self.entryFrame = tk.Frame(self)
        tk.Label(self.entryFrame, text="Note: lat/lon are only needed if site_type=coordinates")
        self.gridFrame = tk.Frame(self.entryFrame)
        for (ix, (name, (desc, opts, default))) in enumerate(descs.items()):
            tk.Label(self.gridFrame, text=f"[{name}] " + desc + ":").grid(row=ix)
            if opts is None:
                self.entries[name] = tk.StringVar(self.master)
                entry = tk.Entry(self.gridFrame, width=30, textvariable=self.entries[name])
                entry.grid(row=ix, column=1)
                if name in self.values:
                    # Keep previous value
                    self.entries[name].set(self.values[name])
                if name in self.browse:
                    tk.Button(self.gridFrame, text="Browse",
                              command=lambda: self.entries[name].set(askopenfilename())).grid(row=ix, column=2)
            else:
                self.entries[name] = tk.StringVar(self.master)
                if name in self.values:
                    self.entries[name].set(self.values[name])
                else:
                    self.entries[name].set(opts[0] if default is None else default)
                tk.OptionMenu(self.gridFrame, self.entries[name], *opts).grid(row=ix, column=1)
        self.gridFrame.pack()
        tk.Button(self.entryFrame, text="Run", command=self.run).pack(side="bottom")
        self.entryFrame.pack()
    def run(self):
        for (k, v) in self.entries.items():
            self.values[k] = v.get()
        self.entryFrame.destroy()
        self.resultFrame = tk.Frame(self)
        tk.Label(self.resultFrame, text=str(self.values)).pack()
        def reset():
            self.resultFrame.destroy()
            self.createEntries()
        tk.Button(self.resultFrame, text="Return", command=reset).pack()
        self.resultFrame.pack()


def parser():
    ap = argparse.ArgumentParser(
        description="TempEst-NEXT Command Line\n"
        "Carry out full TempEst-NEXT model runs from the "
            "terminal/command prompt with no code. "
            "All arguments are optional, with missing values requested "
            "interactively. Site ID and lat/lon are mutually exclusive ("
            "lat/lon will be ignored if site ID is provided).\n"
            "To launch a GUI instead, run NEXT --gui")
    for (name, (desc, opts, default)) in descs.items():
        name = "--" + name
        ap.add_argument(name, choices=opts, required=False, default=default,
                        help=desc)
    return ap


def cmdrun():
    args = parser().parse_args()
    run(**vars(args))
    
def gui():
    mainframe = tk.Frame(tk.Tk())
    gui = GUI(mainframe)
    mainframe.master.title("TempEst-NEXT GUI")
    mainframe.pack()
    gui.mainloop()

# if __name__ == "__main__":
#     # Test args
#     # --site_type usgs --site 10343500 --start 2024-01-01 --end 2025-12-31 --modpath cache.pickle --datafile sagehen.csv --output sagehen_pred.csv
#     cmdrun()

