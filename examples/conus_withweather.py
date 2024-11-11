# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:45:21 2024

@author: dphilippus

In this example script, we generate lumped model predictions for all CONUS
HUC-12, -10, ..., -2 using preprocessed daymet weather data.  This analysis
has several implementation-specific characteristics:
    - File locations, etc
    - Preprocessed data structure
    - Partitioning into 100 jobs for HPC applications
So it will not run unmodified for a different user, but should be fairly straightforward
to modify.  It could also be switched to directly retrieve and process weather
data from any source.  In my use case, we already have subsetted and summarized
daymet data for all HUCs 12-2.

To keep track of pour points, this needs to start from HUC-12s and recursively
build contributing areas.  To do that:
    1. First, run through all HUC-12s and build a dataset of what drains into
    what.
    2. For a given HUC-12, retrieve the pour point (from huc12pp) and all contributing
    HUC-12s.
    3. Combine weather data as area-weighted mean.
    4. Define combined drainage area and retrieve associated watershed data.
    5. Run model.
"""

from glob import glob
import os
import NEXT.data as data
from NEXT import NEXT
import pandas as pd
from pygeohydro.watershed import WBD
from pynhd.pynhd import NLDI

wbd = WBD("huc12")
nldi = NLDI()

inp_base = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/daymet_aoi/" # local machine
# inp_base = "/scratch/dphilippus/pyproc/daymet_aoi/"  # HPC
networkf = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/huc12network.csv" # local
# networkf = "/scratch/dphilippus/pyproc/huc12network.csv" # HPC


def list_all_inputs():
    # List all valid input files.
    return glob(inp_base + "HUC*")


def build_networkf(infiles, reset=False):
    # Build a network overview that we can work with.  Just maps HUC-12s to
    # their downstream watersheds, if any.
    if reset or not os.path.exists(networkf):
        hucs = [get_huc(inf) for inf in infiles]
        hucs = [h for h in hucs if len(h) == 12]
        res = wbd.byids('huc12', hucs)[["huc12", "tohuc"]]
        res.to_csv(networkf, index=False)
        return res
    else:
        return pd.read_csv(networkf)
    

def get_streaminfo(huc, network):
    # Identify all contributing areas and the pour point.
    contrib = [huc]
    new_upstream = [huc]
    while len(new_upstream) > 0:
        nnu = []
        for up in new_upstream:
            nnu += network[network["tohuc"] == up]["huc12"].to_list()
        contrib += nnu
        new_upstream = nnu
    pp = nldi.getfeature_byid('huc12pp', huc).geometry
    return (contrib, pp)


def get_partition(index, files, N=100):
    # Retrieve partition [index] of N from the file list
    lf = len(files)
    # 1, 101...; 2, 102...; etc.
    return files[index:lf:N]


def get_huc(fname):
    # Extract HUC-code from file path
    last = fname.split("HUC-")[-1]  # after the HUC
    return last.split(".")[0]  # before the .csv


def proc_weather(fname):
    # Process weather file.
    return pd.read_csv(fname, parse_dates=["datetime"])[["id", "datetime",
                                                            "variable",
                                                            "mean"]].\
        pivot(index=["id", "datetime"], columns="variable", values="mean").\
            reset_index().\
            rename(columns={"datetime": "date"})

def prepare_huc(fname, network):
    huc = get_huc(fname)
    (contrib, pour) = get_streaminfo(huc, network)
    geoms = wbd.byids('huc12', contrib)
    wpath = lambda h: inp_base + "HUC-" + h + ".csv"
    weathers = pd.concat([proc_weather(wpath(h)) for h in contrib
                          if os.path.exists(h)])
    comb_geom = geoms.assign(id=huc).dissolve(by="id")
    


