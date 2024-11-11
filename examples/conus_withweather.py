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
import sys
from time import sleep

wbd = WBD("huc12")
nldi = NLDI()

# inp_base = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/daymet_aoi/" # local machine
inp_base = "/scratch/dphilippus/pyproc/daymet_aoi/"  # HPC
# networkf = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/huc12network.csv" # local
networkf = "/scratch/dphilippus/pyproc/huc12network.csv" # HPC
# out_base = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/CONUS12/" # local
out_base = "/scratch/dphilippus/pyproc/CONUS12/" # HPC
# pickle = r"C:\Users\dphilippus\OneDrive - Colorado School of Mines\PhD\NEXT\next\src\NEXT\coefs.pickle" # local
pickle = "/u/wy/ch/dphilippus/bins/tempest-next/src/NEXT/coefs.pickle" # HPC

def list_all_inputs():
    # List all valid input files.
    return glob(inp_base + "HUC*")


def build_networkf(infiles, reset=False, increment=False):
    # Build a network overview that we can work with.  Just maps HUC-12s to
    # their downstream watersheds, if any.
    if reset or increment or not os.path.exists(networkf):
        hucs = [get_huc(inf) for inf in infiles]
        hucs = [h for h in hucs if len(h) == 12]
        res = wbd.byids('huc12', hucs)[["huc12", "tohuc"]]
        if increment and os.path.exists(networkf):
            prior = pd.read_csv(networkf)
            res = pd.concat([prior, res])
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
    # Prepare all data for a given HUC.
    huc = get_huc(fname)
    # Get contributing area and pour point
    (contrib, pour) = get_streaminfo(huc, network)
    (lon, lat) = pour.geometry[0].coords[0]
    geoms = wbd.byids('huc12', contrib)
    # Combine all geometries into one polygon
    comb_geom = geoms.assign(id=huc).dissolve(by="id")
    wpath = lambda h: inp_base + "HUC-" + h + ".csv"
    # Retrieve subwatershed weather and combine with subwatershed area
    weathers = pd.concat([proc_weather(wpath(h)).
                              assign(area = geoms[geoms['huc12' == h]]['areasqkm'].iloc[0])
                          for h in contrib
                          if os.path.exists(h)])
    tot_area = (weathers.groupby('id')['area'].agg("first")).sum()
    # Area-weighted mean
    weather = weathers.drop(columns='id').groupby('datetime').apply(
        lambda x: x * x['area'] / tot_area).assign(id=huc)
    weather['area'] = tot_area * 1e6  # sqkm --> sqm
    weather['lat'] = lat
    weather['lon'] = lon
    # Retrieve required land cover and topo
    general_statics = pd.DataFrame(
        data.lcov_nlcd(comb_geom, 1, 1) |
        data.topo_3dep(comb_geom, tot_area * 1e6),
                       index = [0])
    # And return the whole enchilada
    return weather.merge(general_statics, how="cross")


def run_hucs(index, N=100):
    files = [f for f in list_all_inputs() if len(get_huc(f)) == 12]
    network = build_networkf(files)
    dorun = get_partition(index, files, N)
    nx = NEXT.from_pickle(pickle)
    for f in dorun:
        nx.run(prepare_huc(f, network), reset=True, use_climate=False).\
            to_csv(out_base + get_huc(f) + ".csv")


def prep_network():
    files = [f for f in list_all_inputs() if len(get_huc(f)) == 12]
    for ix in range(100):
        build_networkf(get_partition(ix, files, 100), increment=True)
        sleep(3)
            

if __name__ == "__main__":
    if len(sys.argv) == 3:
        index = int(sys.argv[1])
        N = int(sys.argv[2])
        run_hucs(index, N)
    else:
        print("Usage: python conus_withweather.py <index> <N>")
