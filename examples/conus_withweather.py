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
from time import sleep, time

logalot = False
logfile = "/u/wy/ch/dphilippus/jobs/logconus.log"
def logger(msg, important=False):
    if important or logalot:
        with open(logfile, "a") as lf:
            lf.write(msg + "\n")
            print(msg)

wbd = WBD("huc12")
nldi = NLDI()

# inp_base = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/daymet_aoi/" # local machine
inp_base = "/scratch/dphilippus/pyproc/daymet_aoi/"  # HPC
# networkf = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/huc12network.csv" # local
networkf = "/scratch/dphilippus/pyproc/huc12network.csv" # HPC
# out_base = "X:/Rio.Data/StreamTemperature/NEXT/ReadyData/CONUS12/" # local
out_base = "/scratch/dphilippus/pyproc/CONUS12/" # HPC
out_raw_base = "/scratch/dphilippus/pyproc/CONUS12Inputs/" # HPC
# pickle = r"C:\Users\dphilippus\OneDrive - Colorado School of Mines\PhD\NEXT\next\src\NEXT\coefs.pickle" # local
pickle = "/u/wy/ch/dphilippus/bins/tempest-next/src/NEXT/coefs.pickle" # HPC
bad_id = "errors.txt"
hasrun = "hucs.txt"

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
            prior = pd.read_csv(networkf, dtype={"huc12": "str", "tohuc": "str"})
            res = pd.concat([prior, res])
        res.to_csv(networkf, index=False)
        return res
    else:
        return pd.read_csv(networkf, dtype={"huc12": "str", "tohuc": "str"})
    

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


def list_hucs():
    existing = [get_huc(f) for f in os.listdir(out_base)]
    with open(hasrun, "w") as f:
        f.write("\n".join(existing))


def proc_weather(fname):
    # Process weather file.
    return pd.read_csv(fname, parse_dates=["datetime"])[["id", "datetime",
                                                            "variable",
                                                            "mean"]].\
        pivot(index=["id", "datetime"], columns="variable", values="mean").\
            reset_index().\
            rename(columns={"datetime": "date"})

def prepare_huc(fname, network, small):
    logger("Preparing HUC")
    # Prepare all data for a given HUC.
    huc = get_huc(fname)
    # Get contributing area and pour point
    (contrib, pour) = get_streaminfo(huc, network)
    (lon, lat) = pour.geometry[0].coords[0]
    logger("Got contributing IDs")
    if small is not None and len(contrib) > small:
        return None
    geoms = wbd.byids('huc12', contrib)
    # Combine all geometries into one polygon
    comb_geom = geoms.assign(id=huc).dissolve(by="id")
    logger("Got geometry")
    wpath = lambda h: inp_base + "HUC-" + h + ".csv"
    # Retrieve subwatershed weather and combine with subwatershed area
    weathers = pd.concat([proc_weather(wpath(h)).
                              assign(area = geoms[geoms['huc12'] == h]['areasqkm'].iloc[0])
                          for h in contrib
                          if os.path.exists(wpath(h))])
    logger("Got weather")
    tot_area = (weathers.groupby('id')['area'].agg("first")).sum()
    # Area-weighted mean
    wdate = weathers["date"]
    weather = weathers.drop(columns=["id", "date"]).apply(
        lambda x: x * x['area'] / tot_area, axis=1).\
                assign(id=huc, date=wdate).\
                groupby(["id", "date"], as_index=False).\
                sum()
    weather['area'] = tot_area * 1e6  # sqkm --> sqm
    weather['lat'] = lat
    weather['lon'] = lon
    logger("Finalized weather")
    # Retrieve required land cover and topo
    lcov = data.lcov_nlcd(comb_geom, 1, 1)
    logger("Got land cover")
    topo = data.topo_3dep(comb_geom, tot_area * 1e6)
    logger("Got topo")
    general_statics = pd.DataFrame(lcov | topo, index = [0])
    # And return the whole enchilada
    res = weather.merge(general_statics, how="cross")
    logger("Finished data prep, starting model run")
    return res


def run_hucs(index, N=100, catchup=False, small=None, skip=0):
    # Iterate through specified HUCs and run them.  Catchup flag can be used to return to watersheds that crashed,
    # especially if it was because of out-of-memory errors (i.e., run one big job with more memory allowed).
    # `small` flag only runs sites with <10 contributing HUCs.
    logger("Starting HUCs run")
    if os.path.exists(bad_id):
        with open(bad_id, "r") as f:
            bad_hucs = [x.strip() for x in f.readlines()]
    else:
        bad_hucs = []
    existing = []
    if os.path.exists(hasrun):
        with open(hasrun, "r") as f:
            existing = [x.strip() for x in f]
    files = [f for f in list_all_inputs() if len(get_huc(f)) == 12 and get_huc(f) not in (bad_hucs + existing)]
    if skip >= len(files) // N:
        return None
    start = time()
    network = build_networkf(files)
    dorun = get_partition(index, files, N)[skip:] if not catchup else files
    logger(f"Got files and network; took {(time() - start):.0f} seconds.  Attempting {len(dorun)} HUCs; ignored {len(bad_hucs)} errors and {len(existing)} already run.", True)
    nx = NEXT.from_pickle(pickle)
    logger("Built NEXT model")
    for f in dorun:
        logger(f"Running: {get_huc(f)}")
        instart = time()
        try:
            output = out_base + get_huc(f) + ".csv"
            raw = out_raw_base + get_huc(f) + ".csv"
            if not os.path.exists(output):
                if os.path.exists(raw):
                    prep = pd.read_csv(raw, dtype={"id": "str"}, parse_dates=["date"])
                else:
                    prep = prepare_huc(f, network, small)
                if prep is not None:
                    if not os.path.exists(raw):
                        prep.to_csv(out_raw_base + get_huc(f) + ".csv", index=False)
                    nx.run(prep, reset=True, use_climate=False)[["id", "lat", "lon", "date", "temp.mod", "area"]].\
                        to_csv(output, index=False)
                    logger(f"Ran one watershed in {(time() - instart):.0f} seconds", True)
        except Exception as e:
            logger(f"Failed file {f} with error {e}; took {(time() - instart):.0f} seconds.", True)
            with open(bad_id, "a") as file:
                file.write(get_huc(f) + "\n")
    logger(f"Ran {len(f)} watersheds in {(time() - start)/60:.0f} minutes", True)


def prep_network(resume=0):
    files = [f for f in list_all_inputs() if len(get_huc(f)) == 12]
    for ix in range(resume, 100):
        build_networkf(get_partition(ix, files, 100), increment=True)
        print(ix)
        sleep(1)
            

if __name__ == "__main__":
    logger("Running the thing")
    if len(sys.argv) >= 3:
        index = int(sys.argv[1])
        N = int(sys.argv[2])
        small = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        skip = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
        logger(f"Running HUCs; small is {small}; skipping {skip}", small)
        run_hucs(index, N, catchup=False, small=small, skip=(skip // N))
    elif len(sys.argv) == 2 and sys.argv[1] == "catchup":
        logger("Running catchup")
        run_hucs(0, 100, True)
    elif len(sys.argv) == 2 and sys.argv[1] == "list":
        # Generate a list of HUCs that have been run and can be skipped.
        list_hucs()
    else:
        print("Usage: python conus_withweather.py <index> <N> [max size] [no. to skip]")
