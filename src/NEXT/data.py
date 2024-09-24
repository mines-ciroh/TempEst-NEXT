# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:47:37 2024

@author: dphilippus

This file contains automatic utilities for data retrieval to easily set
up a NEXT model.  The idea is that, ultimately, you provide a watershed and
NEXT does the rest.

Organization: there are a set of low-level utilities that pull required data
from specific sources, and a high-level data retrieval function that pulls all
required data from specified or default sources.

All retrieval functions provide the required column names, plus a date column,
except for the geometry and topo functions, which return a dictionary.

Training requirements: ['tmax', 'prcp', 'srad', 'vp', 'area', 'elev_min', 'elev', 'slope', 'forest', 'wetland', 'developed', 'ice_snow', 'water', 'lat', 'lon', 'id', 'temperature']
Prediction requirements: everything except temperature
"""

# import NEWT.datatools as dt
import pandas as pd
# import geopandas as gpd
import dataretrieval.nwis as nwis
from pynhd import NLDI
import pydaymet.pydaymet as dym
import pygeohydro.nlcd as nlcd
import py3dep.py3dep as p3d
import xrspatial
import numpy as np

tid = "10343500"
nldi = NLDI()
projstr = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

def gage_geom(usgs_id):
    # Geometries return (geometry, lat, lon, area in km2)
    shp = nldi.get_basins(usgs_id)
    area = (shp.to_crs(projstr).area.rename("area").iloc[0])/1e6
    rc = nwis.get_record(sites=usgs_id, service="site")[["dec_lat_va", "dec_long_va"]]
    return (shp,
            rc["dec_lat_va"].iloc[0], rc["dec_long_va"].iloc[0], area)


def nhd_geom(nhd_id):
    pass

def merit_geom(merit_id):
    # I don't know if this is straightforwardly doable, but useful for non-CONUS.
    pass

geom_fns = {"usgs": gage_geom, "nhd": nhd_geom, "merit": merit_geom}

# Weather requirements: tmax, prcp, srad, vp
wvars = ["tmax", "prcp", "srad", "vp"]
def weather_daymet(geom, start, end):
    return dym.get_bygeom(geom.geometry.iloc[0], (start, end),
                             variables=wvars).\
        groupby("time", squeeze=False).\
            map(lambda x: x.mean()).to_dataframe()[wvars].reset_index().\
                rename(columns={"time": "date"})

def weather_nldas(geom, start, end):
    # Can't do srad, bit of a problem
    pass

def weather_hrrr(geom, start, end):
    pass

weather_fns = {"daymet": weather_daymet, "nldas": weather_nldas,
               "hrrr": weather_hrrr}

# lcov requirements: forest, wetland, developed, ice_snow, water
def lcov_nlcd(geom, start, end):
    avail_years = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
    lcmap = {
        1: "water",
        2: "developed",
        3: "barren",
        4: "forest",
        5: "shrubland",
        7: "herbaceous",
        8: "cultivated",
        9: "wetland",
        10: "unknown",
        -1: "ice_snow"
        }
    
    def nlcd_map(num):
        if num == 12:
            return "ice_snow"
        if num >= 100:
            return "unknown"
        flag = num // 10
        return lcmap[flag]
    
    def convert_nlcd(nlcd_array):
        # nlcd land cover xarray -> dictionary of proportions
        series = nlcd_array.where(lambda x: (x < 127) & (x > 0)).\
            to_series().dropna()
        res = {v: 0.0 for _, v in lcmap.items()}
        for k, v in series.value_counts().items():
            krn = nlcd_map(k)
            res[krn] += v / len(series)
        return res
    
    year = 2021
    dat = nlcd.nlcd_bygeom(geom, years={"cover": [year]})[geom.index[0]]
    return convert_nlcd(dat["cover_" + str(year)])
    # return pd.concat([pd.DataFrame(convert_nlcd(dat["cover_" + str(x)]), index=[x])
    #                   for x in avail_years])

lcov_fns = {"nlcd": lcov_nlcd}

# topo requirements: slope, elev_min, elev
def topo_3dep(geom):
    dem = p3d.get_dem(geom.geometry.iloc[0], 30)
    elev_mean = dem.mean().to_numpy()
    elev_min = dem.min().to_numpy()
    slope = np.sin(xrspatial.slope(dem) * 2 * np.pi / 180).mean().to_numpy()
    return {"elev": elev_mean, "slope": slope, "elev_min": elev_min}

def topo_merit(geom):
    pass

topo_fns = {"3dep": topo_3dep, "merit": topo_merit}

def obs_usgs(usgs_id, start, end):
    # Observed temperature
    pass

obs_fns = {"usgs": obs_usgs}


def full_data(site, start, end,
              site_type="usgs", weather="daymet", lc="nlcd",
              topo="3dep", obs=None):
    geom_fn = geom_fns[site_type]
    weather_fn = weather_fns[weather]
    lcov_fn = lcov_fns[lc]
    topo_fn = topo_fns[topo]
    obs_fn = obs_fns[obs] if obs is not None else None
    (geom, lat, lon, area) = geom_fn(site)
    statics = pd.DataFrame({"id": site, "id_type": site_type,
                            "lat": lat, "lon": lon, "area": area} |
                                      lcov_fn(geom, 1, 1) |
                                      topo_fn(geom),
                                      index = [site])
    dynamics = weather_fn(geom, start, end)#.merge(
        # lcov_fn(geom, start, end), how="left", on="date")
    if obs_fn is not None:
        dynamics = dynamics.merge(obs_fn(site, start, end),
                                  how="left", on="date")
    return statics.merge(dynamics, how="cross")
    

