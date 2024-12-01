# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:35:51 2024

@author: dphilippus

This utility script efficiently generates forecasts over a large area.
It uses densely-nested watersheds to do so, which is, it must be noted, an
inefficient approach.
"""

import NEXT
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
from pynhd import NLDI
import sys
import time
nldi = NLDI()

def fix_sitename(site):
    # Coordinate-type site format is lon:lat, which doesn't work as a file name.
    return site.replace(":", "_")

def get_site_type(site):
    if ":" in site:
        return "coordinates"
    else:
        return "usgs"
    
def prep_name(site, basepath):
    return basepath + f"prep_data_{fix_sitename(site)}.csv"

def statics_path(site, basepath):
    return basepath + f"statics_{fix_sitename(site)}.csv"

def get_statics(site, basepath):
    path = statics_path(site, basepath)
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"id": "str"})
    (geom, lat, lon, area) = NEXT.data.geom_fns[get_site_type(site)](site)
    statics = NEXT.data.geom_static_data(site, get_site_type(site),
                                         geom, lat, lon, area)
    statics.to_csv(path, index=False)
    return statics


def prepare_model(model, site, basepath):
    dpath = prep_name(site, basepath)
    if os.path.exists(dpath):
        ws_data = pd.read_csv(dpath, dtype={"id": "str"}, parse_dates=["date"])
    else:
        statics = get_statics(site, basepath)
        (geom, lat, lon, area) = NEXT.data.geom_fns[get_site_type(site)](site)
        weather = NEXT.data.weather_daymet(geom, "2018-01-01", "2022-12-31")
        ws_data = statics.merge(weather, how="cross")
        ws_data.to_csv(dpath, index=False)
    return model.make_newt(ws_data, use_climate=False, reset=True).get_newt()


def forecast_inputs(site, forecast_bp):
    earlier = (datetime.date.today() - datetime.timedelta(6)).strftime("%Y%m%d")
    today = datetime.date.today().strftime("%Y%m%d")
    end = (datetime.date.today() + datetime.timedelta(16)).strftime("%Y-%m-%d")
    (geom, lat, lon, area) = NEXT.data.geom_fns[get_site_type(site)](site)
    recent_weath = pd.DataFrame(NEXT.wforecast.get_gfs_downloaded(geom, earlier, forecast_bp)).reset_index()
    fcst_weath = pd.DataFrame(NEXT.wforecast.get_gfs_downloaded(geom, today, forecast_bp)).reset_index()
    res = pd.concat([fcst_weath, recent_weath]).groupby("date", as_index=False)["tmax"].first().sort_values("date")
    res["lat"] = lat
    res["lon"] = lon
    res["id"] = site
    return res


def run_forecast(model, site, basepath, forecast_bp):
    today = datetime.date.today().strftime("%Y-%m-%d")
    newt = prepare_model(model, site, basepath)
    fcst_input = forecast_inputs(site, forecast_bp)
    fcst_input = pd.concat([fcst_input, pd.DataFrame(fcst_input.iloc[22]).T])
    fcst_input.index = range(len(fcst_input))
    fcst_input.loc[len(fcst_input)-1, "date"] = fcst_input["date"].iloc[-2] + np.timedelta64(1, 'D')
    fcst_input["date"] = pd.to_datetime(fcst_input["date"])
    forecast = newt.run_series(fcst_input)[["id", "lat", "lon", "date", "actemp", "anom", "temp.mod"]]
    forecast.to_csv(basepath + f"forecast_{fix_sitename(site)}_{today}.csv", index=False)
    return forecast


def expand_line(line, resolution):
    count = line.length / resolution
    if count < 0.5:
        steps = []   # we don't want super-dense points on very short reaches
    elif count < 1.5:
        steps = [line.length]  # just the bottom
    else:
        steps = np.arange(resolution, line.length, resolution)
    return gpd.GeoSeries(line.interpolate(steps))

def expand_lines(geometry, resolution):
    return pd.concat([expand_line(g, resolution) for g in geometry])


def run_forecasts(end_id, pickle, basepath, forecast_bp, resolution=0.01, dist=1000):
    # Actually returns a list of runnable forecasts.
    (ws, lat, lon, area) = NEXT.data.gage_geom(end_id)
    coords = (lon, lat)
    model = NEXT.NEXT.from_pickle(pickle)
    lines = nldi.navigate_byloc(coords, "upstreamTributaries",
                                source="flowlines", distance=dist)
    points = expand_lines(lines.geometry, resolution)
    clist = [pt.coords[0] for pt in points]
    return [
        (lambda x, y: (lambda: run_forecast(model, f"{x}:{y}", basepath, forecast_bp)))(a, b)
        for (a, b) in clist
        ]


if __name__ == "__main__":
    args = sys.argv
    if len(args) >= 3:
        end_id = args[1]
        basepath = args[2]
        index = int(args[3]) if len(args) >= 4 else 0
        N = int(args[4]) if len(args) >= 5 else 1
        res = float(args[5]) if len(args) >= 6 else 0.01
        dist = int(args[6]) if len(args) >= 7 else 1000
        max_t = int(args[7]) if len(args) >= 8 else -1
        runs = run_forecasts(end_id, "coefs.pickle", basepath, "/scratch/dphilippus/gfs/", res, dist)
        if N > 1:
            total = len(runs)
            dorun = runs[index:total:N]
        else:
            dorun = runs
        for run in run_queue:
            run()
    else:
        print("""Arguments: <USGS gage> <output basepath> [<partition index> <total #partitions>] [resolution=0.01] [distance=1000] [max runtime (seconds)]
    Runs nested watershed forecasts upstream of the specified gage.
    After completing a run, switch to the Notebook to quickly retrieve all
    runs.""")

