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

def fcst_inp_name(site, basepath, today):
    return basepath + f"inp_forecast_{fix_sitename(site)}_{today}.csv"

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
    dpath = prep_name(site)
    if os.path.exists(dpath):
        ws_data = pd.read_csv(dpath, dtype={"id": "str"}, parse_dates=["date"])
    else:
        statics = get_statics(site, basepath)
        (geom, lat, lon, area) = NEXT.data.geom_fns[get_site_type(site)](site)
        weather = NEXT.data.weather_daymet(geom, "2018-01-01", "2022-12-31")
        ws_data = statics.merge(weather, how="cross")
        ws_data.to_csv(dpath, index=False)
    return model.make_newt(ws_data, use_climate=False, reset=True).get_newt()


def forecast_inputs(site, basepath):
    earlier = (datetime.date.today() - datetime.timedelta(6)).strftime("%Y-%m-%d")
    today = datetime.date.today().strftime("%Y-%m-%d")
    fcst_path = fcst_inp_name(site, basepath, today)
    end = (datetime.date.today() + datetime.timedelta(16)).strftime("%Y-%m-%d")
    if os.path.exists(fcst_path):
        fcst_input = pd.read_csv(fcst_path, dtype={"id": "str"}, parse_dates=["date"])
    else:
        statics = get_statics(site, basepath)
        (geom, lat, lon, area) = NEXT.data.geom_fns[get_site_type(site)](site)
        recent_weath = NEXT.data.weather_gfs(geom, earlier, today)
        fcst_weath = NEXT.data.weather_gfs(geom, today, end)
        recent = statics.merge(recent_weath, how="cross")
        fcst = statics.merge(fcst_weath, how="cross")
        recent = recent[recent["date"] < pd.to_datetime(today)]
        fcst_input = pd.concat([recent, fcst])
        fcst_input.index = range(len(fcst_input))
        fcst_input.to_csv(fcst_path, index=False)
    return fcst_input


def run_forecast(model, site, basepath):
    today = datetime.date.today().strftime("%Y-%m-%d")
    newt = prepare_model(model, site, basepath)
    fcst_input = forecast_inputs(site, basepath)
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


def run_forecasts(end_id, pickle, basepath, resolution=0.01, dist=1000):
    # Actually returns a list of runnable forecasts.
    (ws, lat, lon, area) = NEXT.data.gage_geom(end_id)
    coords = (lon, lat)
    model = NEXT.NEXT.from_pickle(pickle)
    lines = nldi.navigate_byloc(coords, "upstreamTributaries",
                                source="flowlines", distance=dist)
    points = expand_lines(lines.geometry, resolution)
    clist = [pt.coords[0] for pt in points]
    return [
        lambda: (run_forecast(model, f"{x}:{y}", basepath))
        for (x, y) in clist
        ]

