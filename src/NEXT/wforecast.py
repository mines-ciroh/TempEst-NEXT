# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:59:00 2024

@author: dphilippus

This file contains specialized tools for weather forecast retrieval, since
that can be a bit more involved than pulling daymet etc. and I don't want
to clutter up data.py.
"""

import pandas as pd
import numpy as np
import s3fs
import zarr
import xarray as xr
import rioxarray as rio
import cartopy.crs as ccrs
import metpy
import warnings
import getgfs as gfs
import urllib.request as urq
import os


# Direct copy-paste from https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_HowToDownload.html
hrrr_projection = ccrs.LambertConformal(central_longitude=262.5, 
                                       central_latitude=38.5, 
                                       standard_parallels=(38.5, 38.5),
                                        globe=ccrs.Globe(semimajor_axis=6371229,
                                                         semiminor_axis=6371229))


def get_hrrr(date, basin, var="TMP", operator=lambda x: x.max()):
    """
    Retrieve a full forecast run as an xarray.
    date: YYYYMMDD, using the 06z run (which should roughly correspond to day-of in US timezones)
    var: TMP, etc.  Variable to retrieve.

    Returns an xarray dataset for full CONUS, every hour for 48 hours.

    See https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_HowToDownload.html
    """
    offset = 273 if var == "TMP" else 0
    url_base = "s3://hrrrzarr/"
    tmpn = f"sfc/{date}/{date}_06z_fcst.zarr/surface/{var}"
    group_url = url_base + tmpn
    subgroup_url = group_url + "/surface"
    fs = s3fs.S3FileSystem(anon=True)
    # grid = zarr.open(s3fs.S3Map(url_base, s3=fs))["grid/HRRR_chunk_index.zarr"]
    ds = xr.open_mfdataset([s3fs.S3Map(u, s3=fs) for u in [group_url, subgroup_url]], engine="zarr")
    ds["time"] = (ds["time"] - np.timedelta64(6, 'h')).astype("datetime64[D]")  # approximate, but gets it to the right day-ish
    
    ds = ds.rename(projection_x_coordinate="x", projection_y_coordinate="y").\
        metpy.assign_crs(hrrr_projection.to_cf()).\
        metpy.assign_latitude_longitude()
    ds = ds.rio.write_crs(hrrr_projection).rio.clip(basin.geometry.values, basin.crs)
    ds = ds[var] - offset
    return operator(ds[:-1, :, :].groupby("time", squeeze=False))


def hrrr_areal_summary(basin, date, var, new_name, time_operator, 
                       areal_operator=lambda x: x.mean()):
    """
    Generate areal summary of a selected forecast, grouped by time.
    """
    try:
        forecast = get_hrrr(date, basin, var, time_operator)
        summary = forecast.groupby("time", squeeze=False).\
            map(areal_operator).to_pandas().rename(new_name)
        return pd.DataFrame({"date": summary.index, new_name: summary})
    except Exception as e:
        warnings.warn(f"Failed to retrieve HRRR with error: {e}")
        return None


def download_gfs_gribs(start, basepath, time="06", until=384, res="0p25"):
    # Download all GFS gribs for a given forecast.  Note grib parsing requires a POSIX system.
    # start must be as YYYYMMDD or "today".
    if start == "today":
        start = pd.to_datetime(np.datetime64("today")).strftime("%Y%m%d")
    for timestep in range(0, until+3, 3):
        url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{res}.pl?dir=%2Fgfs.{start}%2F{time}%2Fatmos&file=gfs.t{time}z.pgrb2.{res}.f{timestep:03d}&var_DSWRF=on&var_PRATE=on&var_SPFH=on&var_TMP=on&lev_2_m_above_ground=on&lev_surface=on"
        urq.urlretrieve(url, basepath + f"GFS_{res}_{start}_{time}_h{timestep}.grib")

        
def get_gfs_downloaded(basin, start, basepath='./gfs_cache/', var="t2m", new_name="tmax", op = lambda x: x.max(), step_type="instant"):
    """
    Retrieve GFS forecast data by downloading the entire forecast and subsetting
    it. This is relatively fast and reliable, but memory-intensive and requires
    support for cfgrib.

    Parameters
    ----------
    basin : GeoSeries | GeoDataFrame
        Geometry/ies for which to retrieve GFS data. If there are multiple rows,
        there should be an `id` column.
    start : str | datetime
        Start date; date on which forecast was run. Format: "YYYYMMDD".
        Alternatively, may be a Pandas or Python datetime object or anything
        which can be converted to such (e.g., 'today').
    basepath : str, optional
        Directory to which to download/find individual GFS outputs and compiled
        NetCDFs. The default is './gfs_cache/'.
    var : str, optional
        Variable to retrieve (GFS name). The default is "t2m".
    new_name : str, optional
        Output variable name. The default is "tmax".
    op : function series -> number, optional
        Temporal summary function. The default is lambda x: x.max().
    step_type : str, optional
        GFS step type for variables other than air temp. The default is "instant".

    Returns
    -------
    res : pd.Series
        Series of output variable (new_name), indexed by date and (if applicable) ID.
    
    Notes
    -----
    This works better than get_gfs, but it requires ecCodes and therefore may not work on Windows.
    """
    if not (type(start) is str and start.isnumeric()):
        start = pd.to_datetime(start).strftime("%Y%m%d")
    if not basepath.endswith('/'):
        basepath = basepath + '/'
    offset = 273 if var == "t2m" else 0
    if start == "today":
        start = pd.to_datetime(np.datetime64("today")).strftime("%Y%m%d")
    ncpath = basepath + start + ".nc"
    if os.path.exists(ncpath):
        data = xr.open_dataset(ncpath, decode_times=True)[var]
    else:
        contents = [basepath + f for f in os.listdir(basepath) if start in f and f.endswith(".grib")]
        if len(contents) > 0:
            data = xr.concat([xr.open_dataset(file, engine="cfgrib", filter_by_keys={'stepType': step_type})[var] for file in contents], dim="time")
            data.to_netcdf(ncpath)
        else:
            download_gfs_gribs(start, basepath)
            contents = [basepath + f for f in os.listdir(basepath) if start in f and f.endswith(".grib")]
            data = xr.concat([xr.open_dataset(file, engine="cfgrib", filter_by_keys={'stepType': step_type})[var] for file in contents], dim="time")
            data.to_netcdf(ncpath)
    data["time"] = data["valid_time"]
    data = data.rio.write_crs(4326)  # wgs84.  Don't think it matters much at quarter-degree resolution.
    # data["date"] = data["time"].to_series().dt.normalize()
    data = data.assign_coords(date = ("time", data["time"].to_series().dt.normalize()))
    def proc_series(series):
        return op(series.groupby("date")).to_series().rename(new_name) - offset
    def pull_data(geom, df, allow_interp):
        try:
            clip = data.rio.clip(basin.geometry)
            series = clip.groupby("time").mean(dim=["latitude", "longitude"])
        except:  # no data in bounds - watershed too small.  Fall back to interpolation.
            if allow_interp:
                ctr = basin.geometry.iloc[0].centroid.coords[0]
                coords = {"longitude": ctr[0] % 360, "latitude": ctr[1]}  # GFS uses positive degrees east
                series = data.interp(coords)
            else:
                return None
        res = proc_series(series)
        if df:
            return pd.DataFrame(res)
        else:
            return res
    if len(basin) == 1:
        res = pull_data(basin, False, True)
    else:
        # Assume multi-basins are small
        basin['ctr'] = basin.centroid
        interp = data.interp({"latitude": basin['ctr'].y,
                              "longitude": basin['ctr'].x % 360})
        res = basin.groupby('id').apply(
            lambda x: pd.DataFrame(proc_series(
                interp.loc[{'latitude': x['ctr'].iat[0].y,
                            'longitude': x['ctr'].iat[0].x % 360}]
                )).reset_index(),
            include_groups=False
            ).reset_index().loc[:, ['id', 'date', new_name]]
    data.close()
    return res
    

def get_gfs_timestep(fcst, time, lat, lon, varbs):
    # try:
        f = fcst.get(varbs, time, lat, lon).variables
        checker = lambda x: x if x < 1000 else None  # drop bad values
        return {"time": time} | {
                v: checker(f[v].data[0][0][0] - (273 if v=="tmp2m" else 0))
                for v in varbs}
    # except:
    #     return {v: None for v in varbs}


def sphum_to_vp(q, p=1e5):
    # Convert specific humidity, in kg/kg, to vapor pressure.
    return q*p / (0.622 + 0.378 * q)


def get_gfs(basin, date, varbs=["tmp2m", "spfh2m", "dswrfsfc", "pratesfc"],
            new_names=["tmax", "vp", "srad", "prcp"],
            operators={
                "tmax": lambda x: x.max(),
                "vp": lambda x: sphum_to_vp(x.mean()),
                "srad": lambda x: x.mean() * 2, # assume half-day sun; daymet training data srad is daylight only
                "prcp": lambda x: x.sum() * 3600 * 3  # seconds per 3 hours
                }):
    """
    Retrieve GFS forecast for the basin and date.  Summarize by day using operator.
    For now, since GFS is quite coarse-resolution, we just use the centroid
    of the watershed.  Note that the GFS archive doesn't go far back in time,
    so this is for true forecasts only.
    
    Note precip rate is kg/m2/s = mm/s.  Needs conversion to mm/3hr and sum.
    
    This works on Windows, but it's relatively unreliable.
    """
    renamer = {varbs[i]: new_names[i] for i in range(len(varbs))}
    (lon, lat) = basin.geometry.iloc[0].centroid.coords[0]
    fcst = gfs.Forecast("0p25")
    start = np.datetime64(date + " 06:00:00")  # ~start of the day in CONUS
    # Get every 3 hours for forecast window.
    steps = pd.Series(start + np.arange(0, np.timedelta64(383, 'h'),
                              np.timedelta64(3, 'h'))).dt.strftime("%Y-%m-%d %H:%M:%S")
    # Now pull all the data.
    raw = pd.DataFrame([get_gfs_timestep(fcst, time, lat, lon, varbs)
                         for time in steps]).rename(columns=renamer)
    raw["date"] = pd.to_datetime(raw["time"]).dt.normalize()
    return raw.groupby("date", as_index=False)[new_names].agg(operators)
    


def hrrr_series(basin, dates, var, new_name, operator):
    with warnings.catch_warnings(action="ignore"):
        raw = pd.concat([hrrr_areal_summary(basin, date, var, new_name, operator)
            for date in dates])
        return raw.groupby("date").first()
