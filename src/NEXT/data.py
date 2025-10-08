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

Training requirements: ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'forest', 'wetland', 'developed', 'ice_snow', 'water', 'lat', 'lon', 'id', 'temperature']
Prediction requirements: everything except temperature
"""

# import NEWT.datatools as dt
import pandas as pd
# import geopandas as gpd
import dataretrieval.nwis as nwis
from pynhd import NLDI
import pynhd.pynhd as nhd
import pydaymet.pydaymet as dym
import pynldas2.pynldas2 as nldas
import pygeohydro.nlcd as nlcd
import py3dep.py3dep as p3d
import xrspatial
import numpy as np
import geopandas as gpd
import shapely as shp
import NEXT.wforecast as wfc

tid = "10343500"
nldi_inst = [None]
# catchments = nhd.NHDPlusHR("catchment")
projstr = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
# This is on a river, but there are no gages nearby.
testco = (-106.2403, 38.5408)
testco_site = ":".join([str(x) for x in testco])
testpt = gpd.GeoSeries([shp.Point(testco)], crs=4326)
tcomid = "918351"
tusgs = "usgs:10343500"


def nldi():
    # Simple wrapper to generate an NLDI instance on demand.
    if nldi_inst[0] is None:
        nldi_inst[0] = NLDI()
    return nldi_inst[0]


def get_endpoint(lstr):
    # Retrieve endpoint coordinates of a LineString.
    return lstr.coords[-1]


def unroll_coords(reaches):
    """
    Reaches is a GeoDataFrame containing linestring geometries (i.e., reaches).
    Unrolls to extract all coordinate pairs as a list.
    """
    # List of lists of coordinates; need to un-nest
    clists = list(reaches.geometry.apply(lambda x: list(x.coords)))
    return [coords for cl in clists for coords in cl]


def get_watershed(coordinates: tuple):
    """
    Retrieve watershed shape for a given set of coordinates.
    
    Parameters
    ----------
    coordinates : (float, float)
        lon, lat (in E, N)
    
    Returns
    -------
    (GeoDataFrame, float, float, float)
        Watershed boundary, lat, lon, area in m2.
    """
    comid = nldi().comid_byloc(coordinates)["comid"].iloc[0]
    ws = nldi().get_basins(comid, "comid")
    area = ws.to_crs(projstr).area
    return (ws, coordinates[1], coordinates[0], area.iloc[0])

def watershed_geom(site):
    """
    Watershed retriever with appropriate syntax.  Uses a coordinate string
    which is 'lon:lat'.
    """
    coords = [float(x) for x in site.split(":")]
    return get_watershed(coords)


def get_river(site, site_type, dist):
    """
    Get river geometry upstream.
    Site may be a tuple of (lon, lat) ("coordinates"), a USGS ID ("usgs"),
    or a COMID ("nhd").
    """
    riv = None
    if site_type == "usgs":
        site = "USGS-" + site
        ltype = "nwissite"
        riv = nldi().navigate_byid(ltype, site, "upstreamMain", source="flowlines",
                                 distance=dist)
    if site_type == "comid" or site_type == "nhd":
        ltype = "comid"
        riv = nldi().navigate_byid(ltype, site, "upstreamMain", source="flowlines",
                                 distance=dist)
    if site_type == "coordinates":
        riv = nldi().navigate_byloc(site, "upstreamMain", source="flowlines",
                                  distance=dist)
    if riv is None:
        raise ValueError(f"get_river: Invalid site type '{site_type}'. Must be usgs, comid or coordinates.")
    return riv


def centroid(geom):
    """
    Smart centroid function that works for GeoPandas rows _or_ Shapely shapes.

    Parameters
    ----------
    geom : Series or Polygon
        Geometry to be analyzed, either as a series (with .geometry) or a
        Shapely object.

    Returns
    -------
    Centroid as a Shapely point.

    """
    if type(geom) is pd.Series:
        geom = geom.geometry
    if type(geom) is gpd.GeoDataFrame:
        geom = geom.geometry.iloc[0]
    return geom.centroid


def get_upstream_buffer(site, site_type, dist, buffer, geom=None, original=4326):
    """
    Retrieve a buffer around a specified distance upstream (of the nearest NHD+ point).
    Site may be a tuple of (lon, lat) ("coordinates"), a USGS ID ("usgs"),
    or a COMID ("nhd").
    Distance and buffer should be in km.
    """
    if site_type == "coordinates" and type(site) == str:
        site = tuple([float(x) for x in site.split(":")])
    dist *= 1000  # meters
    buffer *= 1000  # meters
    try:
        riv = get_river(site, site_type, dist).to_crs(projstr)  # need projected.  Use the HRRR projection.
        lens = riv.length.cumsum()
        # How many segments do we need?
        howfar = len(lens[lens < dist]) + 1
        if howfar == len(riv):
            # Use the whole river, because we ran out of river.
            subset = riv
        else:
            sofar = lens[lens < dist].max()  # how far we got before
            remaining = dist - sofar  # how much more we need
            subset = riv.head(howfar)
            # The last (most upstream segment)
            last_bit = subset.geometry.iloc[-1]
            # Why are we going backwards?  Because the individual linestrings go
            # from top to bottom, even though the sequence of linestrings goes
            # from bottom to top.  Fun, right?
            last_bit = shp.ops.substring(last_bit,
                                         last_bit.length - remaining,
                                         last_bit.length)
            # Now our river should be exactly the right length.
            subset.loc[howfar-1, "geometry"] = last_bit
        subset = subset.assign(site=str(site)).dissolve("site")
        buf = subset.buffer(buffer)
        return buf.to_crs(original)
    except Exception as e:
        # NLDI probably broke.
        if site_type != "coordinates" or geom is None:
            raise e
        # If we have coordinates, we can go from the pour point to the
        # centroid.
        origin = shp.Point(site)
        ctr = centroid(geom)
        line = gpd.GeoDataFrame(geometry=[shp.LineString([origin, ctr])],
                                crs=original).to_crs(projstr).geometry.iloc[0]
        subset = shp.ops.substring(line, 0, dist)
        # buf = subset.buffer(buffer)
        buf = gpd.GeoDataFrame(geometry=[subset.buffer(buffer)],
                                crs=projstr)
        return buf.to_crs(original)


def get_mean_direction(site, site_type):
    """
    Retrieve the mean direction of the last ~kilometer of the river.
    Site may be a tuple of (lon, lat) ("coordinates"), a USGS ID ("usgs"),
    or a COMID ("nhd").
    """
    if site_type == "coordinates" and type(site) == str:
        site = tuple([float(x) for x in site.split(":")])
    riv = get_river(site, site_type, 1).geometry.iloc[0].reverse()
    (lon1, lat1) = riv.coords[0]  # bottom
    (lon0, lat0) = riv.interpolate(0.01).coords[0]  # approx. 1 km upstream
    dy = lat1 - lat0  # distance north
    dx = lon1 - lon0  # distance east
    # We want the usual map angle (clockwise from north), but we're going to
    # throw in some if/else logic to avoid division by zero, etc.
    if dx != 0 or dy != 0:
        if dx == 0:
            # This is either 0 or 2, so 180 if negative, 0 if positive.
            return 90.0 * (dy - abs(dy)) / dy
        if dy == 0:
            # 90 if dx is positive, 270 if it's negative.
            return 90.0 + 90.0 * (dx - abs(dx)) / dx
        # Now we don't have to worry about zeroes.
        # Now, we work with the arctangent.
        # If dy is positive, its just atan(dx/dy) mod 360.
        # Otherwise, atan(dx/dy) + 180.
        base = (180/3.14)*np.arctan(dx/dy)
        if dy > 0:
            return base % 360
        else:
            return base + 180
    # both zero
    return 0.0
    


def get_upstream(coordinates, dist=1):
    """
    Retrieve upstream flow network from a given coordinates.
    Coordinates: (lon, lat) in WGS84 decimal degrees.  Alternatively,
        a string specifying a USGS gage ID or NHD+ COMID, of the form
        "usgs:12345" or "comid:12345".
    dist: Range in km.  Set to the length of the reach of interest.
    Tributaries are used to construct subwatershed models.  Mainstem is passed
    along as-is for use to extract a riparian buffer, etc.
    """
    if type(coordinates) != tuple and type(coordinates) != str:
        raise ValueError("Invalid coordinate format: get_upstream")
    if type(coordinates) == tuple:
        loc = nldi().feature_byloc(coordinates)["comid"].iloc[0]
        ltype = "comid"
    else:
        [ltype, loc] = coordinates.split(":")
        if ltype not in ["usgs", "comid"]:
            raise ValueError("Invalid site ID category: get_upstream")
        if ltype == "usgs":
            ltype = "nwissite"
            loc = "USGS-" + loc
    main = nldi().navigate_byid(ltype, loc, "upstreamMain", source="flowlines",
                                distance=dist)
    mco = unroll_coords(main)
    tribs = nldi().navigate_byid(ltype, loc, "upstreamTributaries", source="flowlines",
                                distance=dist)
    # Get tributaries that aren't in the mainstem, but do drain into it
    tribs = tribs[-(tribs["nhdplus_comid"].isin(main["nhdplus_comid"])) &
                  (tribs.geometry.apply(get_endpoint).isin(mco))]
    # Retrieve all contributing basins EXCEPT direct mainstem reaches
    if len(tribs) > 0:
        trib_bas = nldi().get_basins(tribs["nhdplus_comid"], "comid").assign(what="tributary")
        trib_bas["endpoint"] = tribs.geometry.apply(get_endpoint).to_list()
        trib_bas["area"] = trib_bas.to_crs(projstr).area
    else:
        trib_bas = None
    # Now figure out the mainstem basin for the top of the reach of interest
    upper = nldi().get_basins(main["nhdplus_comid"].iloc[-1], "comid").assign(what="mainstem")
    return (trib_bas, upper, main)


def combined_areas(coordinates, dist=1):
    (trib, up, main) = get_upstream(coordinates, dist)
    return pd.concat([trib, up])


def buffer(data, buffer):
    """
    Apply a buffer (in meters) to data that is in decimal degrees.
    Buffer is a straightforward Shapely/GeoPandas method in a projected CRS
    (with meters, etc), but it's inappropriate with lat/lon.
    This function simply projects into meters, buffers, and reprojects.
    Assumes original CRS is 4326 (WGS84).
    """
    return data.to_crs(projstr).buffer(buffer).to_crs(4326)


def get_area(geom):
    return geom.to_crs(projstr).area.rename("area").iloc[0]

def gage_geom(usgs_id):
    # Geometries return (geometry, lat, lon, area in m2)
    # geometry should be a Geopandas, not a raw geometry
    shp = nldi().get_basins(usgs_id)
    area = (shp.to_crs(projstr).area.rename("area").iloc[0])
    rc = nwis.get_record(sites=usgs_id, service="site")[["dec_lat_va", "dec_long_va"]]
    return (shp,
            rc["dec_lat_va"].iloc[0], rc["dec_long_va"].iloc[0], area)

def gpkg_geoms(path, cumulative=False):
    # Parse geometries from a geopackage, e.g. for reverse-engineering geometries
    # for an ngen setup
    # Returns dictionary of: {id: (geometry, lat, lon, area)}
    # Assumes columns: id, areasqkm (or tot_drainage_areasqkm), geometry
    # Uses ws area if not cumulative, otherwise total area.
    df = gpd.read_file(path)
    crs = df.crs
    return {
        row.id: (gpd.GeoDataFrame(row, index=row._fields).T.set_geometry("geometry").set_crs(crs),  # make the tuple back into a single-row gdf
                 row.geometry.centroid.y,
                 row.geometry.centroid.x,
                 (row.tot_drainage_areasqkm if cumulative else row.areasqkm) * 1000)  #km2 -> m2
        for row in df.itertuples()
        }

def nhd_geom(nhd_id):
    pass

def merit_geom(merit_id):
    # I don't know if this is straightforwardly doable, but useful for non-CONUS.
    pass

geom_fns = {"usgs": gage_geom, "nhd": nhd_geom, "merit": merit_geom,
            "coordinates": watershed_geom}

# Weather requirements: date, tmax, prcp, vp
wvars = ["tmax", "prcp", "vp"]
def weather_daymet(geom, start, end):
    return dym.get_bygeom(geom.geometry.iloc[0], (start, end),
                             variables=wvars).\
        groupby("time", squeeze=False).\
            map(lambda x: x.mean()).to_dataframe()[wvars].reset_index().\
                rename(columns={"time": "date"})

def weather_nldas(geom, start, end):
    nvars = ["temp", "prcp", "humidity"]
    rename = {nv: wvars[i] for (i, nv) in enumerate(nvars)}
    if type(geom) is gpd.GeoDataFrame:
        geom = geom.geometry.iloc[0]  # doesn't work with a GDF
    data = nldas.get_bygeom(geom, start, end, variables=nvars)
    data["date"] = data.time.dt.date
    data = data.mean(["x", "y"]).to_pandas().groupby("date", as_index=False).agg({"temp": "max", "prcp": "mean", "humidity": "mean"})
    data["temp"] = data["temp"] - 273  # K to C
    data["humidity"] = wfc.sphum_to_vp(data["humidity"])  # convert to vapor pressure
    return data.rename(columns=rename).loc[:, ["date"] + wvars]


hrrr_varnames = {
    "TMP": ("tmax", lambda x: x.max()),
    "PRATE": ("prcp", lambda x: x.mean() * 3600 * 24)  # mm/s --> mm/day
    # "SPFH": ("vp", lambda x: wfc.sphum_to_vp(x.mean()))  # humidity, but not available at surface.
    }

def weather_hrrr(geom, start, end):
    start = np.datetime64(start)
    end = np.datetime64(end)
    dates = pd.Series(np.arange(start, end + np.timedelta64(1, "D")))
    date_str = dates.dt.strftime("%Y%m%d")
    return pd.concat([
        wfc.hrrr_series(geom, date_str, var,
                        hrrr_varnames[var][0],
                        hrrr_varnames[var][1])
        for var in hrrr_varnames
        ], axis=1).reset_index()


def weather_gfs(geom, start, end):
    # The "downloaded" version works better, but may not work on Windows.
    # Only retrieve tmax, since GFS is not suitable for the full timeseires.
    try:
        return wfc.get_gfs_downloaded(geom, start, "./gfs_cache")
    except:
        return wfc.get_gfs(geom, start)

# HRRR can be used for prediction, but not to build coefficient estimation weather,
# as HRRR-Zarr doesn't have srad.
weather_fns = {"daymet": weather_daymet, "nldas": weather_nldas,
        "gfs": weather_gfs, "hrrr": weather_hrrr}

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

def get_canopy(geom, date):
    """
    Get mean canopy cover for the specified geometry and date.
    Date can be anything parseable by numpy.
    It will be moved into the nearest date in the range (2011, 2021), which
    is supported by NLCD.
    """
    year = np.datetime64(date).astype('datetime64[Y]').astype(int) + 1970
    if year < 2011:
        year = 2011
    if year > 2021:
        year = 2021
    cc = list(nlcd.nlcd_bygeom(geom, years={"canopy": [year]}).values()
              )[0][f"canopy_{year}"]
    return float(cc.mean())
    

# topo requirements: slope, elev_min, elev
def topo_3dep(geom, area):
    if area < 1e8:
        dem = p3d.get_dem(geom.geometry.iloc[0], 30)
    else:
        # For larger watersheds, use lower-resolution data retrieval to keep
        # the size manageable.
        dem = p3d.get_map("DEM", geom.geometry.iloc[0], resolution=1000)
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


def geom_static_data(site, site_type, geom, lat, lon,
                     area, lc="nlcd", topo="3dep"):
    lcov_fn = lcov_fns[lc]
    topo_fn = topo_fns[topo]
    return pd.DataFrame({"id": site, "id_type": site_type,
                        "lat": lat, "lon": lon, "area": area,
                        # "flowdir": get_mean_direction(site, site_type)
                        } |
                                  lcov_fn(geom, 1, 1) |
                                  topo_fn(geom, area),
                                  index = [site])


def geom_full_data(site, site_type, geom, lat, lon, area, start, end,
                   weather="daymet", lc="nlcd",
                   topo="3dep", obs=None, buffer_fallback=True):
    # Implements data-retrieval logic for a specified geometry.  See full_data
    # docs.
    # Extra args: if buffer_fallback is True, then, if buffer retrieval fails,
    # just use watershed canopy as riparian canopy.
    weather_fn = weather_fns[weather]
    obs_fn = obs_fns[obs] if obs is not None else None
    if site_type not in ["usgs", "comid", "coordinates"]:
        # For weird stuff like geopackage, just switch it to the coordinates
        site = f"{lon}:{lat}"
        site_type = "coordinates"
    statics = geom_static_data(site, site_type, geom, lat, lon, area)
    if len(start) > 4:
        dynamics = weather_fn(geom, start, end)
        fyr = pd.to_datetime(start).year
        lyr = pd.to_datetime(end).year + 1
    else:
        dynamics = pd.concat([
            weather_fn(geom, str(st) + "-01-01", str(st) + "-12-31")
            for st in range(int(start), int(end)+1)
            ])
        fyr = int(start)
        lyr = int(end) + 1
    ws_canopy = pd.DataFrame([{"year": year, "ws_canopy": get_canopy(geom, str(year))} for year in range(fyr, lyr)])
    buf_canopy = ws_canopy.rename(columns={"ws_canopy": "canopy"})
    try:
        buf = get_upstream_buffer(site, site_type, 1, 0.015, geom=geom)
        buf_canopy = pd.DataFrame([{"year": year, "canopy": get_canopy(buf, str(year))} for year in range(fyr, lyr)])
    except Exception as e:
        if not buffer_fallback:
            raise e
    dynamics["date"] = pd.to_datetime(dynamics["date"])  # just in case
    dynamics["year"] = dynamics["date"].dt.year
    dynamics["day"] = dynamics["date"].dt.dayofyear
    dynamics = dynamics.merge(buf_canopy, on="year"
                              ).merge(ws_canopy, on="year"
                                      ).drop(columns="year")
    if obs_fn is not None:
        dynamics = dynamics.merge(obs_fn(site, start, end),
                                  how="left", on="date")
    return statics.merge(dynamics, how="cross")


def full_data(site, start, end,
              site_type="usgs", weather="daymet", lc="nlcd",
              topo="3dep", obs=None, **kwargs):
    """
    Retrieves all required data for a given site, from start to end.  This
    high-level function allows the user to simply specify sources by name and
    handles the rest.

    Parameters
    ----------
    site : str
        Site identifier, format depending on site type. Examples are a USGS gage ID
        or a coordinate string of the form "lon:lat".
    start, end : str
        either full dates "YYYY-MM-DD" or just years, "YYYY".  If they are just years,
        then each year will be run individually for weather retrieval.  Why does this
        matter to a high-level function? Because running one year at a time uses much
        less memory, so providing years is a good solution if you are running out of memory.
        They must both be in the same format.
    site_type, weather, lc, topo : str, optional
        Specify the type of site and data sources for weather, land cover, and topography.
    **kwargs : passed to geom_full_data
    
    Returns
    -------
    DataFrame
        All required TempEst-NEXT prediction data in a DataFrame for the site.
    
    Notes
    -----
    Currently supported sources are as follows.
    site_type: "usgs" (USGS gage ID) or "coordinates" ("lon:lat").
    weather: "daymet", "nldas", "gfs" (forecasting only), "hrrr".
    lc: "nlcd"
    topo: "3dep"
    """
    if (len(start) == 4) != (len(end) == 4):
        raise ValueError("Start and end must both be YYYY or YYYY-MM-DD.  It appears that one year and one full date were provided.")
    geom_fn = geom_fns[site_type]
    (geom, lat, lon, area) = geom_fn(site)
    return geom_full_data(site, site_type, geom, lat, lon, area, start, end,
                          weather, lc, topo, obs, **kwargs)


def all_data_reaches(coords, dist, buff, start, end, weather="daymet", lc="nlcd",
                  topo="3dep", as_df=False):
    (trib, upper, main) = get_upstream(coords, dist)
    trib["id"] = trib.index
    main_geo = gpd.GeoDataFrame(geometry=buffer(main, buff)).\
        assign(id=main["nhdplus_comid"].iloc[0]).dissolve(by="id").\
            reset_index()
    trib_data = pd.concat(trib.apply(lambda x: geom_full_data(
        x["id"], "tributary", gpd.GeoSeries(x.geometry, crs=4326),
        x["endpoint"][1], x["endpoint"][0],
        x["area"], start, end, weather, lc, topo),
        axis=1).to_list())
    if type(coords) == str:
        coords = get_endpoint(main.geometry.iloc[0])
    upper_data = geom_full_data(upper.index[0], "mainstem_ws", upper, coords[1],
                                coords[0], upper.to_crs(projstr).area.iloc[0],
                                start, end, weather, lc, topo)
    main_data = geom_full_data(main_geo["id"].iloc[0], "mainstem", main_geo,
                               coords[1], coords[0], main_geo.to_crs(projstr).area.iloc[0],
                               start, end, weather, lc, topo)
    if as_df:
        return pd.concat([trib_data, upper_data, main_data])
    return (trib_data, upper_data, main_data)


def all_data_gpkg(path, start, end, weather="daymet", lc="nlcd",
                  topo="3dep", obs=None, cumulative=False, handler=lambda k, g, e: None):
    """
    Wraps full_data to get everything for each site in a geopackage at path.
    """
    gpdata = gpkg_geoms(path, cumulative)
    for k, v in gpdata.items():
        try:
            yield (k, geom_full_data(k, "geopackage", v[0], v[1], v[2], v[3], start, end, weather=weather, lc=lc, topo=topo))
        except Exception as e:
            yield (k, handler(k, v[0], e))
