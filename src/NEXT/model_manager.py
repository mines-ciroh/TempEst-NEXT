# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:50:08 2024

@author: dphilippus

This file contains a model-manager class that actually handles model building,
etc.  Mostly a wrapper around coef_est.
"""

import NEXT.coef_est as coef_est
from NEWT import Watershed, engines
from NEWT.watershed import Seasonality, Anomaly
import rtseason as rts
import pandas as pd
import pickle
import scipy
import pygam
import numpy as np

def fit_anomgam(data, N=500000):
    # Data has: id, date, tmax, temperature
    # N: number of rows to sample for training.  The dev. sample ~3M is too many.
    at_conv = np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ])
    data["day"] = data["date"].dt.day_of_year
    dailies = data.groupby(["id", "day"])[["temperature", "tmax"]].mean().rename(columns={"temperature": "actemp"})
    anom = data.merge(dailies, on=["id", "day"], suffixes=["", "_mean"])
    anom["delta_st"] = anom["temperature"] - anom["actemp"]
    anom["delta_at"] = anom["tmax"] - anom["tmax_mean"]
    anom = anom[["id", "date", "day", "actemp", "delta_st", "delta_at"]].sort_values(["id", "date"])
    def fit_anom(grp):
        grp["delta_at"] = scipy.signal.fftconvolve(grp["delta_at"],
                                                   at_conv, mode="full")[:-(len(at_conv) - 1)]
        base = grp["delta_at"].abs().mean()
        if base > 0:
            grp["delta_at"] *= grp["delta_st"].abs().mean() / base
            return grp
        else:
            return None
    anom = anom.groupby("id").apply(fit_anom, include_groups=False)
    if len(anom) > N:
        anom = anom.sample(n=N)
    X = anom[["actemp", "delta_at"]]
    y = anom["delta_st"]
    gam = pygam.LinearGAM(pygam.te(0, 1)).fit(X, y)
    noise = np.sqrt(np.mean((gam.predict(X) - y)**2))
    return (gam, noise)
    

class NEXT(object):
    def __init__(self, model, anomgam, anomnoise):
        # Initialize with a fully-built coefficient estimator model
        self.model = model
        self.anomgam = anomgam
        self.anomnoise = anomnoise
        self.newt = None
    
    def from_preproc_data(data, anomgam, anomnoise):
        # Initialize from pre-processed data
        return NEXT(coef_est.build_model_from_data(data), anomgam, anomnoise)
    
    def from_data(data):
        # Initialize from raw data
        (anomgam, anomnoise) = fit_anomgam(data)
        return NEXT.from_preproc_data(
            coef_est.build_training_data(data),
            anomgam,
            anomnoise
            )
    
    def to_pickle(self, file):
        with open(file, 'wb') as f:
            pickle.dump(NEXT(self.model, self.anomgam, self.anomnoise), f)
    
    def from_pickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    def make_components(self, data, lookback=0, draw=False, quantiles=None,
                        internal=False):
        # Build the three model components Seasonality, Anomaly, and periodics.
        # Lookback = lookback window in years. 0 for full timeseries.
        # Internal = return extra components for internal use
        data = data.copy()
        if lookback > 0:
            data = data[-(lookback*365):]
        data["date"] = pd.to_datetime(data["date"])
        data["day"] = data["date"].dt.day_of_year
        pdata = coef_est.preprocess(data)
        coefs = coef_est.predict_site_coefficients(self.model,
                                                   pdata,
                                                   draw)
        at_day = data.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        ssn = rts.ThreeSine(
            Intercept=coefs["Intercept"].iloc[0],
            Amplitude=coefs["Amplitude"].iloc[0],
            SpringSummer=coefs["SpringSummer"].iloc[0],
            FallWinter=coefs["FallWinter"].iloc[0],
            SpringDay=coefs["SpringDay"].iloc[0],
            SummerDay=coefs["SummerDay"].iloc[0],
            FallDay=coefs["FallDay"].iloc[0],
            WinterDay=coefs["WinterDay"].iloc[0]
        )
        season = Seasonality(ssn),
        anom = Anomaly(sensitivity=coefs["at_coef"].iloc[0],
                       anomgam=self.anomgam,
                       quantiles=quantiles),
        dailies = at_day.rename(columns={"day": "period",
                                       "mean_tmax": "tmax"})
        if internal:
            return (season, anom, dailies, coefs)
        return (season, anom, dailies)
    
    def make_newt(self, data, start_date="2020-01-01", reset=False, use_climate=False,
                  climyears=0, draw=False, quantiles=None, **kwargs):
        # Build a model using provided site data
        # If draw is True, generate a random draw instead of the main estimate.
        self.use_climate = use_climate
        self.climyears = climyears
        if reset or self.newt is None:
            (season, anom, dailies, coefs) = self.make_components(data,
                                                                  climyears,
                                                                  draw,
                                                                  quantiles,
                                                                  True)
            climeng = []
            if use_climate:
                climeng = [(365,
                            engines.ClimateEngine(lambda x:
                                                  self.make_components(x, climyears, draw, quantiles)))]
            extcol = []
            if use_climate:
                extcol = [col for col in coef_est.req_cols
                          if not col in Watershed.basic_histcol]
            model = Watershed(season,
                              anom,
                              dailies,
                              climeng,  # engines
                              extcol,  # extra columns
                              **kwargs
                             )
            self.coefficients = coefs
            self.newt = model
            self.newt.initialize_run(period=data["day"].iloc[0])
        return self
    
    def run(self, data, reset=False, **args):
        # Prepare and run model.
        self.make_newt(data, reset=reset, **args)
        if self.use_climate:
            yrs = list(data["date"].dt.year.unique())
            yrs.sort()
            upto = lambda data, yr: data[data["date"].dt.year <= yr]
            exact = lambda data, yr: data[data["date"].dt.year == yr]
            return pd.concat([
                self.make_newt(upto(data, yr), reset=True, use_climate=True,
                               climyears=self.climyears).get_newt().run_series(exact(data, yr))
                for yr in yrs
                ])
        else:
            return self.newt.run_series(data)
    
    def get_newt(self):
        return self.newt
    
    def make_config(self, outfile, data=None):
        if data is not None:
            self.make_newt(data, False)
        if data is None and self.newt is None:
            raise ValueError("make_config: Must provide data or use pre-fitted model")
        self.newt.to_file(outfile)

