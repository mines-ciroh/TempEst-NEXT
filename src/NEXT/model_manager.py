# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:50:08 2024

@author: dphilippus

This file contains a model-manager class that actually handles model building,
etc.  Mostly a wrapper around coef_est.
"""

import NEWT.coef_est as coef_est
from NEWT import Watershed, engines
import rtseason as rts
import pandas as pd
import pickle

class NEXT(object):
    def __init__(self, model):
        # Initialize with a fully-built coefficient estimator model
        self.model = model
        self.newt = None
    
    def from_preproc_data(data):
        # Initialize from pre-processed data
        return NEXT(coef_est.build_model_from_data(data))
    
    def from_data(data):
        # Initialize from raw data
        return NEXT.from_preproc_data(
            coef_est.build_training_data(data)
            )
    
    def to_pickle(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.model, f)
    
    def from_pickle(file):
        with open(file, 'rb') as f:
            return NEXT(pickle.load(f))
    
    def from_default_pickle():
        return NEXT.from_pickle("coefs.pickle")
    
    def make_newt(self, data, start_date="2020-01-01", climyears=0, reset=False,
                  **kwargs):
        # Build a model using provided site data
        if reset or self.newt is None:
            data = data.copy()
            data["date"] = pd.to_datetime(data["date"])
            data["day"] = data["date"].dt.day_of_year
            pdata = coef_est.preprocess(data)
            coefs = coef_est.predict_site_coefficients(self.model,
                                                       pdata)
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
            min_temp = ssn.generate_ts()["actemp"].min()
            min_temp = min_temp if min_temp > 0 else 0
            model = Watershed(seasonality=ssn,
                              at_coef=coefs["threshold_coef_max"].iloc[0],
                              at_day=at_day,
                              dynamic_period=7,
                              dynamic_engine=engines.ThresholdSensitivityEngine(
                                  act_min=min_temp,
                                  coef_min=coefs["threshold_coef_min"].iloc[0],
                                  act_cutoff=coefs["threshold_act_cutoff"].iloc[0],
                                  coef_max=coefs["threshold_coef_max"].iloc[0]
                              ),
                              climate_engine=engines.ClimateCoefficientEngine(self.model, years=climyears),
                              climate_period=365,
                              extra_history_columns=engines.ClimateCoefficientEngine.required_columns,
                              **kwargs
                             )
            self.coefficients = coefs
            self.newt = model
            self.newt.initialize_run(start=start_date)
        return self
    
    def run(self, data, reset=False):
        # Prepare and run model.
        self.make_newt(data, reset)
        return self.newt.run_series(data)
    
    def get_newt(self):
        return self.newt
    
    def make_config(self, outfile, data=None):
        if data is not None:
            self.make_newt(data, False)
        if data is None and self.newt is None:
            raise ValueError("make_config: Must provide data or use pre-fitted model")
        self.newt.to_file(outfile)

