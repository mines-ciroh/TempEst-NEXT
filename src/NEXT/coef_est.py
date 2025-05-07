# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:03:40 2024

@author: dphilippus

This file handles data preprocessing and coefficient estimation.
"""

from pygam import LinearGAM, s, l, te
import pandas as pd
import numpy as np
from NEWT import analysis, statics, Watershed

rand = np.random.default_rng()

# Used: ['slope', 'elev_min', 'elev', 'area', 'intercept', 'srad_sd', 'cold_prcp', 'prcp', 'prcp_sd', 'srad', 'water', 'wetland', 'developed', 'ssn_phi', 'Intercept', 'ice_snow', 'vp_sd', 'lat', 'tamp', 'frozen', 'lon', 'ssn_index', 'forest']

inp_cols = ["tmax", "prcp", "srad", "vp",
            "area", "elev_min", "elev", "slope",
            "wetland", "developed", "ice_snow", "water",
            "canopy", "ws_canopy",
            "lat", "lon", "date", "day"]
req_cols = inp_cols + ["id"]
training_req_cols = req_cols + ["temperature"]

def ssn_df(col):
    def f(data):
        ctr, I = analysis.circular_season(data["date"], data[col])
        return pd.DataFrame({col + "_phi": [ctr], col + "_index": I})
    return f

def preprocess(data, allow_no_id=True):
    """
    Convert raw input data into appropriate format, with all required covariates.
    """
    if not "id" in data.columns and allow_no_id:
        data["id"] = "null"
    if not all([col in data.columns for col in req_cols]):
        missing = [col for col in req_cols if not col in data.columns]
        raise ValueError(f"Missing columns in input data; required: {req_cols}; missing: {missing}")
    data["frozen"] = data["tmax"] < 0    
    data["cold_prcp"] = data["prcp"] * data["frozen"]
    predictors = data.groupby("id", as_index=False)[
        inp_cols + ["frozen", "cold_prcp"]].mean().assign(
                    snowfrac=lambda x: x["cold_prcp"]/x["prcp"]).drop(
                    columns=["cold_prcp"]).merge(
        data.groupby("id", as_index=False)[["prcp", "srad", "vp"]].std(),
        on="id", suffixes=["", "_sd"]).merge(
            # Why different grouping?  apply was dropping id
            data.groupby("id").apply(ssn_df("prcp"), include_groups=False).reset_index(),
            on="id").merge(
                data.groupby("id").apply(ssn_df("tmax"), include_groups=False).reset_index(),
                on="id"
                )
    return predictors

var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'prcp', 'vp', 'area', 'wetland', 'water', 'snowfrac', 'vp_sd', 'elev', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 32, "noise":  0.815},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'slope', 'wetland', 'developed', 'water', 'ws_canopy', 'frozen', 'snowfrac', 'vp_sd', 'elev', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 58, "noise":  1.003},
    {"name": "PCA2", "vars": ['tmax', 'vp', 'area', 'wetland', 'frozen', 'snowfrac', 'prcp_phi', 'prcp_index', 'elev', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 360, "noise":  0.833},
    {"name": "PCA3", "vars": ['prcp', 'vp', 'area', 'elev', 'wetland', 'developed', 'water', 'canopy', 'ws_canopy', 'frozen', 'vp_sd'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10), "lam": 900, "noise":  0.759},
    {"name": "PCA4", "vars": ['tmax', 'prcp', 'area', 'developed', 'ws_canopy', 'vp_sd', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6), "lam": 3000, "noise":  0.785},
    {"name": "PCA5", "vars": ['tmax', 'vp', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'frozen', 'snowfrac', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10), "lam": 300, "noise":  0.722},
    {"name": "PCA6", "vars": ['tmax', 'vp', 'area', 'slope', 'water', 'frozen', 'snowfrac', 'prcp_phi', 'prcp_index', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 100, "noise":  0.465},
    {"name": "PCA7", "vars": ['tmax', 'prcp', 'vp', 'area', 'wetland', 'ice_snow', 'ws_canopy', 'frozen', 'vp_sd', 'prcp_phi', 'elev', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 3000, "noise":  0.563},
    {"name": "PCA8", "vars": ['tmax', 'vp', 'area', 'developed', 'water', 'canopy', 'ws_canopy', 'snowfrac', 'vp_sd', 'prcp_phi', 'elev', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 300, "noise":  0.497},
]

# Shared
coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([ 12.74045259,   8.88225679, 323.98182474,  70.20482269,
       150.54965511, 218.01908066,   0.73922948,   1.3983637 ,
         0.61015365],
   index=col_order)
scale = pd.Series([ 3.99350655,  2.72327991, 10.54315388, 24.98668967, 17.37028795,
       14.80270361,  0.95517945,  0.92035171,  0.16615406],
                  index=col_order)

pca_components = np.array([[-0.45538956, -0.31404083, -0.10310545, -0.40538516, -0.21287365,
         0.16529587,  0.42218866, -0.24480047, -0.45631392],
       [-0.23117283,  0.40309686, -0.39847152, -0.17666709, -0.35069812,
        -0.49089832,  0.03177782,  0.4769362 , -0.04040952],
       [ 0.24986664, -0.09583789,  0.57766394, -0.31240809, -0.59769297,
        -0.11632982, -0.33375202,  0.02184174, -0.12020955],
       [-0.23641776, -0.40057095,  0.1458864 ,  0.45021412,  0.1756954 ,
        -0.56431082, -0.19792132,  0.03926654, -0.41187823],
       [-0.06619692,  0.16363308,  0.5310667 , -0.22536575,  0.45038622,
         0.06700262,  0.30162264,  0.55538102, -0.17105331],
       [-0.08221556, -0.10865766,  0.29618724,  0.31313608, -0.303729  ,
        -0.2391    ,  0.63764204, -0.05527334,  0.4864056 ],
       [ 0.46765154, -0.64479329, -0.2999677 , -0.3031819 ,  0.1046663 ,
        -0.16495217,  0.13406052,  0.32033365,  0.15777719],
       [-0.2966813 ,  0.01218628,  0.12396925, -0.50827357,  0.34718424,
        -0.41629897, -0.18718909, -0.36389985,  0.4204958 ],
       [-0.55044853, -0.33876265,  0.02243962,  0.0710974 , -0.13037981,
         0.37087001, -0.34584401,  0.40502555,  0.37214333]])




def build_model_from_data(tr_data):
    """
    Prepares a coefficient estimation model from the provided training data.  Training data is assumed to have coefficients listed in col_order,
    which will be converted through PCA.
    """
    vars_local = var_sets.copy()
    # To reduce noise, set "weak-anomaly" dates to their mean.
    means = tr_data[col_order].mean()
    fwt = tr_data["FallWinter"].quantile(0.25)
    sst = tr_data["SpringSummer"].quantile(0.25)
    tr_data.loc[tr_data["FallWinter"] < fwt, "FallDay"] = means["FallDay"]
    tr_data.loc[tr_data["FallWinter"] < fwt, "WinterDay"] = means["WinterDay"]
    tr_data.loc[tr_data["SpringSummer"] < sst, "SpringDay"] = means["SpringDay"]
    tr_data.loc[tr_data["SpringSummer"] < sst, "SummerDay"] = means["SummerDay"]
    # Resume analysis
    X = tr_data.drop(columns=col_order)
    Y = tr_data[["id"] + col_order].set_index("id")
    Y = (Y - offset) / scale  # normalize scale
    Y = Y @ np.transpose(pca_components)
    Y.columns = coef_names
    for vs in vars_local:
        vs["gam"] = LinearGAM(vs["eq"], lam=vs["lam"]).fit(X[vs["vars"]], Y[vs["name"]])
        vs["noise"] = np.sqrt(np.mean((vs["gam"].predict(X[vs["vars"]]) - Y[vs["name"]])**2))
    return vars_local


def predict_site_coefficients(model, data, draw=False):
    """
    Predicts model coefficients using the provided (pre-processed) data for
    a specific site.  Then invert PCA to produce NEWT coefficients.
    If draw is True, generate a random draw.
    """
    # Calibrated "fudge factor": if we don't use all the PCs, we don't capture all the noise. Adjust this to get
    # the correct distribution width.
    noise_factor = 1.5
    if draw:
        predictor = lambda cols, gam, ws, noise: (gam.confidence_intervals(ws[cols], quantiles=[rand.uniform()])[0,0] +
                                                  rand.normal(scale=noise * noise_factor))
    else:
        predictor = lambda cols, gam, ws, noise: gam.predict(ws[cols])[0]
    pcaed = {}
    for cn in coef_names:
        # For when we don't fit all PCAs.
        pcaed[cn] = 0
    for vs in model:
        pcaed[vs["name"]] = predictor(vs["vars"], vs["gam"], data, vs["noise"])
    pcaed = pd.DataFrame(pcaed, index=[0])[coef_names]  # ensure correct order
    inv = pcaed @ pca_components
    inv.columns = col_order
    return inv * scale + offset


def predict_all_coefficients(model, data, draw=False):
    """
    Predicts model coefficients for all sites.
    """
    keepll = "lat" in data.columns and "lon" in data.columns
    keep = data[["id", "elev", "lat", "lon"]] if keepll else data[["id", "elev"]]
    coefs = data.groupby("id").apply(
        lambda x: predict_site_coefficients(model, x, draw),
        include_groups=False)
    return coefs.droplevel(1).merge(keep, how="left", on="id")


def build_training_data(data):
    """
    Prepare a training dataset by fitting watershed models.
    """
    if not all([col in data.columns for col in training_req_cols]):
        raise ValueError(f"Missing columns in input data; required: {training_req_cols}")
    coefs = data.groupby("id").apply(lambda x: 
        Watershed.from_data(x).coefs_to_df().drop(columns=["R2", "RMSE"]) if
        len(x[["day", "temperature"]].dropna()["day"].unique()) >= 181 else None,
        include_groups=False)
    coefs.index = coefs.index.get_level_values("id")
    covar = preprocess(data)
    return coefs.merge(covar, on="id")
