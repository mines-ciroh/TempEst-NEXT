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

inp_cols = ["tmax", "prcp", "vp",
            "area", "elev_min", "elev", "slope",
            "wetland", "developed", "ice_snow", "water",
            "canopy", "ws_canopy",
            "date", "day"]
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
        data.groupby("id", as_index=False)[["prcp", "vp"]].std(),
        on="id", suffixes=["", "_sd"]).merge(
            # Why different grouping?  apply was dropping id
            data.groupby("id").apply(ssn_df("prcp"), include_groups=False).reset_index(),
            on="id").merge(
                data.groupby("id").apply(ssn_df("tmax"), include_groups=False).reset_index(),
                on="id"
                )
    return predictors

var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'prcp', 'vp', 'ice_snow', 'water', 'snowfrac', 'vp_sd', 'prcp_index', 'tmax_phi', 'elev', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11) + te(12, 13), "lam": 30, "noise":  0.818, "scale":  1.19},
    {"name": "PCA1", "vars": ['tmax', 'prcp', 'vp', 'vp_sd', 'prcp_index', 'tmax_phi', 'elev_min', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + te(7, 8), "lam": 3, "noise":  0.501, "scale":  1.13},
    {"name": "PCA2", "vars": ['tmax', 'prcp', 'wetland', 'water', 'snowfrac', 'vp_sd', 'prcp_index', 'tmax_phi', 'elev', 'ws_canopy', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 21, "noise":  0.622, "scale":  1.45},
    {"name": "PCA3", "vars": ['tmax', 'ice_snow', 'snowfrac', 'tmax_index', 'elev_min', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + te(5, 6), "lam": 4, "noise":  0.555, "scale":  1.52},
    {"name": "PCA4", "vars": ['tmax', 'prcp', 'vp', 'area', 'wetland', 'snowfrac', 'vp_sd', 'tmax_index', 'elev', 'elev_min'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 30, "noise":  0.601, "scale":  1.86},
    {"name": "PCA5", "vars": ['tmax', 'vp', 'area', 'vp_sd', 'prcp_phi', 'tmax_phi', 'elev_min', 'ws_canopy', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 3, "noise":  0.248, "scale":  1.28},
]

coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([ 13.0056967 ,   8.20795656, 326.31318681,  74.14044198,
       156.8       , 214.73186813,   0.58916219,   1.84876426,
         0.72359814],
   index=col_order)
scale = pd.Series([ 3.99160561,  3.43586619,  1.        , 25.38558154,  1.        ,
        1.        ,  1.20308955,  1.52486907,  0.50466527],
                  index=col_order)

pca_components = np.array([[ 3.41756961e-01, -2.61355435e-01, -5.55111512e-17,
         4.20259492e-01,  1.73472348e-18,  3.12560562e-30,
        -3.63303205e-01,  4.70312769e-01,  5.33942091e-01],
       [ 4.71762540e-01,  6.30409106e-01,  1.11022302e-16,
         1.92012911e-01, -0.00000000e+00,  2.16840434e-19,
        -4.64511826e-01, -1.61613840e-01, -3.18221994e-01],
       [-3.87150503e-01,  3.94110963e-01, -1.11022302e-16,
        -4.69134836e-01,  0.00000000e+00, -2.29213584e-29,
        -3.07138668e-01,  6.13781957e-01,  6.03414407e-02],
       [ 6.89037889e-01,  2.42842441e-02, -1.38777878e-16,
        -6.16784244e-01, -5.55111512e-17,  6.93889390e-18,
         3.14530607e-01,  8.58458389e-02,  1.94718384e-01],
       [-3.22430482e-02,  5.40764079e-01,  1.73472348e-17,
         4.14137910e-01, -0.00000000e+00,  1.11022302e-16,
         6.59558509e-01,  2.61545280e-01,  1.77766779e-01],
       [-1.86933003e-01,  2.93153083e-01,  1.66533454e-16,
        -1.20384780e-01,  0.00000000e+00, -1.73472348e-18,
        -1.54785186e-01, -5.47889709e-01,  7.35175653e-01],
       [ 2.34342554e-18, -1.11613983e-16,  5.53604827e-02,
        -7.70407156e-17,  1.84730811e-02,  9.98295529e-01,
        -1.31345009e-16, -5.62176760e-17, -3.77791731e-17],
       [ 4.39134634e-17, -6.00026241e-17,  9.97848118e-01,
        -1.55562006e-16,  3.41576381e-02, -5.59677458e-02,
         3.23505183e-17,  7.93158315e-17,  5.34770085e-17],
       [-2.19955299e-18, -1.44874609e-17, -3.51333141e-02,
         3.71401416e-17,  9.99245716e-01, -1.65423459e-02,
         1.85164759e-17, -1.66091718e-17, -4.95729384e-18]])


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
    tr_data.loc[tr_data["FallWinter"] < fwt, "WinterDay"] = means["WinterDay"]
    tr_data["FallDay"] = means["FallDay"]
    tr_data["SpringDay"] = means["SpringDay"]
    tr_data["SummerDay"] = means["SummerDay"]
    # Resume analysis
    X = tr_data.drop(columns=col_order)
    Y = tr_data[["id"] + col_order].set_index("id")
    Y = (Y - offset) / scale  # normalize scale
    Y = Y @ np.transpose(pca_components)
    Y.columns = coef_names
    Y["FallDay"] = 0
    Y["SpringDay"] = 0
    Y["SummerDay"] = 0
    for vs in vars_local:
        vs["gam"] = LinearGAM(vs["eq"], lam=vs["lam"]).fit(X[vs["vars"]], Y[vs["name"]])
        vs["noise"] = np.sqrt(np.mean((vs["gam"].predict(X[vs["vars"]]) - Y[vs["name"]])**2))
        vs["scale"] = Y[vs["name"]].std() / vs["gam"].predict(X[vs["vars"]]).std()
    return vars_local


def predict_site_coefficients(model, data, draw=False, noise_factor=0.9):
    """
    Predicts model coefficients using the provided (pre-processed) data for
    a specific site.  Then invert PCA to produce NEWT coefficients.
    If draw is True, generate a random draw.
    """
    if draw:
        predictor = lambda cols, gam, ws, noise, scale: (gam.confidence_intervals(ws[cols], quantiles=[rand.uniform()])[0,0] +
                                                          rand.normal(scale=noise * noise_factor))
    else:
        predictor = lambda cols, gam, ws, noise, scale: gam.predict(ws[cols])[0]
    pcaed = {}
    for cn in coef_names:
        # For when we don't fit all PCAs.
        pcaed[cn] = 0
    for vs in model:
        pcaed[vs["name"]] = predictor(vs["vars"], vs["gam"], data, vs["noise"], vs["scale"])
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
        (len(x[["day", "temperature"]].dropna()["day"].unique()) >= 181) and
                                     len(x) >= 730 else None,
        include_groups=False)
    coefs.index = coefs.index.get_level_values("id")
    covar = preprocess(data)
    return coefs.merge(covar, on="id")
