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
    {"name": "PCA0", "vars": ['tmax', 'prcp', 'vp', 'slope', 'wetland', 'developed', 'water', 'snowfrac', 'vp_sd', 'tmax_phi', 'elev', 'elev_min', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11) + te(12, 13) + te(14, 15), "lam": 95, "noise":  0.763, "scale":  1.17},
    {"name": "PCA1", "vars": ['tmax', 'area', 'wetland', 'water', 'snowfrac', 'vp_sd', 'tmax_phi', 'elev', 'ws_canopy', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 33, "noise":  0.789, "scale":  1.43},
    {"name": "PCA2", "vars": ['tmax', 'vp', 'ice_snow', 'snowfrac', 'elev', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + te(5, 6) + te(7, 8), "lam": 120, "noise":  0.722, "scale":  1.94},
    {"name": "PCA3", "vars": ['tmax', 'prcp', 'vp', 'slope', 'wetland', 'ice_snow', 'water', 'snowfrac', 'vp_sd', 'prcp_index', 'tmax_phi', 'elev_min', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + te(13, 14), "lam": 300, "noise":  0.649, "scale":  1.81},
    {"name": "PCA4", "vars": ['tmax', 'water', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'elev', 'elev_min', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + te(6, 7) + te(8, 9) + te(10, 11), "lam": 100, "noise":  0.374, "scale":  1.18},
    {"name": "PCA5", "vars": ['tmax', 'prcp', 'vp', 'area', 'slope', 'wetland', 'developed', 'ice_snow', 'water', 'snowfrac', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'frozen', 'elev', 'elev_min', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13) + s(14) + te(15, 16) + te(17, 18), "lam": 900, "noise":  0.498, "scale":  1.89},
]

coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([ 12.74728664,   8.88411531, 326.74835165,  70.17814877,
       154.06593407, 217.67912088,   0.73641769,   1.39860397,
         0.60761769],
   index=col_order)
scale = pd.Series([ 3.99956942,  2.72834273,  1.        , 24.9749798 ,  1.        ,
        1.        ,  0.95228087,  0.92086495,  0.16206394],
                  index=col_order)

pca_components = np.array([[ 4.75273148e-01,  3.53017223e-01, -0.00000000e+00,
         3.89964302e-01,  4.33680869e-19,  6.87080990e-30,
        -4.33972379e-01,  2.62777555e-01,  4.89936781e-01],
       [-2.98526209e-01,  5.75733187e-01,  2.22044605e-16,
        -3.51360414e-01,  1.38777878e-17,  1.73472348e-18,
         1.40231899e-01,  6.56430019e-01, -7.34432019e-02],
       [-1.99812934e-01,  1.10527951e-01, -0.00000000e+00,
         4.05686198e-01, -0.00000000e+00, -1.11022302e-16,
         7.37245878e-01, -7.40061442e-02,  4.84013097e-01],
       [-4.53376419e-01, -1.17950445e-03,  1.11022302e-16,
         7.25830343e-01,  1.38777878e-17,  1.11022302e-16,
        -2.21043018e-01,  1.82393601e-01, -4.30687013e-01],
       [ 3.38474188e-01, -6.06349571e-01,  2.77555756e-17,
         7.62238071e-02,  0.00000000e+00,  1.11022302e-16,
         2.58643763e-01,  6.62516936e-01, -7.83585718e-02],
       [-5.70050328e-01, -4.05009279e-01, -3.33066907e-16,
        -1.65083785e-01, -2.77555756e-17,  2.08166817e-17,
        -3.64300551e-01,  1.49581284e-01,  5.73295735e-01],
       [-2.28222730e-17,  2.14447949e-17,  2.44517236e-01,
        -1.50217073e-17,  6.80988603e-01,  6.90265054e-01,
         4.62816197e-18, -3.41355305e-17,  6.05612031e-17],
       [ 7.09295599e-18, -7.35218129e-19, -8.10461498e-02,
        -1.51829973e-17,  7.23736480e-01, -6.85300685e-01,
        -5.71988276e-17,  1.01952746e-17, -5.83613217e-17],
       [-9.10120242e-17,  3.59809750e-17,  9.66251956e-01,
         2.12446194e-17, -1.11624504e-01, -2.32157548e-01,
        -3.43148925e-17,  7.42260318e-17,  1.10636100e-16]])


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
