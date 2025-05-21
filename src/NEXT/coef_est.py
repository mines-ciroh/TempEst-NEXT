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
    {"name": "PCA0", "vars": ['tmax', 'prcp', 'vp', 'area', 'wetland', 'developed', 'water', 'snowfrac', 'vp_sd', 'tmax_phi', 'elev', 'elev_min', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11) + te(12, 13) + te(14, 15), "lam": 52, "noise":  0.749, "scale":  1.16},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'area', 'slope', 'wetland', 'water', 'snowfrac', 'vp_sd', 'tmax_phi', 'elev', 'ws_canopy', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 55, "noise":  0.787, "scale":  1.43},
    {"name": "PCA2", "vars": ['tmax', 'vp', 'slope', 'developed', 'ice_snow', 'water', 'snowfrac', 'prcp_index', 'elev', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10) + te(11, 12), "lam": 120, "noise":  0.705, "scale":  1.87},
    {"name": "PCA3", "vars": ['tmax', 'prcp', 'vp', 'wetland', 'ice_snow', 'water', 'vp_sd', 'prcp_index', 'tmax_phi', 'elev_min', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11) + te(12, 13), "lam": 300, "noise":  0.660, "scale":  1.84},
    {"name": "PCA4", "vars": ['tmax', 'slope', 'water', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'elev', 'elev_min', 'frozen', 'tmax_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + te(7, 8) + te(9, 10) + te(11, 12), "lam": 100, "noise":  0.370, "scale":  1.17},
    {"name": "PCA5", "vars": ['tmax', 'vp', 'area', 'slope', 'wetland', 'developed', 'water', 'snowfrac', 'vp_sd', 'prcp_phi', 'frozen', 'elev_min', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + te(12, 13), "lam": 300, "noise":  0.503, "scale":  1.89}
]



coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([ 12.74045259,   8.88225679, 326.75461455,  70.20482269,
       154.07057546, 217.70684039,   0.73922948,   1.3983637 ,
         0.61015365],
   index=col_order)
scale = pd.Series([ 3.99350655,  2.72327991,  1.        , 24.98668967,  1.        ,
        1.        ,  0.95517945,  0.92035171,  0.16615406],
                  index=col_order)

pca_components = np.array([[-4.75654675e-01, -3.53122432e-01, -5.55111512e-17,
        -3.94265151e-01, -0.00000000e+00, -0.00000000e+00,
         4.38963144e-01, -2.58179195e-01, -4.84011395e-01],
       [-2.96737149e-01,  5.77346238e-01,  0.00000000e+00,
        -3.37987594e-01, -1.38777878e-17,  0.00000000e+00,
         1.36470307e-01,  6.62367291e-01, -8.38343578e-02],
       [-1.98611733e-01,  1.10851201e-01,  3.08086889e-15,
         3.76740053e-01,  9.99200722e-16, -0.00000000e+00,
         7.32213736e-01, -7.91983289e-02,  5.13734360e-01],
       [ 4.49651596e-01, -8.58603556e-03,  3.16413562e-15,
        -7.44913823e-01,  9.15933995e-16,  0.00000000e+00,
         1.90086148e-01, -1.56310705e-01,  4.26939503e-01],
       [-3.48498827e-01,  5.96225184e-01,  3.21964677e-15,
        -9.60257405e-02,  7.77156117e-16,  0.00000000e+00,
        -2.70471037e-01, -6.57554377e-01,  9.11636696e-02],
       [-5.67992152e-01, -4.17288498e-01, -4.44089210e-16,
        -1.55784350e-01, -8.32667268e-17, -0.00000000e+00,
        -3.78500901e-01,  1.77566621e-01,  5.51537509e-01],
       [-0.00000000e+00,  7.75647890e-16, -2.06005185e-01,
        -4.67777530e-16, -2.49364141e-01,  9.46244888e-01,
         7.49593827e-16, -9.52884011e-16,  1.00875968e-15],
       [ 0.00000000e+00,  1.89872290e-15, -9.06606093e-01,
        -1.10633612e-15,  4.12560259e-01, -8.86533946e-02,
         1.65079296e-15, -2.34190242e-15,  2.26705487e-15],
       [-0.00000000e+00,  1.40248804e-15, -3.68276059e-01,
        -8.90472886e-16, -8.76134441e-01, -3.11064602e-01,
         1.25942530e-15, -1.75960297e-15,  1.77068319e-15]])


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
