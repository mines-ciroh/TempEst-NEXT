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
            "forest", "wetland", "developed", "ice_snow", "water",
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
        inp_cols + ["frozen", "cold_prcp"]].mean().merge(
        data.groupby("id", as_index=False)[["prcp", "srad", "vp"]].std(),
        on="id", suffixes=["", "_sd"]).merge(
            # Why different grouping?  apply was dropping id
            data.groupby("id").apply(ssn_df("prcp"), include_groups=False).reset_index(),
            on="id").merge(
                data.groupby("id").apply(ssn_df("tmax"), include_groups=False).reset_index(),
                on="id"
                )
    return predictors




# FWPCA Version
# var_sets = [
#     {"name": "PCA0", "vars": ['tmax', 'vp', 'area', 'elev_min', 'elev', 'developed', 'ws_canopy', 'cold_prcp', 'prcp_index', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 30, "noise":  0.087},
#     {"name": "PCA1", "vars": ['tmax', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'ws_canopy', 'vp_sd', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 30, "noise":  0.036},
#     {"name": "PCA2", "vars": ['vp', 'elev_min', 'wetland', 'developed', 'water', 'canopy', 'ws_canopy', 'vp_sd', 'prcp_phi', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 324, "noise":  0.004},
#     {"name": "PCA3", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'frozen'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7), "lam": 30, "noise":  0.004},
#     {"name": "PCA4", "vars": ['tmax', 'area', 'elev_min', 'elev', 'wetland', 'developed', 'prcp_phi', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 270, "noise":  0.003},
#     {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'ice_snow', 'water', 'ws_canopy', 'cold_prcp', 'vp_sd', 'prcp_index', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 270, "noise":  0.002},
#     {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ws_canopy', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13), "lam": 30, "noise":  0.002},
#     {"name": "PCA7", "vars": ['tmax', 'vp', 'area', 'wetland', 'water', 'ws_canopy', 'vp_sd', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7), "lam": 30, "noise":  0.002},
#     {"name": "PCA8", "vars": ['vp', 'area', 'elev', 'ice_snow', 'water', 'frozen', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6), "lam": 47829690, "noise":  0.001},
# ]


# Restriced variables FWPCA
var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'vp', 'area', 'elev_min', 'developed', 'frozen', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 10, "noise":  0.094},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'area', 'elev_min', 'slope', 'wetland', 'ws_canopy', 'frozen', 'vp_sd', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 30, "noise":  0.038},
    # {"name": "PCA2", "vars": ['tmax', 'prcp', 'elev_min', 'slope', 'ice_snow', 'water', 'frozen', 'prcp_phi', 'prcp_index', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 108, "noise":  0.004},
    # {"name": "PCA3", "vars": ['tmax', 'vp', 'elev_min', 'wetland', 'water', 'frozen', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 810, "noise":  0.004},
    # {"name": "PCA4", "vars": ['tmax', 'vp', 'elev_min', 'slope', 'wetland', 'frozen', 'vp_sd', 'prcp_phi', 'tmax_phi', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 270, "noise":  0.003},
    # {"name": "PCA5", "vars": ['tmax', 'vp', 'area', 'slope', 'wetland', 'water', 'ws_canopy', 'frozen', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 270, "noise":  0.002},
    # {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'slope', 'wetland', 'water', 'ws_canopy', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 270, "noise":  0.002},
    # {"name": "PCA7", "vars": ['tmax', 'area', 'elev_min', 'wetland', 'water', 'frozen', 'vp_sd'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6), "lam": 30, "noise":  0.002},
    # {"name": "PCA8", "vars": ['elev_min', 'wetland', 'developed', 'ice_snow', 'frozen', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 2430, "noise":  0.001},
]




# Regular PCA optimal version
# frozen x tmax_index AND canopy x canopy
# var_sets = [
#     {"name": "PCA0", "vars": ['tmax', 'elev_min', 'slope', 'wetland', 'developed', 'water', 'ws_canopy', 'cold_prcp', 'vp_sd', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 58},
#     {"name": "PCA1", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'ws_canopy', 'vp_sd', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 35},
#     {"name": "PCA2", "vars": ['tmax', 'elev_min', 'elev', 'ice_snow', 'water', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 120},
#     {"name": "PCA3", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'water', 'ws_canopy', 'vp_sd', 'prcp_phi', 'prcp_index', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 300},
#     {"name": "PCA4", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'frozen', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 100},
#     {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'ws_canopy', 'vp_sd', 'prcp_index', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 100},
#     {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'canopy', 'frozen', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 100},
#     {"name": "PCA7", "vars": ['tmax', 'area', 'elev', 'developed', 'vp_sd', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + te(6, 7), "lam": 105},
#     {"name": "PCA8", "vars": ['tmax', 'area', 'elev_min', 'elev', 'wetland', 'water', 'ws_canopy', 'prcp_phi', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 105},
# ]

# Shared
coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([ 12.74045259,   8.88225679, 323.98182474,  70.20482269,
       150.54965511, 218.01908066,   0.73922948,   1.3983637 ,
         0.61015365],
   index=col_order)
# offset = pd.Series([12.740452585874603, 8.88225678976997, 326.75461454940285, 66.92073832790444, 154.07057546145495, 217.70684039087948, 0.7392294811845936, 1.39835333783497, 0.6101532876129115], index=col_order)

# FWPCA
pca_components = np.array([[ 9.99482701e-01,  3.04114428e-02,  5.18896366e-03,
         4.86075504e-03,  3.72667184e-03,  3.61687834e-04,
        -3.83874824e-03,  1.13986046e-03,  5.37298110e-03],
       [ 3.03963926e-02, -9.98768807e-01,  2.07181942e-02,
        -3.41009351e-03,  6.74797899e-03,  7.54246632e-03,
         6.26977656e-03, -2.80604082e-02, -1.29203676e-02],
       [-8.05426117e-03,  2.28125644e-02,  8.87916707e-01,
         2.16406937e-01,  3.80027159e-01,  5.48766005e-02,
        -3.68581099e-02, -1.14984678e-01,  4.66325565e-02],
       [ 3.27831929e-03,  7.01015219e-03,  4.46467373e-01,
        -4.32167844e-01, -7.70817371e-01, -1.40424793e-02,
         3.16908240e-02,  1.62793905e-02, -1.34951892e-01],
       [ 3.86793301e-03,  2.02903702e-02, -4.25788472e-02,
        -7.49394444e-01,  4.35493559e-01,  1.92204451e-01,
         1.36718775e-01, -3.52755204e-01, -2.57765205e-01],
       [-1.80515119e-04,  2.03850886e-02, -9.22150546e-02,
         2.80032083e-01, -2.51328385e-01,  1.26158216e-01,
         7.35498743e-04, -9.06050118e-01,  1.12488320e-01],
       [-2.82307604e-03, -9.91639490e-03,  2.26810480e-02,
        -2.81862772e-01,  1.50429608e-02,  4.04387903e-02,
         1.98416294e-01,  2.82993086e-02,  9.36963809e-01],
       [ 3.24185574e-03,  6.37481126e-03,  1.64990065e-02,
         2.00828513e-01, -3.81796906e-02,  1.19899241e-01,
         9.57242498e-01,  7.00603813e-02, -1.49296247e-01],
       [-1.13981274e-03, -4.84013224e-04, -2.66723796e-02,
         8.10865113e-02, -8.28006039e-02,  9.63257379e-01,
        -1.52342820e-01,  1.86151716e-01,  1.14242886e-02]])

# Fixed dates
scale = pd.Series([  14.42784611,   44.65373443, 2012.31658041, 6585.14924421,
       3894.52975922, 9921.9668602 ,  423.79953993,  254.32678676,
         55.16868856],
                  index=col_order)

# Unfixed dates
# scale = pd.Series([14.44476515,    44.26105306,  2975.71797978,  7373.2794227 ,
#                    4703.27830566, 11743.60917166,   315.77109317,   316.70030422,
#                    55.04709882], index=col_order)

# Standard PCA
# pca_components = np.array([[-0.3635536 , -0.36253655,  0.18978457, -0.4401217 , -0.24528224,
#          0.21700805,  0.38140546, -0.3270315 , -0.38339944],
#        [ 0.38491387, -0.29840726,  0.46560378,  0.12243858,  0.37358605,
#          0.43343509, -0.16075314, -0.381675  ,  0.17907109],
#        [ 0.21515426,  0.42568867,  0.14601904, -0.32076195, -0.53292759,
#          0.31257888,  0.12651994, -0.02602839,  0.49988585],
#        [-0.13209679,  0.1202627 , -0.36689606,  0.14785714,  0.42261061,
#          0.43907167,  0.63080335,  0.10279969,  0.17818129],
#        [-0.07619769, -0.26799642,  0.40814928,  0.20224315, -0.03685127,
#         -0.52885627,  0.4819317 ,  0.06431149,  0.44434513],
#        [-0.22202771,  0.01563487, -0.361087  ,  0.40879354, -0.24807321,
#         -0.06783746, -0.08212493, -0.7319151 ,  0.21065805],
#        [ 0.35877293, -0.61655974, -0.53625048, -0.33023217, -0.08368292,
#         -0.06183943, -0.03335436,  0.0757308 ,  0.27720677],
#        [-0.47126455,  0.1246728 ,  0.0134238 , -0.52115533,  0.45438736,
#         -0.15212244, -0.28078702, -0.14495279,  0.40139952],
#        [-0.5000185 , -0.34072077,  0.08090875,  0.27907185, -0.24756717,
#          0.40409877, -0.3095345 ,  0.4117611 ,  0.24396599]])

# scale = pd.Series([3.9935065545919537, 2.7232799140867754, 15.606734293670241, 32.47165509597058, 21.642058805538785, 17.322166212360017, 0.9551794545253263, 0.9203378017283892, 0.1661541252404621],
#                   index=col_order)





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
