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
var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'vp', 'area', 'elev_min', 'elev', 'developed', 'ws_canopy', 'cold_prcp', 'prcp_index', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 30, "noise":  0.087},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'ws_canopy', 'vp_sd', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 30, "noise":  0.036},
    # {"name": "PCA2", "vars": ['vp', 'elev_min', 'wetland', 'developed', 'water', 'canopy', 'ws_canopy', 'vp_sd', 'prcp_phi', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 324, "noise":  0.004},
    # {"name": "PCA3", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'frozen'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7), "lam": 30, "noise":  0.004},
    # {"name": "PCA4", "vars": ['tmax', 'area', 'elev_min', 'elev', 'wetland', 'developed', 'prcp_phi', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 270, "noise":  0.003},
    # {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'ice_snow', 'water', 'ws_canopy', 'cold_prcp', 'vp_sd', 'prcp_index', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 270, "noise":  0.002},
    # {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ws_canopy', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13), "lam": 30, "noise":  0.002},
    # {"name": "PCA7", "vars": ['tmax', 'vp', 'area', 'wetland', 'water', 'ws_canopy', 'vp_sd', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7), "lam": 30, "noise":  0.002},
    # {"name": "PCA8", "vars": ['vp', 'area', 'elev', 'ice_snow', 'water', 'frozen', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6), "lam": 47829690, "noise":  0.001},
]




"""
# Regular PCA optimal version
# frozen x tmax_index AND canopy x canopy
var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'elev_min', 'slope', 'wetland', 'developed', 'water', 'ws_canopy', 'cold_prcp', 'vp_sd', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 58},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'ws_canopy', 'vp_sd', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 35},
    {"name": "PCA2", "vars": ['tmax', 'elev_min', 'elev', 'ice_snow', 'water', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index', 'canopy', 'ws_canopy'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 120},
    {"name": "PCA3", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'water', 'ws_canopy', 'vp_sd', 'prcp_phi', 'prcp_index', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 300},
    {"name": "PCA4", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'frozen', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 100},
    {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'ws_canopy', 'vp_sd', 'prcp_index', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(8, 9), "lam": 100},
    {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'canopy', 'frozen', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 100},
    {"name": "PCA7", "vars": ['tmax', 'area', 'elev', 'developed', 'vp_sd', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + te(6, 7), "lam": 105},
    {"name": "PCA8", "vars": ['tmax', 'area', 'elev_min', 'elev', 'wetland', 'water', 'ws_canopy', 'prcp_phi', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 105},
]
"""
# Shared
coef_names = ["PCA" + str(i) for i in range(9)]
col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
offset = pd.Series([12.740452585874603, 8.88225678976997, 326.75461454940285, 66.92073832790444, 154.07057546145495, 217.70684039087948, 0.7392294811845936, 1.39835333783497, 0.6101532876129115],
   index=col_order)

# FWPCA
pca_components = np.array([[ 9.99454115e-01,  3.07506109e-02,  3.06067218e-03,
         6.46470573e-03,  6.14249380e-03,  6.79675649e-04,
        -5.15852754e-03,  9.17242028e-04,  5.39149961e-03],
       [ 3.07865649e-02, -9.98357009e-01,  3.48626045e-02,
        -1.60068482e-02,  8.95669215e-03,  7.10782446e-03,
         8.35311119e-03, -2.23326493e-02, -1.28290392e-02],
       [ 4.12226348e-03,  3.42188248e-02,  6.80420782e-01,
        -5.16135278e-01, -4.64218554e-01,  7.91640459e-02,
         1.19135425e-01, -1.71162012e-01, -6.47187583e-02],
       [-9.77329079e-03,  2.45539478e-02,  6.74093266e-01,
         3.41598165e-01,  6.27819599e-01,  5.49107165e-02,
        -9.07408938e-02, -1.20571027e-01,  9.09210142e-02],
       [-2.58006249e-03, -1.46284839e-02,  1.70758298e-01,
         7.16328322e-01, -6.19968365e-01, -1.13983240e-01,
        -1.79232983e-01, -4.15691675e-02,  1.62139691e-01],
       [ 8.94495017e-04,  1.49406330e-03, -7.07241391e-02,
         1.14895214e-01,  3.86393381e-03,  1.33255094e-01,
         8.08824067e-01, -2.08336073e-01,  5.16160840e-01],
       [-2.43066427e-04,  1.28401093e-02, -2.15439269e-01,
        -2.31824827e-02,  1.46340837e-02,  1.82467377e-01,
        -2.70071882e-01, -9.19809427e-01, -1.96688019e-02],
       [-4.65539485e-03, -1.33547129e-02, -1.45331727e-02,
        -2.89274920e-01,  1.45996798e-02, -6.19595184e-02,
        -4.52494501e-01,  1.13493959e-01,  8.33201357e-01],
       [ 1.40033064e-03,  2.10154135e-03,  2.48086232e-02,
        -7.79260851e-02,  7.36391398e-02, -9.60616708e-01,
         1.16041056e-01, -2.27166080e-01, -5.34260786e-03]])
scale = pd.Series([14.44476515,    44.26105306,  2975.71797978,  7373.2794227 ,
        4703.27830566, 11743.60917166,   315.77109317,   316.70030422,
          55.04709882],
                  index=col_order)
"""
# Standard PCA
pca_components = np.array([[-0.3635536 , -0.36253655,  0.18978457, -0.4401217 , -0.24528224,
         0.21700805,  0.38140546, -0.3270315 , -0.38339944],
       [ 0.38491387, -0.29840726,  0.46560378,  0.12243858,  0.37358605,
         0.43343509, -0.16075314, -0.381675  ,  0.17907109],
       [ 0.21515426,  0.42568867,  0.14601904, -0.32076195, -0.53292759,
         0.31257888,  0.12651994, -0.02602839,  0.49988585],
       [-0.13209679,  0.1202627 , -0.36689606,  0.14785714,  0.42261061,
         0.43907167,  0.63080335,  0.10279969,  0.17818129],
       [-0.07619769, -0.26799642,  0.40814928,  0.20224315, -0.03685127,
        -0.52885627,  0.4819317 ,  0.06431149,  0.44434513],
       [-0.22202771,  0.01563487, -0.361087  ,  0.40879354, -0.24807321,
        -0.06783746, -0.08212493, -0.7319151 ,  0.21065805],
       [ 0.35877293, -0.61655974, -0.53625048, -0.33023217, -0.08368292,
        -0.06183943, -0.03335436,  0.0757308 ,  0.27720677],
       [-0.47126455,  0.1246728 ,  0.0134238 , -0.52115533,  0.45438736,
        -0.15212244, -0.28078702, -0.14495279,  0.40139952],
       [-0.5000185 , -0.34072077,  0.08090875,  0.27907185, -0.24756717,
         0.40409877, -0.3095345 ,  0.4117611 ,  0.24396599]])

scale = pd.Series([3.9935065545919537, 2.7232799140867754, 15.606734293670241, 32.47165509597058, 21.642058805538785, 17.322166212360017, 0.9551794545253263, 0.9203378017283892, 0.1661541252404621],
                  index=col_order)
"""




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
