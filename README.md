# TempEst-NEXT

TempEst-NEXT (stream temperature estimation: near-term expected temperatures) is a statistical model to hindcast and forecast daily stream temperature for watersheds without local calibration data.  It uses site climate and geography data to estimate the coefficients for the [TempEst-NEWT](https://github.com/mines-ciroh/TempEst-NEWT) calibrated statistical model.

NEWT and NEXT are part of the TempEst family of models, including [TempEst 1](https://github.com/mines-ciroh/tempest1) for remote sensing-based monthly mean temperatures and [TempEst 2](https://github.com/mines-ciroh/TempEst2/) for remote sensing-based daily mean and maximum temperatures, both in ungaged watersheds.  TempEst 2 and NEXT have similar functionality, but TempEst-NEXT is capable of forecasting and disturbance modeling, while TempEst 2 is only for historical estimation (but is much faster and uses less data).  TempEst 1/2 are intended for fast analyses of large-domain historical patterns, while NEWT/NEXT are more focused on in-depth analysis of changes over time, as well as forecasting.

## Quick Start

### Installation

Install from PyPI: `pip install tempest-next`.  The installed module is called NEXT: `import NEXT`.

### Data Preparation

TempEst-NEXT provides automatic tools for data retrieval, for example: `NEXT.data.full_data("-105.1235:40.5723", start="2020", end="2024", site_type="coordinates")`. You can also provide an input data frame. Required columns are `["id", "tmax", "prcp", "vp",
            "area", "elev_min", "elev", "slope",
            "wetland", "developed", "ice_snow", "water",
            "canopy", "ws_canopy",
            "date", "day"]`, with the addition of a "temperature" column if you wish to train a model.

Column details:

- id: site ID
- day: Julian day of year
- tmax: daily max air temperature (C)
- prcp: daily mean precipitation (mm/day)
- vp: daily mean vapor pressure (Pa)
- area: watershed area (m2) - note *m*2, not km2
- elev_min: minimum watershed elevation (m), i.e. pour point elevation
- elev: mean watershed elevation (m)
- slope: mean watershed slope (m/m)
- wetland, developed, ice_snow, water: land cover abundances in the watershed as a proportion (e.g., wetland=0.2)
- canopy, ws_canopy: mean forest canopy cover as a proportion (e.g., 0.8) for the riparian zone and the whole watershed, respectively

### Model Generation and Execution

NEXT generates a model from a data frame with the appropriate input data.

From there, there are three ways to use the resulting NEWT model:

- For convenience, NEXT can silently generate and run the model, then just return the output timeseries without explicitly returning the model.  This allows you to use NEXT itself as the model directly, rather than as a NEWT-generator with extra steps.
- NEXT can also return the generated NEWT model as-is.  For using that, refer to [TempEst-NEWT](https://github.com/mines-ciroh/TempEst-NEWT) documentation.
- Another convenience approach is to simply write the NEWT model as a configuration file, rather than returning a model.  This is useful to prepare a model for later.

NEXT can also predict coefficients without building a model, which can be used to look at stream thermal regimes in general.

## Detailed Documentation

See the [documentation](https://rivertempest.org/next/next.html) at RiverTempest.org.

## Design Overview

The task of NEXT is to create standalone NEWT models.  It does not run anything itself, and NEWT is, in turn, independent of NEXT.  This can be used to generate and run models right away in Python, or to export model configuration files for later use (e.g., in nextgen).

NEXT implements the dynamic SCHEMA approach (SCHEMA = "Seasonal Conditions Historical Estimation with Modeled daily Anomaly"; see TempEst 2).  The basic framework is to estimate (1) seasonal conditions, used to predict day-of-year mean, and (2) the sensitivity of stream temperature to weather variation, used to predict the difference between actual temperature and day-of-year mean.  The classic SCHEMA approach in TempEst 2 is stationary; TempEst-NEXT modifies this to allow the model coefficients to shift with changing watershed climate.

NEXT will process an input dataset to estimate model coefficients and will then return a NEWT model object.  It will also create and apply an appropriate modification engine to handle dynamic conditions.

## Citation

If you use TempEst-NEXT in your research, please cite:

Philippus, Corona and Hogue. "Kilometer-Resolution Daily Stream Water Temperature Modeling and Forecasting for Ungaged Watersheds at the CONUS Scale." In preparation.
