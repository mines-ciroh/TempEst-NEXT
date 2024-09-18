# TempEst-NEXT

TempEst-NEXT (stream temperature estimation: near-term expected temperatures) is a statistical model to hindcast and forecast daily stream temperature for watersheds without local calibration data.  It uses site climate and geography data to estimate the coefficients for the TempEst-NEWT calibrated statistical model.

NEWT and NEXT are part of the TempEst family of models, including [TempEst 1](https://github.com/river-tempest/tempest) for remote sensing-based monthly mean temperatures and TempEst 2 (link) for remote sensing-based daily mean and maximum temperatures, both in ungaged watersheds.  TempEst 2 and NEXT have similar functionality, but TempEst-NEXT is capable of forecasting and disturbance modeling, while TempEst 2 is only for historical estimation (but is much faster and uses less data).  TempEst 1/2 are intended for fast analyses of large-domain historical patterns, while NEWT/NEXT are more focused on in-depth analysis of changes over time, as well as forecasting.

## Quick Start

### Installation

Install from PyPI: `pip install tempest-next`.  The installed module is called NEXT: `import NEXT`.

### Data Preparation

### Model Generation and Execution

NEXT generates a model from a data frame with the appropriate input data.

From there, there are three ways to use the resulting NEWT model:

- For convenience, NEXT can silently generate and run the model, then just return the output timeseries without explicitly returning the model.  This allows you to use NEXT itself as the model directly, rather than as a NEWT-generator with extra steps.
- NEXT can also return the generated NEWT model as-is.  For using that, refer to [TempEst-NEWT](https://github.com/mines-ciroh/TempEst-NEWT) documentation.
- Another convenience approach is to simply write the NEWT model as a configuration file, rather than returning a model.  This is useful to prepare a model for later.

NEXT can also predict coefficients without building a model, which can be used to look at stream thermal regimes in general.

## Design Overview

The task of NEXT is to create standalone NEWT models.  It does not run anything itself, and NEWT is, in turn, independent of NEXT.  This can be used to generate and run models right away in Python, or to export model configuration files for later use (e.g., in nextgen).

NEXT implements the dynamic SCHEMA approach (SCHEMA = "Seasonal Conditions Historical Estimation with Modeled daily Anomaly"; see TempEst 2).  The basic framework is to estimate (1) seasonal conditions, used to predict day-of-year mean, and (2) the sensitivity of stream temperature to weather variation, used to predict the difference between actual temperature and day-of-year mean.  The classic SCHEMA approach in TempEst 2 is stationary; TempEst-NEXT modifies this to allow the model coefficients to shift with changing watershed climate.

NEXT will process an input dataset to estimate model coefficients and will then return a NEWT model object.  It will also create and apply an appropriate modification engine to handle dynamic conditions.

## Science Overview



## Citation

...
