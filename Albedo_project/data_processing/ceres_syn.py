import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.stats import linregress
import os

DATA_PATH = "/Users/mawa7160/dev/data/CERES/"
LEAP_YEAR_OFFSET = (1-0.2425)/2
NON_LEAP_YEAR_OFFSET = 0.2425/2


def make_leap_year_weighted_syn_annual_mean(syn_data, year, offset):
    syn_year = syn_data.sel(time=slice(f"{year-1}-12-31", f"{year+1}-01-01"))
    weights = np.ones(len(syn_year))
    weights[0] = offset
    weights[-1] = offset
    syn_weighted = syn_year.weighted(xr.DataArray(weights, dims='time'))
    return syn_weighted.mean().values