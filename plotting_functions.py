import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from data_manipulation_functions import *


def plot_time_series_with_mean_std_once_per_year(time_series):

    sorted_data_frame = create_yearly_sorted_data(time_series)

    start_year = str(time_series[0].time.dt.year.data)
    end_year = str(time_series[-1].time.dt.year.data)
    dates = make_datetime_once_per_year(start_year, end_year)
    
    means = sorted_data_frame.mean().values
    stds = sorted_data_frame.std().values
    
    time_series.plot()
    plt.plot(dates, means)
    plt.fill_between(dates, means-stds, means+stds,color='blue', alpha=0.2)

    return
