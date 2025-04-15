import xarray as xr
import numpy as np
from scipy.optimize import minimize

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


def make_annual_mean_from_daily_syn_data_four_offsets(syn_data, year, offsets):
    syn_year = syn_data.sel(time=slice(f"{year-1}-12-31", f"{year+1}-01-01"))
    offset_before_day, offset_first_day, offset_last_day, offset_after_day = offsets
    weightings = np.ones(len(syn_year))
    weightings.put(0, offset_before_day)
    weightings.put(1, offset_first_day)
    weightings.put(-2, offset_last_day)
    weightings.put(-1, offset_after_day)
    syn_weighted = syn_year.weighted(xr.DataArray(weightings, dims='time'))
    return syn_weighted.mean().values


def annual_mean_from_daily_data_two_extra_days_two_offset(year_of_data, optimized_params):
    offset_early, offset_late = optimized_params
    weightings = np.ones(len(year_of_data))
    weightings.put(0, offset_early)
    weightings.put(-1, offset_late)
    syn_weighted = year_of_data.weighted(xr.DataArray(weightings, dims='time'))
    return syn_weighted.mean().values


def annual_mean_from_daily_data_two_extra_days_one_offset(year_of_data, optimized_params):
    offset = optimized_params
    weightings = np.ones(len(year_of_data))
    weightings.put(0, offset)
    weightings.put(-1, offset)
    syn_weighted = year_of_data.weighted(xr.DataArray(weightings, dims='time'))
    return syn_weighted.mean().values


def annual_mean_from_monthly_data_two_extra_days_two_offset(year_of_data, optimized_params):
    offset_early, offset_late = optimized_params
    data_14_month = year_of_data.resample(time='M').mean()
    weightings = data_14_month.time.dt.days_in_month.values.astype(float)
    weightings.put(0, offset_early)
    weightings.put(-1, offset_late)
    syn_weighted = data_14_month.weighted(xr.DataArray(weightings, dims='time'))
    mean = syn_weighted.mean().values
    return mean


def annual_mean_from_monthly_data_two_extra_days_one_offset(year_of_data, optimized_params):
    offset = optimized_params
    data_14_month = year_of_data.resample(time='M').mean()
    weightings = data_14_month.time.dt.days_in_month.values.astype(float)
    weightings.put(0, offset)
    weightings.put(-1, offset)
    syn_weighted = data_14_month.weighted(xr.DataArray(weightings, dims='time'))
    mean = syn_weighted.mean().values
    return mean


def optimize_against_global_year(model_function, ref_data, year_array, prediction_data, initial_guess, method="Nelder-Mead"):
    def objective(optimized_params):
        y_est_array = np.array([])
        for year in year_array:
            year_int = year.astype(int)
            data_year = ref_data.sel(time=slice(f"{year_int - 1}-12-31", f"{year_int + 1}-01-01"))
            y_est_array = np.append(y_est_array, model_function(data_year, optimized_params))
        error = prediction_data - y_est_array
        return np.sum(error ** 2)

    result = minimize(objective, initial_guess, method=method)
    return result


def optimize_with_global_year(model_function, ref_data, year_array, prediction_data, initial_guess):
    def objective(optimized_params):
        y_est_array = np.array([])
        y_pred_array = np.array([])
        for year in year_array:
            year_int = year.astype(int)
            pred_year = prediction_data.sel(time=slice(f"{year_int - 1}-12-31", f"{year_int + 1}-01-01"))
            data_year = ref_data.sel(time=slice(f"{year_int - 1}-12-31", f"{year_int + 1}-01-01"))
            y_est_array = np.append(y_est_array, model_function(data_year, optimized_params))
            y_pred_array = np.append(y_pred_array, model_function(pred_year, optimized_params))
        error = y_pred_array - y_est_array
        return np.sum(error ** 2)

    result = minimize(objective, initial_guess)
    return result
