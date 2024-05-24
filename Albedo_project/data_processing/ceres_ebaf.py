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

def slice_dataset_with_year_and_month(ceres_dataset: xr.Dataset, start_yr: str, start_mon, end_yr, end_mon):
    """
    Wrapper function to help select by month and year on a Pandas DataFrame
    """
    start = start_yr + "-" + start_mon
    end = end_yr + "-" + end_mon
    data = ceres_dataset.sel(time=slice(start, end))
    return data


def create_lat_weights(method: int, data_path: str = DATA_PATH, desired_lat_locs=None):
    if method == 1:
        lat_weights = pd.read_csv(data_path + "lat_weights.csv", header=0)
        lat_weights = lat_weights.to_numpy()

        lats = lat_weights[0:-1, 0]
        lats = lats.astype(float)
        vals = lat_weights[0:-1, 1]
        vals = vals.astype(float)
    elif method == 2:
        lats = np.arange(-90, 90, 1)
        cosine_vals = np.cos(lats / 180 * np.pi)
        normalized_cosine = cosine_vals / np.sum(cosine_vals)
        vals = normalized_cosine
    else:  # No weighting
        lats = np.arange(-90, 90, 1)
        vals = np.ones_like(lats)
    if desired_lat_locs is None:
        return vals
    else:
        interp_vals = np.interp(desired_lat_locs, lats, vals)
        return interp_vals


def apply_spatial_weights(dataset, low_lat=-90.0, high_lat=90.0, method=1):
    lat_weights = create_lat_weights(method, desired_lat_locs=dataset.lat)

    space_weights = xr.DataArray(data=lat_weights, coords=[dataset.lat], dims=["lat"])
    space_weights.name = "weights"

    slice_of_data = dataset.sel(lat=slice(low_lat, high_lat))
    weighted_slice = slice_of_data.weighted(space_weights)

    mean_slice = weighted_slice.mean(("lat", "lon"))
    return mean_slice


def calculate_weighted_annual_mean(dataset, day_in_months):
    specific_t_weighted = dataset.groupby("time.year").sum().copy()

    time_weights = np.zeros(len(dataset))
    year_count = int(np.floor(len(dataset) / 12))
    difference = len(specific_t_weighted) - year_count
    if (difference > 0):
        specific_t_weighted = specific_t_weighted[:-difference]
    for i in range(year_count):
        time_weights[i * 12:(i + 1) * 12] = day_in_months[i * 12:(i + 1) * 12] / np.sum(
            day_in_months[i * 12:(i + 1) * 12])
        if (len(dataset.shape) == 3):
            specific_t_weighted[i] = np.average(dataset[i * 12:(i + 1) * 12, :, :],
                                                weights=time_weights[i * 12:(i + 1) * 12], axis=0)
        else:
            specific_t_weighted[i] = np.average(dataset[i * 12:(i + 1) * 12],
                                                weights=time_weights[i * 12:(i + 1) * 12])
    # Add a datetime coordinate
    years_date_time = []
    for year in specific_t_weighted.year.values:
        years_date_time.append(datetime(year, 6, 1))
    specific_t_weighted.coords["time"] = ("year", years_date_time)

    return specific_t_weighted


def calculate_weighted_running_mean(dataset, weights, running_length=12, use_shifting_weights=False):
    local_weights = weights.values.astype("float").copy()
    if len(dataset.shape) == 3:
        # Create the running avg data on just the time axis
        if not use_shifting_weights:
            t_weighted = np.array([np.average(dataset[x:x + running_length],
                                              weights=local_weights[x:x + running_length], axis=0) for x in
                                   range(len(dataset) - running_length)])
        else:
            t_weighted = np.ones(
                (len(dataset[:, 0, 0]) - running_length, len(dataset[0, :, 0]), len(dataset[0, 0, :])))
            for x in range(len(dataset) - running_length):
                idx = int(x + running_length / 2)
                left = idx - int(running_length / 2)
                right = idx + int(running_length / 2)
                time_weights = (local_weights[left:right]).copy()
                if (time_weights == 29).sum() == 1:
                    time_weights[0] = time_weights[0] - LEAP_YEAR_OFFSET
                    time_weights[running_length - 1] = time_weights[running_length - 1] - LEAP_YEAR_OFFSET
                else:
                    time_weights[0] = time_weights[0] + NON_LEAP_YEAR_OFFSET
                    time_weights[running_length - 1] = time_weights[running_length - 1] + NON_LEAP_YEAR_OFFSET

                t_weighted[x, :, :] = np.average(dataset[left:right], weights=time_weights, axis=0)
    # TODO not sure if this works or when I needed this
    # elif len(dataset.shape) == 1:
    #     if not use_shifting_weights:
    #         t_weighted = np.array([np.average(dataset[x:x + running_length],
    #                                           weights=weights[x:x + running_length]) for x in
    #                                range(len(dataset) - running_length)])
    #     else:
    #         t_weighted = np.empty((len(dataset),1))
    #         for x in range(len(dataset) - running_length):
    #             idx = int(x + running_length / 2)
    #             left = idx - int(running_length / 2)
    #             right = idx + int(running_length / 2)
    #             time_weights = weights[left:right]
    #             if (time_weights == 29).sum() == 1:
    #                 time_weights[0] = time_weights[0] - 3 / 8
    #                 time_weights[running_length - 1] = time_weights[running_length - 1] - 3 / 8
    #             else:
    #                 time_weights[0] = time_weights[0] + 1 / 8
    #                 time_weights[running_length - 1] = time_weights[running_length - 1] + 1 / 8
    #             np.append(t_weighted, np.average(dataset[left:right], weights=time_weights))
    else:
        raise Exception("Problem in data shape in running weighted running average")

    # Shorten the original data to match the running avg length with the date being the center point of the running avg
    left = int(running_length / 2 - 1)
    right = int(len(dataset) - running_length / 2 - 1)
    specific_t_weighted = dataset[left:right, :, :]
    specific_t_weighted.data = t_weighted

    return specific_t_weighted


def apply_time_averaging(dataset, averaging_method=2, feb_leap_year_correction=27.638,
                         feb_non_leap_year_correction=28.454, running_length=0):
    month_length = dataset.time.dt.days_in_month

    # Incorrectly calculate with equal weight for each month
    if (averaging_method == -1):
        month_length_equal = 30*np.ones_like(month_length)
        specific_t_weighted = calculate_weighted_annual_mean(dataset, month_length_equal)
        return specific_t_weighted

    # When not using a leap year correction
    if (averaging_method == 0):
        # First case no running average
        if (running_length == 0):
            specific_t_weighted = calculate_weighted_annual_mean(dataset, month_length)
        # Otherwise there is a running average
        else:
            specific_t_weighted = calculate_weighted_running_mean(dataset, month_length, running_length)

        return specific_t_weighted

    # Use the February only correction (Matt method)
    if (averaging_method == 1):
        month_length.values = month_length.values.astype("float")
        month_length[month_length == 28] = feb_non_leap_year_correction
        month_length[month_length == 29] = feb_leap_year_correction

        if (running_length == 0):
            specific_t_weighted = calculate_weighted_annual_mean(dataset, month_length)
        else:
            specific_t_weighted = calculate_weighted_running_mean(dataset, month_length, running_length)
        return specific_t_weighted

    # Jake Method of weighting edge months
    if (averaging_method == 2):
        month_length.values = month_length.values.astype("float")
        if running_length == 0:
            year_count = int(np.floor(len(dataset) / 12))
            for i in range(year_count):
                year_months = month_length[12 * i:12 * (i + 1)]
                if (year_months.where(year_months.isin(29), drop=True).size == 1):
                    month_length.values[i * 12] = month_length.values[i * 12] - LEAP_YEAR_OFFSET
                    month_length.values[(i + 1) * 12 - 1] = month_length.values[(i + 1) * 12 - 1] - LEAP_YEAR_OFFSET
                else:
                    month_length.values[i * 12] = month_length.values[i * 12] + NON_LEAP_YEAR_OFFSET
                    month_length.values[(i + 1) * 12 - 1] = month_length.values[(i + 1) * 12 - 1] + NON_LEAP_YEAR_OFFSET
            specific_t_weighted = calculate_weighted_annual_mean(dataset, month_length)
        elif running_length == 12:
            specific_t_weighted = calculate_weighted_running_mean(dataset, month_length,
                                                                  running_length, use_shifting_weights=True)
        else:
            specific_t_weighted = calculate_weighted_running_mean(dataset, month_length,
                                                                  running_length)
        return specific_t_weighted


def create_hemisphere_data(dataset, time_weighting=2, space_weighting=1, start_yr="2001",
                           start_mon="01", end_yr="2023", end_mon="01", ly_feb=27.65, nly_feb=28.45,
                           running_length=0, use_mask=False, mask_data=None,
                           masked_name=None, unmasked_name=None):
    return create_zonal_data(dataset, time_weighting=time_weighting, space_weighting=space_weighting,
                             start_yr=start_yr, end_yr=end_yr, start_mon=start_mon, end_mon=end_mon,
                             ly_feb=ly_feb, nly_feb=nly_feb, running_length=running_length,
                             use_mask=use_mask, mask_data=mask_data,
                             masked_name=masked_name, unmasked_name=unmasked_name,
                             zonal_cuts=1, zonal_names=["sh", "nh"])


def create_zonal_data(dataset, start_yr="2001", start_mon="01", end_yr="2023", end_mon="01", space_weighting=1,
                      time_weighting=2, running_length=0, ly_feb=27.65, nly_feb=28.45,
                      zonal_cuts:int=1, zonal_names:list=None,
                      use_mask:bool=False, mask_data=None, masked_name=None, unmasked_name=None):
    cleaned_dat = slice_dataset_with_year_and_month(dataset, start_yr, start_mon, end_yr, end_mon)
    specific_t_weighted = apply_time_averaging(cleaned_dat, averaging_method=time_weighting,
                                               running_length=running_length, feb_leap_year_correction=ly_feb,
                                               feb_non_leap_year_correction=nly_feb)
    # Create Global Average
    ds_new = xr.Dataset({"global":apply_spatial_weights(specific_t_weighted,
                                                        method=space_weighting)})
    # Generate Zonal Data
    zonal_step_size = 180 / (zonal_cuts + 1)
    if zonal_names is None:
        zonal_names = [f"{-90 + (zonal_step_size)*(x+1)-zonal_step_size/2}" for x in range(zonal_cuts+1)]
    if use_mask:
        mask_name = f"{masked_name}_mask"
        specific_t_weighted.coords[mask_name] = (('lat', 'lon'), mask_data.data)
        mask_t_weighted = specific_t_weighted.where(specific_t_weighted[mask_name] == 1)
        unmasked_t_weighted = specific_t_weighted.where(specific_t_weighted[mask_name] == 0)

        # Make Global Masked Values
        all_ts_masked_mean = apply_spatial_weights(mask_t_weighted, method=space_weighting)
        all_ts_unmasked_mean = apply_spatial_weights(unmasked_t_weighted, method=space_weighting)

        ds_new = ds_new.assign({f"global_{masked_name}": all_ts_masked_mean,
                                f"global_{unmasked_name}": all_ts_unmasked_mean})

    for i in range(zonal_cuts+1):
        start_lat = -90 + zonal_step_size*i
        stop_lat = start_lat + zonal_step_size
        zonal_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=start_lat, high_lat=stop_lat,
                                              method=space_weighting)
        ds_new = ds_new.assign({f"{zonal_names[i]}": zonal_ts_mean})

        if use_mask:
            masked_zonal_ts_mean = apply_spatial_weights(mask_t_weighted, low_lat=start_lat, high_lat=stop_lat,
                                                         method=space_weighting)
            unmasked_zonal_ts_mean = apply_spatial_weights(unmasked_t_weighted, low_lat=start_lat, high_lat=stop_lat,
                                                    method=space_weighting)
            ds_new = ds_new.assign({f"{zonal_names[i]}_{masked_name}": masked_zonal_ts_mean,
                                    f"{zonal_names[i]}_{unmasked_name}": unmasked_zonal_ts_mean})
    return ds_new


def lin_regress_slope(data_in):
    results = linregress(data_in.year, data_in[dict(paired_points=0)]).slope
    return results


def lin_regress_pValue(data_in):
    results = linregress(data_in.year, data_in[dict(paired_points=0)]).pvalue
    return results


def lin_regress_rSquared(data_in):
    results = linregress(data_in.year, data_in[dict(paired_points=0)]).rvalue
    return results ** 2


def create_trends_at_each_location(dataset, time_weighting, start_yr="2001",
                                   start_mon="01", end_yr="2022", end_mon="01", ly_feb=27.65, nly_feb=28.45):
    cleaned_dat = slice_dataset_with_year_and_month(dataset, start_yr, start_mon, end_yr, end_mon)
    specific_t_weighted = apply_time_averaging(cleaned_dat, averaging_method=time_weighting,
                                               running_length=0, start_mon=start_mon, feb_leap_year_correction=ly_feb,
                                               feb_non_leap_year_correction=nly_feb)

    trends = np.zeros_like(specific_t_weighted[0])
    pvals = np.zeros_like(specific_t_weighted[0])
    r2 = np.zeros_like(specific_t_weighted[0])

    lat_length = len(specific_t_weighted.lat.values)
    lon_length = len(specific_t_weighted.lon.values)
    for i in range(lat_length):
        for j in range(lon_length):
            current_lat = specific_t_weighted[0].lat.values[i]
            current_lon = specific_t_weighted[0].lon.values[j]
            reg_results = linregress(specific_t_weighted.year,
                                     specific_t_weighted.sel(lat=current_lat, lon=current_lon))
            trends[i][j] = reg_results.slope
            pvals[i][j] = reg_results.pvalue
            r2[i][j] = reg_results.rvalue ** 2
    new_dataset = xr.Dataset(
        data_vars={
            "trend": (["lat", "lon"], trends),
            "rSquared": (["lat", "lon"], r2),
            "pValue": (["lat", "lon"], pvals)
        },
        coords=dict(
            lat=(["lat"], specific_t_weighted[0].lat.values),
            lon=(["lon"], specific_t_weighted[0].lon.values)
        )
    )
    return new_dataset


def create_polyfit_trend_at_each_location(dataset, time_weighting, start_yr="2001",
                                          start_mon="01", end_yr="2022", end_mon="01", ly_feb=27.65, nly_feb=28.45):
    cleaned_dat = slice_dataset_with_year_and_month(dataset, start_yr, start_mon, end_yr, end_mon)
    specific_t_weighted = apply_time_averaging(cleaned_dat, averaging_method=time_weighting,
                                               running_length=0, start_mon=start_mon, feb_leap_year_correction=ly_feb,
                                               feb_non_leap_year_correction=nly_feb)

    coeffs = specific_t_weighted.polyfit('year', deg=1)
    trends = coeffs.sel(degree=1)
    return trends["polyfit_coefficients"]
