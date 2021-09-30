from data_manipulation_functions import*

def create_weighted_mean_time_series_from_CERES_ERBF_data(full_data, variable_name, hemisphere="none", deseason=True):
    # Use "nh" or "sh" for the hemisphere or 'none' for a global mean
    # variable names include: 'toa_sw_all_mon', 'toa_lw_all_mon', 'toa_sw_clr_c_mon', 'toa_lw_clr_c_mon',
    # 
    # Create/Load the latitude weighting
    lat_weights = create_lat_weights()
    weights = xr.DataArray(data=lat_weights, coords=[full_data.lat], dims=["lat"])
    weights.name = "weights"    

    if hemisphere.lower() == "nh":
        full_data = full_data.sel(lat=slice(0, 90))
        weights = weights.sel(lat=slice(0, 90))
    elif hemisphere.lower() == "sh":
        full_data = full_data.sel(lat=slice(-90, 0))
        weights = weights.sel(lat=slice(-90, 0))
    elif hemisphere.lower() != "none":
        print("Unrecognized argument for hemisphere. Please try again.")
        return -1
    
    mean_data = create_lat_and_lon_weighted_mean(full_data, variable_name, weights)
    clean_mean_data = remove_incomplete_years(mean_data)

    if(deseason):
        clean_mean_data = calculate_deseasonalized_data(clean_mean_data)

    return clean_mean_data