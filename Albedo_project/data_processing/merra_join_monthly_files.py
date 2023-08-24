import os
import xarray as xr
from cloudpathlib import AnyPath

merra_raw_location = AnyPath("/Users/mawa7160/dev/data/MERRA2/23Aug/Raw")
merra_output_location = AnyPath("/Users/mawa7160/dev/data/MERRA2/23Aug")

merra_data = xr.Dataset()
for filename in os.listdir(merra_raw_location):
    file_path = merra_raw_location / filename
    if os.path.isfile(file_path):
        next_file = xr.open_dataset(file_path)
        merra_data = merra_data.merge(next_file)

merra_data.to_netcdf(merra_output_location / "Merra2_monthly_01_00-06_23.nc")