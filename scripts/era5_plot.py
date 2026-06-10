import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")

temp_c = ds["t2m"] - 273.15

# Pick the Helsinki centre grid point
helsinki = temp_c.sel(latitude=60.25, longitude=24.75, method="nearest")

plt.figure(figsize=(14, 5))
plt.plot(ds.valid_time, helsinki.values, linewidth=0.8, color="tomato")
plt.title("ERA5 2m Temperature - Helsinki Summer 2023")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/era5_summer_timeseries.png", dpi=150)
plt.show()
print("Done!")