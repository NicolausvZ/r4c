import xarray as xr

# Open the downloaded ERA5 file
ds = xr.open_dataset("data/era5_helsinki_test.nc")

print("=== Dataset Info ===")
print(ds)
print("\n=== Variables ===")
for var in ds.data_vars:
    print(f"  {var}: {ds[var].dims} {ds[var].shape}")

print("\n=== Coordinates ===")
for coord in ds.coords:
    print(f"  {coord}: {ds[coord].values}")

# Convert temperature from Kelvin to Celsius
if "t2m" in ds:
    temp_c = ds["t2m"] - 273.15
    print(f"\n=== 2m Temperature (Celsius) ===")
    print(f"  Min: {float(temp_c.min()):.1f}°C")
    print(f"  Max: {float(temp_c.max()):.1f}°C")
    print(f"  Mean: {float(temp_c.mean()):.1f}°C")
    