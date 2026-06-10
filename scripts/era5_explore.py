import xarray as xr

# Open the full summer dataset
ds = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")

print("=== Dataset Info ===")
print(ds)
print("\n=== Variables ===")
for var in ds.data_vars:
    print(f"  {var}: {ds[var].dims} {ds[var].shape}")

print("\n=== Coordinates ===")
for coord in ds.coords:
    print(f"  {coord}: {ds[coord].values[:5]}...")

# Convert temperature from Kelvin to Celsius
if "t2m" in ds:
    temp_c = ds["t2m"] - 273.15
    print(f"\n=== 2m Temperature Summary (Celsius) ===")
    print(f"  Min: {float(temp_c.min()):.1f}°C")
    print(f"  Max: {float(temp_c.max()):.1f}°C")
    print(f"  Mean: {float(temp_c.mean()):.1f}°C")
    print(f"  Time steps: {len(ds.valid_time)}")
    print(f"  From: {str(ds.valid_time.values[0])[:16]}")
    print(f"  To:   {str(ds.valid_time.values[-1])[:16]}")
