import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Open the downloaded ERA5 file
ds = xr.open_dataset("data/era5_helsinki_test.nc")

# Convert to Celsius
temp_c = ds["t2m"].squeeze() - 273.15

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

im = ax.contourf(
    temp_c.longitude,
    temp_c.latitude,
    temp_c.values,
    levels=20,
    cmap="RdYlBu_r"
)

plt.colorbar(im, ax=ax, label="2m Temperature (°C)")

# Mark Helsinki and pilot areas
locations = {
    "Helsinki": (24.9384, 60.1699),
    "Laajasalo": (25.0849, 60.1756),
    "Koivukyla": (25.0500, 60.3333),
}

for name, (lon, lat) in locations.items():
    ax.plot(lon, lat, "ko", markersize=6)
    ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(5, 5))

ax.set_title("ERA5 2m Temperature - Helsinki Area\n15 July 2023 12:00 UTC")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("data/era5_helsinki_temp.png", dpi=150)
plt.show()
print("Plot saved to data/era5_helsinki_temp.png")
