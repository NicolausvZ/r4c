import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr

# Load ERA5 - pick hottest afternoon in the dataset
ds = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
temp_c = ds["t2m"] - 273.15

# Find the hottest time step
max_idx = temp_c.mean(dim=["latitude", "longitude"]).argmax().item()
hottest = temp_c.isel(valid_time=max_idx)
hottest_time = str(ds.valid_time.values[max_idx])[:16]
print(f"Hottest time step: {hottest_time}")
print(f"Mean temp: {float(hottest.mean()):.1f}°C")

# Load WorldCover
CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare vegetation", "#b4b4b4"),
    80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
}

with rasterio.open("data/worldcover_helsinki.tif") as src:
    window = rasterio.windows.from_bounds(
        left=24.8, bottom=60.1, right=25.2, top=60.4,
        transform=src.transform
    )
    lc_data = src.read(1, window=window)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# WorldCover base
from matplotlib.colors import ListedColormap
cmap_colors = ["#ffffff"] * 256
for code, (name, color) in CLASSES.items():
    cmap_colors[code] = color
cmap = ListedColormap(cmap_colors)

ax.imshow(lc_data, cmap=cmap, vmin=0, vmax=255,
          extent=[24.8, 25.2, 60.1, 60.4], alpha=0.7)

# ERA5 temperature overlay
cf = ax.contourf(
    hottest.longitude, hottest.latitude, hottest.values,
    levels=10, cmap="RdYlBu_r", alpha=0.5
)
plt.colorbar(cf, ax=ax, label="2m Temperature (°C)")

# Contour lines
ax.contour(
    hottest.longitude, hottest.latitude, hottest.values,
    levels=10, colors="black", linewidths=0.5, alpha=0.3
)

# Pilot areas
locations = {
    "Laajasalo": (25.0849, 60.1756),
    "Koivukyla": (25.0500, 60.3333),
}
for name, (lon, lat) in locations.items():
    ax.plot(lon, lat, "k^", markersize=8)
    ax.annotate(name, (lon, lat), textcoords="offset points",
                xytext=(5, 5), fontweight="bold")

ax.set_title(f"ERA5 Temperature over Land Cover\n{hottest_time}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Legend
patches = [mpatches.Patch(color=CLASSES[c][1], label=CLASSES[c][0])
           for c in CLASSES]
ax.legend(handles=patches, loc="lower right", fontsize=7)

plt.tight_layout()
plt.savefig("data/era5_worldcover_overlay.png", dpi=150)
plt.show()
print("Saved!")