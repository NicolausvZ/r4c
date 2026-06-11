import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# WorldCover land class definitions
CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}

print("Opening WorldCover data...")
with rasterio.open("data/worldcover_helsinki.tif") as src:
    # Read a subset around Helsinki centre
    data = src.read(1)
    transform = src.transform
    print(f"Full tile shape: {data.shape}")
    print(f"Resolution: {src.res}")
    print(f"CRS: {src.crs}")

    # Crop to Helsinki area
    window = rasterio.windows.from_bounds(
        left=24.8, bottom=60.1, right=25.2, top=60.4,
        transform=src.transform
    )
    data_crop = src.read(1, window=window)
    win_transform = src.window_transform(window)

print(f"Cropped shape: {data_crop.shape}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Build colormap
unique_classes = np.unique(data_crop)
cmap_colors = [CLASSES.get(c, ("Unknown", "#ffffff"))[1] for c in range(256)]

from matplotlib.colors import ListedColormap
cmap = ListedColormap(cmap_colors)

im = ax.imshow(data_crop, cmap=cmap, vmin=0, vmax=255,
               extent=[24.8, 25.2, 60.1, 60.4])

# Legend
patches = [mpatches.Patch(color=CLASSES[c][1], label=CLASSES[c][0])
           for c in unique_classes if c in CLASSES]
ax.legend(handles=patches, loc="lower right", fontsize=8)

ax.set_title("ESA WorldCover 2021 - Helsinki Area (10m)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("data/worldcover_helsinki.png", dpi=150)
plt.show()
print("Saved to data/worldcover_helsinki.png")