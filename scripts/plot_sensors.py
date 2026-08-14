import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import numpy as np
from rasterio.windows import from_bounds

# Load sensor locations
with open("data/r4c_fvh_all_latest.geojson") as f:
    data = json.load(f)

# Parse sensors
sensors = []
for feature in data["features"]:
    p = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    m = p.get("measurement", {})
    sensors.append({
        "name": p.get("name", ""),
        "district": p.get("district", ""),
        "lon": coords[0],
        "lat": coords[1],
        "temp": m.get("temperature"),
        "humidity": m.get("humidity"),
    })

print(f"Loaded {len(sensors)} sensors")

# Load WorldCover as background
with rasterio.open("data/worldcover_helsinki.tif") as src:
    window = from_bounds(24.9, 60.14, 25.2, 60.38, src.transform)
    lc = src.read(1, window=window)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Background land cover
from matplotlib.colors import ListedColormap
CLASSES = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c",
    40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4",
    80: "#0064c8", 90: "#0096a0"
}
cmap_colors = ["#ffffff"] * 256
for code, color in CLASSES.items():
    cmap_colors[code] = color
cmap = ListedColormap(cmap_colors)

ax.imshow(lc, cmap=cmap, vmin=0, vmax=255,
          extent=[24.9, 25.2, 60.14, 60.38], alpha=0.6)

# Plot sensors by district
colors = {"Koivukylä": "blue", "Laajasalo": "red"}
for s in sensors:
    color = colors.get(s["district"], "black")
    ax.plot(s["lon"], s["lat"], "o", color=color,
            markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    if s["name"]:
        ax.annotate(s["name"], (s["lon"], s["lat"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color="black")

# Legend
patches = [mpatches.Patch(color=c, label=d) for d, c in colors.items()]
ax.legend(handles=patches, loc="upper left", fontsize=10)

ax.set_title("FVH Sensor Network - Helsinki Pilot Areas\n20 stations, June 2024")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("data/sensor_map.png", dpi=150)
plt.show()
print("Saved to data/sensor_map.png")