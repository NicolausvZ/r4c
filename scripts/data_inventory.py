import requests

datasets = {
    "ERA5 (ECMWF)": "https://cds.climate.copernicus.eu",
    "ESA WorldCover": "https://esa-worldcover.org",
    "SoilGrids": "https://rest.isric.org/soilgrids/v2.0/properties/query?lon=25&lat=60&property=phh2o&depth=0-5cm&value=mean",
    "HydroSHEDS": "https://www.hydrosheds.org",
    "OpenStreetMap Overpass": "https://overpass-api.de/api/interpreter",
    "Copernicus Land": "https://land.copernicus.eu",
    "MODIS LST (NASA)": "https://modis.gsfc.nasa.gov",
}

print("Checking dataset availability...\n")
for name, url in datasets.items():
    try:
        r = requests.get(url, timeout=8)
        status = "✅ reachable" if r.status_code < 400 else f"⚠️ status {r.status_code}"
    except Exception as e:
        status = f"❌ unreachable ({e})"
    print(f"{name}: {status}")
    