import requests
import json
import time

# Laajasalo, Helsinki
coords = [
    ("Laajasalo", 60.1756, 25.0849),
    ("Koivukyla", 60.3333, 25.0500),
    ("Helsinki centre", 60.1699, 24.9384),
]

properties = ["phh2o", "soc", "clay", "sand"]

print("Fetching SoilGrids data for Helsinki pilot areas\n")

for place, lat, lon in coords:
    print(f"--- {place} (lat={lat}, lon={lon}) ---")
    for prop in properties:
        url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}&property={prop}&depth=0-5cm&value=mean"
        )
        try:
            r = requests.get(url, timeout=20)
            data = r.json()
            value = data["properties"]["layers"][0]["depths"][0]["values"]["mean"]
            print(f"  {prop}: {value}")
        except requests.exceptions.Timeout:
            print(f"  {prop}: timeout - API slow")
        except Exception as e:
            print(f"  {prop}: failed ({e})")
        time.sleep(1)  # be polite to the API
    print()