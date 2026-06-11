import requests
import os

# ESA WorldCover 2021 - Helsinki area tile
# Tiles are 3x3 degree, Helsinki is in tile N60E024
url = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N60E024_Map.tif"

output_path = "data/worldcover_helsinki.tif"

print("Downloading ESA WorldCover for Helsinki area...")

response = requests.get(url, stream=True, timeout=60)

if response.status_code == 200:
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f} MB)", end="")
    print(f"\nDone! Saved to {output_path}")
else:
    print(f"Failed: HTTP {response.status_code}")