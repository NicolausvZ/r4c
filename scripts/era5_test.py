import cdsapi
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CDS_API_KEY")

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key=api_key
)

# Full summer 2023 - June, July, August
client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_solar_radiation_downwards",
            "total_precipitation"
        ],
        "year": "2023",
        "month": ["06", "07", "08"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [61, 24, 60, 26],
        "format": "netcdf4",
    },
    "data/era5_helsinki_summer2023.nc"
)

print("ERA5 summer 2023 download complete!")
