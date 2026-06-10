import cdsapi
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("CDS_API_KEY")

# Set up client
client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key=api_key
)

# Download a small ERA5 sample - 2m temperature over Helsinki
# Just one day, one variable to test
client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "2023",
        "month": "07",
        "day": "15",
        "time": "12:00",
        "area": [61, 24, 60, 26],  # North, West, South, East - Helsinki bbox
        "format": "netcdf",
    },
    "data/era5_helsinki_test.nc"
)

print("ERA5 download complete!")
