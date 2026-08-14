"""Shared helpers for the R4C sensor pipeline."""
import json
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Sensor 24E124136E140283 has an empty "name" field in the FVH GeoJSON — it was
# logged as a "Potential sensor spot" and never given a display name, though it
# reports data normally. Its field_4 describes it as "In middle of vegetation /
# green areas Sahamäenkaari", so we name it after that street. Without this it
# joins as NaN and silently drops out of any merge on sensor name.
# ---------------------------------------------------------------------------
NAME_OVERRIDES = {
    "24E124136E140283": "Sahamaenkaari green",
}


def fix_mojibake(s):
    """FVH exports are UTF-8 read as latin-1 in places: 'Koivukylä' -> 'KoivukylÃ¤'."""
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s:
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def normalise(s):
    """Casefold + strip diacritics, for robust name matching across sources."""
    if not isinstance(s, str):
        return ""
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip().replace("_", " ")


def sensor_name_from_column(col):
    """Extract the sensor name from a column like 'temperature_Koivukyla_Kuusitie 11'."""
    return fix_mojibake(str(col).split("_")[-1])


def load_sensor_metadata(path="data/r4c_fvh_all_latest.geojson"):
    """Sensor locations + install dates, with names guaranteed non-empty."""
    with open(path, encoding="utf-8") as f:
        geo = json.load(f)

    rows = []
    for ft in geo["features"]:
        p = ft["properties"]
        fid = ft.get("id", "")
        name = fix_mojibake(p.get("name", "") or "").strip()

        if not name:
            name = NAME_OVERRIDES.get(fid) or f"sensor_{p.get('Sensor_number', fid)}"
            named_by = "override"
        else:
            named_by = "geojson"

        rows.append({
            "name": name,
            "sensor_id": fid,
            "district": fix_mojibake(p.get("district", "")),
            "lon": ft["geometry"]["coordinates"][0],
            "lat": ft["geometry"]["coordinates"][1],
            "installed": p.get("Date_installed"),
            "name_source": named_by,
        })

    df = pd.DataFrame(rows)
    df["installed"] = pd.to_datetime(df["installed"])
    df["match_key"] = df["name"].map(normalise)
    return df


def match_sensor(name, meta_df):
    """Find the metadata row for a sensor column name. Returns a Series or None."""
    key = normalise(name)
    if not key:
        return None

    exact = meta_df[meta_df["match_key"] == key]
    if len(exact):
        return exact.iloc[0]

    # the unnamed probe appears in the Excel columns under its hardware id
    by_id = meta_df[meta_df["sensor_id"].str.lower() == name.lower()]
    if len(by_id):
        return by_id.iloc[0]

    for _, row in meta_df.iterrows():
        rk = row["match_key"]
        if rk and (key in rk or rk in key):
            return row
    return None