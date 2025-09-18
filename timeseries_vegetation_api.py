# timeseries_vegetation_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# helpers from utils.py
from utils import (
    prefer_http_from_asset,
    sign_href_if_pc,
    aoi_to_scene,
    read_band_window,
    compute_index_array_by_name,
    search_planetary,
    search_aws,
    THREADS,
)

import rasterio
from rasterio.enums import Resampling

router = APIRouter()

# ✅ Only keep the required vegetation indices
SUPPORTED = ["NDVI", "EVI", "SAVI", "SUCROSE"]
MAX_THREADS = min(4, THREADS or 4)

class TSRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    index: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    max_items: Optional[int] = 8

class TimePoint(BaseModel):
    date: str
    value: float
    status: str

class Summary(BaseModel):
    min: Optional[float]
    mean: Optional[float]
    max: Optional[float]

class TSResponse(BaseModel):
    index: str
    summary: Summary
    timeseries: List[TimePoint]

def classify_vegetation(index: str, v: Optional[float]) -> str:
    if v is None:
        return "No Data"
    if index in ("NDVI", "EVI", "SAVI"):
        if v < 0.2: return "Very Poor"
        if v < 0.4: return "Moderate"
        if v < 0.6: return "Good"
        return "Very Good"
    if index == "SUCROSE":
        if v < 0.2: return "Immature"
        if v < 0.4: return "Early Maturity"
        if v < 0.6: return "Optimal"
        return "Overmature"
    return "Unknown"

def _signed_asset_map(item):
    assets = getattr(item, "assets", {}) or {}
    signed = {}
    for k, a in assets.items():
        try:
            url = prefer_http_from_asset(a)
            signed[k.lower()] = sign_href_if_pc(url) if url else None
        except Exception:
            signed[k.lower()] = None
    return signed

def _item_has_required_assets(item, required_bands):
    assets_keys = set(k.lower() for k in (item.assets or {}).keys())
    return required_bands.issubset(assets_keys)

def _read_bands_from_signed(signed_assets, needed, geom, out_h=16, out_w=16):
    band_arrays = {}
    for b in needed:
        url = signed_assets.get(b.lower()) or signed_assets.get(b)
        if not url:
            band_arrays[b] = None
            continue
        try:
            with rasterio.Env():
                with rasterio.open(url) as ds:
                    arr = read_band_window(
                        ds,
                        aoi_to_scene(geom, ds.crs.to_string()),
                        out_h,
                        out_w,
                        ds.transform,
                        Resampling.bilinear
                    )
                    if np.nanmax(arr) > 1.5:
                        arr = arr * (1/10000.0)
                    band_arrays[b] = arr
        except Exception:
            band_arrays[b] = None
    return band_arrays

def _compute_index_for_item(item, geom, idx, out_h=16, out_w=16):
    try:
        cloud = item.properties.get("eo:cloud_cover") or item.properties.get("cloud_cover")
        if cloud is not None:
            try:
                if float(cloud) > 60.0:
                    return None
            except Exception:
                pass

        # ✅ Only needed bands for the 4 indices
        if idx == "NDVI":
            needed = ["B08","B04"]
        elif idx == "EVI":
            needed = ["B08","B04","B02"]
        elif idx == "SAVI":
            needed = ["B08","B04"]
        elif idx == "SUCROSE":
            needed = ["B11","B04"]
        else:
            return None

        reqset = set(b.lower() for b in needed)
        if not _item_has_required_assets(item, reqset):
            return None

        signed = _signed_asset_map(item)
        bands = _read_bands_from_signed(signed, needed, geom, out_h=out_h, out_w=out_w)
        if any(bands.get(b) is None for b in needed):
            return None

        band_dict = {k: bands.get(k) for k in bands}

        try:
            arr = compute_index_array_by_name(idx, band_dict)
        except Exception:
            eps = 1e-6
            if idx == "NDVI":
                arr = (band_dict["B08"] - band_dict["B04"]) / (band_dict["B08"] + band_dict["B04"] + eps)
            elif idx == "EVI":
                NIR, RED, BLUE = band_dict["B08"], band_dict["B04"], band_dict["B02"]
                arr = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1.0 + eps)
            elif idx == "SAVI":
                NIR, RED = band_dict["B08"], band_dict["B04"]
                L = 0.5
                arr = ((NIR - RED) / (NIR + RED + L + eps)) * (1.0 + L)
            elif idx == "SUCROSE":
                arr = (band_dict["B11"] - band_dict["B04"]) / (band_dict["B11"] + band_dict["B04"] + eps)
            else:
                return None

        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        mean_val = round(float(np.nanmean(arr[mask])), 3)
        date = str(item.properties.get("datetime") or item.properties.get("acquired") or "")[:10]
        return (date, mean_val)
    except Exception:
        return None

@router.post("/vegetation", response_model=TSResponse)
def vegetation_timeseries(req: TSRequest):
    idx = req.index.upper()
    if idx not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported index. Supported: {SUPPORTED}")

    satellite = (req.satellite or "s2").lower()
    if satellite.startswith("s1"):
        raise HTTPException(
            status_code=400,
            detail=f"Index {idx} requires Sentinel-2 (optical). Sentinel-1 is radar-only."
        )

    prefer_pc = True
    if req.provider and req.provider.lower() == "aws":
        prefer_pc = False

    try:
        collections = ["sentinel-2-l2a"]
        dt = f"{req.start_date}/{req.end_date}"
        search_limit = min(max(64, (req.max_items or 8) * 8), 500)

        items = []
        if prefer_pc:
            items = search_planetary(collections, req.geometry, dt, limit=search_limit)
            if not items:
                items = search_aws(collections, req.geometry, dt, limit=search_limit)
        else:
            items = search_aws(collections, req.geometry, dt, limit=search_limit)
            if not items:
                items = search_planetary(collections, req.geometry, dt, limit=search_limit)

        if not items:
            return {"index": idx, "summary": {"min": None, "mean": None, "max": None}, "timeseries": []}

        results = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
            futures = {ex.submit(_compute_index_for_item, it, req.geometry, idx, 16, 16): it for it in items}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    results.append(res)

        if not results:
            return {"index": idx, "summary": {"min": None, "mean": None, "max": None}, "timeseries": []}

        date_map: Dict[str, List[float]] = {}
        for date_str, val in results:
            if not date_str:
                continue
            date_map.setdefault(date_str, []).append(val)

        aggregated = [(d, round(float(sum(vals) / len(vals)), 3)) for d, vals in date_map.items()]
        aggregated_sorted = sorted(aggregated, key=lambda x: x[0])

        max_pts = req.max_items or len(aggregated_sorted)
        if max_pts < 1:
            max_pts = len(aggregated_sorted)
        if len(aggregated_sorted) > max_pts:
            n = max_pts
            idxs = [int(round(i * (len(aggregated_sorted) - 1) / float(max(n - 1, 1)))) for i in range(n)]
            aggregated_sorted = [aggregated_sorted[i] for i in idxs]

        times = []
        vals = []
        for d, v in aggregated_sorted:
            times.append({"date": d, "value": v, "status": classify_vegetation(idx, v)})
            vals.append(v)

        summary = {
            "min": round(min(vals), 3),
            "mean": round(float(sum(vals) / len(vals)), 3),
            "max": round(max(vals), 3)
        }
        return {"index": idx, "summary": summary, "timeseries": times}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
