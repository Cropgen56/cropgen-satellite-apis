# shared helpers and constants
import os
import io
import base64
import math
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image

# raster/stac/geometry libraries
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.windows import transform as win_transform, bounds as win_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import geometry_mask
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from pystac_client import Client
import planetary_computer

# ---------- Environment / tuning ----------
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "FALSE")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.JP2,.TIF,.TIFF,.JP2")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_CACHEMAX", "512")

EARTH_SEARCH_AWS = "https://earth-search.aws.element84.com/v1"
PLANETARY_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

THREADS = min(4, (os.cpu_count() or 4))

# ---------- Palettes & labels ----------
PALETTE = ['#000000','#a50026','#d73027','#f46d43','#fdae61',
           '#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63',
           '#1a9850','#006837']
LABELS = ["No Vegetation","Very Poor","Poor","Fair","Moderate","Moderate-Good",
          "Good","Very Good","Excellent","Dense Vegetation","Very Dense","Extremely Dense"]
EDGES = np.array([-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01], dtype="float32")

PALETTE_MAP = {
    "NDVI": (PALETTE, LABELS),
    "SAVI": (PALETTE, LABELS),
    "EVI": (PALETTE, LABELS),
    "EVI2": (PALETTE, LABELS),
    "NDMI": (PALETTE, LABELS),
    "NDWI": (PALETTE, LABELS),
    "SMI": (PALETTE, LABELS),
    "CCC": (PALETTE, LABELS),
    "NITROGEN": (PALETTE, LABELS),
    "SOC": (PALETTE, LABELS),
    "NDRE": (PALETTE, LABELS),
    "RECI": (PALETTE, LABELS),
    "TRUE_COLOR": (["#000000"], ["True Color"]),
}

# ---------- STAC helpers ----------
def sign_href_if_pc(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    try:
        return planetary_computer.sign(href)
    except Exception:
        return href

def s3_to_https(href: str) -> str:
    if href and href.startswith("s3://"):
        parts = href[5:].split("/", 1)
        bucket = parts[0]; key = parts[1] if len(parts) > 1 else ""
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href

def prefer_http_from_asset(asset) -> Optional[str]:
    if asset is None:
        return None
    href = getattr(asset, "href", "") or ""
    alt = getattr(asset, "extra_fields", {}).get("alternate", {}) if hasattr(asset, "extra_fields") else {}
    for k in ("https", "http", "self"):
        v = alt.get(k)
        if isinstance(v, dict):
            url = v.get("href")
            if url and url.startswith("http"):
                return url
        elif isinstance(v, str) and v.startswith("http"):
            return v
    if href.startswith("http"):
        return href
    return s3_to_https(href) if href else None

def compute_index_array_by_name(index_name: str, bands: Dict[str, np.ndarray]) -> np.ndarray:
    for k in list(bands.keys()):
        if bands[k] is not None:
            bands[k] = bands[k].astype("float32")
    name = index_name.upper()
    eps = 1e-6
    def missing(*args):
        return any(arg is None for arg in args)

    if name == "NDVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("NDVI requires B08 and B04")
        return (NIR - RED) / (NIR + RED + eps)
    if name == "EVI":
        NIR, RED, BLUE = bands.get("B08"), bands.get("B04"), bands.get("B02")
        if missing(NIR, RED, BLUE): raise ValueError("EVI requires B08,B04,B02")
        return 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1.0 + eps)
    if name == "EVI2":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("EVI2 requires B08,B04")
        return 2.5 * (NIR - RED) / (NIR + 2.4*RED + 1.0 + eps)
    if name == "SAVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("SAVI requires B08,B04")
        L = 0.5
        return ((NIR - RED) / (NIR + RED + L + eps)) * (1.0 + L)
    if name == "MSAVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("MSAVI requires B08,B04")
        a = 2*NIR + 1.0
        inside = np.maximum((a*a) - 8*(NIR - RED), 0.0)
        return 0.5 * (a - np.sqrt(inside))
    if name == "NDMI":
        NIR, SWIR = bands.get("B08"), bands.get("B11")
        if missing(NIR, SWIR): raise ValueError("NDMI requires B08,B11")
        return (NIR - SWIR) / (NIR + SWIR + eps)
    if name == "NDWI":
        GREEN, SWIR = bands.get("B03"), bands.get("B11")
        if missing(GREEN, SWIR): raise ValueError("NDWI requires B03,B11")
        return (GREEN - SWIR) / (GREEN + SWIR + eps)
    if name == "SMI":
        NIR, SWIR = bands.get("B08"), bands.get("B11")
        if missing(NIR, SWIR): raise ValueError("SMI requires B08,B11")
        return (NIR - SWIR) / (NIR + SWIR + eps)
    if name == "CCC":
        B3, B4, B5 = bands.get("B03"), bands.get("B04"), bands.get("B05")
        if missing(B3, B4, B5): raise ValueError("CCC requires B03,B04,B05")
        return (B5 * B5) / (B4 * B3 + eps)
    if name == "NITROGEN":
        B4, B5 = bands.get("B04"), bands.get("B05")
        if missing(B4, B5): raise ValueError("NITROGEN requires B04,B05")
        return (B5 - B4) / (B5 + B4 + eps)
    if name == "SOC":
        B3, B4, B11, B12 = bands.get("B03"), bands.get("B04"), bands.get("B11"), bands.get("B12")
        if missing(B3, B4, B11, B12): raise ValueError("SOC requires B03,B04,B11,B12")
        return (B3 + B4) / (B11 + B12 + eps)
    if name == "NDRE":
        B8A, B5 = bands.get("B8A"), bands.get("B05")
        if missing(B8A, B5): raise ValueError("NDRE requires B8A,B05")
        return (B8A - B5) / (B8A + B5 + eps)
    if name == "RECI":
        B8, B5 = bands.get("B08"), bands.get("B05")
        if missing(B8, B5): raise ValueError("RECI requires B08,B05")
        return (B8 - B5) / (B5 + eps)
    if name == "TRUE_COLOR":
        raise ValueError("TRUE_COLOR is a special rendering path (RGB)")

    raise ValueError(f"Unsupported index: {index_name}")

# ---------- Grid / read helpers ----------
def aoi_to_scene(aoi_ll_geojson, crs_str):
    t = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), shape(aoi_ll_geojson))

def build_adaptive_grid(crs, aoi_ll_geojson, native_res_m=10.0,
                        MIN_PX_LONG=600, MAX_PX_LONG=1200, MIN_RES_M=0.5, MAX_RES_M=40.0):
    aoi_sc = aoi_to_scene(aoi_ll_geojson, crs.to_string())
    minx, miny, maxx, maxy = aoi_sc.bounds
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    long = max(dx, dy)
    res_for_min = long / MAX_PX_LONG
    res_for_max = long / MIN_PX_LONG
    res_m = min(max(res_for_min, native_res_m / 2.0), res_for_max)
    res_m = float(np.clip(res_m, MIN_RES_M, MAX_RES_M))
    width = max(1, int(math.ceil(dx / res_m)))
    height = max(1, int(math.ceil(dy / res_m)))
    transform = Affine.translation(minx, maxy) * Affine.scale(res_m, -res_m)
    return aoi_sc, transform, height, width, res_m

def read_band_window(src, geom_sc, out_h, out_w, transform, resampling):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.full((out_h, out_w), np.nan, dtype="float32")
    try:
        arr = src.read(1, window=win, out_shape=(out_h, out_w), resampling=resampling, masked=True).filled(0).astype("float32")
        return arr
    except Exception:
        arr = src.read(1, window=win, masked=True).filled(0).astype("float32")
    src_tr = win_transform(win, src.transform)
    dst = np.full((out_h, out_w), np.nan, dtype="float32")
    reproject(arr, dst,
              src_transform=src_tr, src_crs=src.crs,
              dst_transform=transform, dst_crs=src.crs,
              src_nodata=0.0, dst_nodata=np.nan,
              resampling=resampling)
    return dst

def read_scl_window(src, geom_sc, out_h, out_w, transform):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.zeros((out_h, out_w), dtype="int16")
    try:
        arr = src.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.nearest, masked=True).filled(0).astype("int16")
        return arr
    except Exception:
        arr = src.read(1, window=win, masked=True).filled(0).astype("int16")
    src_tr = win_transform(win, src.transform)
    dst = np.zeros((out_h, out_w), dtype="int16")
    reproject(arr, dst,
              src_transform=src_tr, src_crs=src.crs,
              dst_transform=transform, dst_crs=src.crs,
              src_nodata=0, dst_nodata=0,
              resampling=Resampling.nearest)
    return dst

# ---------- STAC helpers ----------
def search_planetary(collections, intersects, dt, limit=50):
    cat = Client.open(PLANETARY_STAC)
    return list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())

def search_aws(collections, intersects, dt, limit=50):
    cat = Client.open(EARTH_SEARCH_AWS)
    return list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())

def items_for_date(aoi_geojson, iso_dt, collection, prefer_pc=True, limit=6):
    day = iso_dt[:10]
    dt = f"{day}/{day}"
    if prefer_pc:
        items = search_planetary([collection], aoi_geojson, dt, limit=limit)
        if items:
            return items
        return search_aws([collection], aoi_geojson, dt, limit=limit)
    else:
        items = search_aws([collection], aoi_geojson, dt, limit=limit)
        if items:
            return items
        return search_planetary([collection], aoi_geojson, dt, limit=limit)

# quick keep pct
def quick_keep_pct(item, aoi_geojson):
    assets = item.assets
    red = prefer_http_from_asset(assets.get("red") or assets.get("B04"))
    nir = prefer_http_from_asset(assets.get("nir") or assets.get("B08"))
    scl = assets.get("scl") or assets.get("SCL")
    scl_url = prefer_http_from_asset(scl) if scl else None
    if not (red and nir):
        return 0.0, False
    try:
        with rasterio.Env():
            with rasterio.open(sign_href_if_pc(red)) as rsrc, rasterio.open(sign_href_if_pc(nir)) as nsrc, \
                 (rasterio.open(sign_href_if_pc(scl_url)) if scl_url else nullcontext()) as sds:
                crs = rsrc.crs or nsrc.crs
                aoi_sc = aoi_to_scene(aoi_geojson, crs.to_string())
                win = from_bounds(*aoi_sc.bounds, rsrc.transform).round_offsets().round_lengths()
                if win.width <= 0 or win.height <= 0:
                    return 0.0, bool(sds)
                th = max(1, min(64, int(win.height))); tw = max(1, min(64, int(win.width)))
                sub = Window(win.col_off, win.row_off, win.width, win.height)
                R = rsrc.read(1, window=sub, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("float32")
                N = nsrc.read(1, window=sub, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("float32")
                if R.max() > 1.5 or N.max() > 1.5:
                    R *= 1/10000.0; N *= 1/10000.0
                den = (N + R); den[den == 0] = np.nan
                nd = (N - R) / den
                tr = win_transform(sub, rsrc.transform) * Affine.scale(win.width/float(tw), win.height/float(th))
                mask = geometry_mask([mapping(aoi_sc)], out_shape=(th, tw), transform=tr, invert=True)
                if sds:
                    tb = win_bounds(sub, rsrc.transform)
                    sw = from_bounds(*tb, transform=sds.transform).round_offsets().round_lengths()
                    sw = sw.intersection(Window(0, 0, sds.width, sds.height)).round_offsets().round_lengths()
                    if sw.width > 0 and sw.height > 0:
                        S = sds.read(1, window=sw, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("int16")
                        classes = [8,9,10,11]
                        nd[np.isin(S, classes)] = np.nan
                kept = np.count_nonzero(np.isfinite(nd) & mask)
                total = np.count_nonzero(mask)
                return (kept / max(total, 1)) * 100.0, bool(sds)
    except Exception:
        return 0.0, False

# pick best item
def pick_best_item(aoi_geojson, start, end, prefer_pc=True, satellite="s2"):
    if satellite and satellite.lower().startswith("s1"):
        collections_try = ["sentinel-1"]
    else:
        collections_try = ["sentinel-2-l2a", "sentinel-2-l1c"]
    items = []
    if prefer_pc:
        items = search_planetary(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
        if not items:
            items = search_aws(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
    else:
        items = search_aws(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
        if not items:
            items = search_planetary(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
    if not items:
        fmt = "%Y-%m-%d"
        s = datetime.strptime(start, fmt) - timedelta(days=14)
        e = datetime.strptime(end, fmt) + timedelta(days=14)
        items = search_planetary(collections_try, aoi_geojson, f"{s.strftime(fmt)}/{e.strftime(fmt)}", limit=24)
        if not items:
            items = search_aws(collections_try, aoi_geojson, f"{s.strftime(fmt)}/{e.strftime(fmt)}", limit=24)
    if not items:
        return None, False, None

    scored = []
    with ThreadPoolExecutor(max_workers=min(6, len(items))) as ex:
        futs = {ex.submit(quick_keep_pct, it, aoi_geojson): it for it in items}
        for f in as_completed(futs):
            try:
                pct, has_scl = f.result()
            except Exception:
                pct, has_scl = 0.0, False
            scored.append((pct, futs[f], has_scl))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    def get_collection_id(it):
        try:
            col = getattr(it, "collection", None)
            if isinstance(col, str):
                return col
            if hasattr(col, "id"):
                return col.id
        except Exception:
            pass
        try:
            props = getattr(it, "properties", {}) or {}
            for k in ("collection", "collection_id"):
                if k in props:
                    return props[k]
        except Exception:
            pass
        return None
    return best[1], best[2], get_collection_id(best[1])

# read tile
def _read_tile_into_stack(item, aoi_geojson, dst_transform, H, W, want_scl):
    out = {"used": False, "bands": {}, "S": None, "id": getattr(item, "id", None)}
    assets = item.assets or {}
    # pick best asset href heuristically
    def first_asset_href(assets_dict):
        # try some known names then fall back to first valid href
        for k in ("red","B04","B04.jp2","B04.tif","B04.TIF"):
            a = assets_dict.get(k)
            if a:
                h = prefer_http_from_asset(a)
                if h:
                    return h
        # fallback any asset with href
        for a in assets_dict.values():
            h = prefer_http_from_asset(a)
            if h:
                return h
        return None

    red_url = prefer_http_from_asset(assets.get("red") or assets.get("B04") or assets.get("RED") or first_asset_href(assets))
    nir_url = prefer_http_from_asset(assets.get("nir") or assets.get("B08") or assets.get("NIR") or first_asset_href(assets))
    if not (red_url and nir_url):
        return out
    red_url = sign_href_if_pc(red_url); nir_url = sign_href_if_pc(nir_url)
    scl_ref = assets.get("scl") or assets.get("SCL")
    scl_url = prefer_http_from_asset(scl_ref) if (want_scl and scl_ref) else None
    scl_url = sign_href_if_pc(scl_url) if scl_url else None
    try:
        with rasterio.open(red_url) as red, rasterio.open(nir_url) as nir, \
             (rasterio.open(scl_url) if scl_url else nullcontext()) as scl:
            crs = red.crs or nir.crs
            aoi_sc = aoi_to_scene(aoi_geojson, crs.to_string())
            R = read_band_window(red, aoi_sc, H, W, dst_transform, Resampling.bilinear)
            N = read_band_window(nir, aoi_sc, H, W, dst_transform, Resampling.bilinear)
            if np.nanmax(R) > 1.5 or np.nanmax(N) > 1.5:
                R *= 1/10000.0; N *= 1/10000.0
            S = None
            if scl:
                S = read_scl_window(scl, aoi_sc, H, W, dst_transform)
            out.update({"used": True, "bands": {"B04": R, "B08": N}, "S": S})
            for bkey in ("B03","B02","B11","B12","B05","B8A","B04","B08","B05"):
                a = assets.get(bkey) or assets.get(bkey.lower())
                if a:
                    url = prefer_http_from_asset(a)
                    if url:
                        url = sign_href_if_pc(url)
                        try:
                            with rasterio.open(url) as ds:
                                arr = read_band_window(ds, aoi_sc, H, W, dst_transform, Resampling.bilinear)
                                if np.nanmax(arr) > 1.5:
                                    arr *= 1/10000.0
                                out["bands"][bkey] = arr
                        except Exception:
                            pass
            return out
    except Exception:
        return out

# rendering helpers
def ndvi_to_bins(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, -1, 1)
    bins = np.digitize(arr, EDGES).astype("uint8")
    return np.where(np.isfinite(arr), bins, 0)

def compute_bounds_wgs84(transform: Affine, width: int, height: int, crs) -> List[float]:
    tl = transform * (0, 0)
    tr = transform * (width, 0)
    br = transform * (width, height)
    bl = transform * (0, height)
    xs = [tl[0], tr[0], br[0], bl[0]]
    ys = [tl[1], tr[1], br[1], bl[1]]
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    return [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]

def hex_to_rgba_tuple(hexcolor: str) -> Tuple[int,int,int,int]:
    h = hexcolor.lstrip('#')
    if len(h) == 6:
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        return (r,g,b,255)
    return (0,0,0,255)

def render_spread_png_fast(bins_canvas: np.ndarray, NDVI_canvas: np.ndarray, res_m: float,
                           supersample: int, smooth: bool, gaussian_sigma: float,
                           out_w: int, out_h: int, palette: Optional[List[str]] = None,
                           labels: Optional[List[str]] = None, nodata_transparent: bool = True) -> str:
    if palette is None:
        palette = PALETTE
    if labels is None:
        labels = LABELS

    z = max(1, int(supersample))
    if (z > 1 or smooth) and NDVI_canvas is not None:
        V = np.where(np.isfinite(NDVI_canvas), NDVI_canvas, 0.0).astype("float32")
        M = np.isfinite(NDVI_canvas).astype("float32")
        if z > 1:
            Vz = zoom(V, z, order=1)
            Mz = zoom(M, z, order=1)
            NDVI_up = np.where(Mz > 1e-6, Vz / Mz, np.nan)
        else:
            NDVI_up = NDVI_canvas
        if smooth:
            sigma = max(0.1, float(gaussian_sigma))
            inside = np.isfinite(NDVI_up).astype("float32")
            vals = np.where(np.isfinite(NDVI_up), NDVI_up, 0.0).astype("float32")
            num = gaussian_filter(vals, sigma=sigma)
            den = gaussian_filter(inside, sigma=sigma)
            NDVI_up = np.where(den > 1e-6, num / den, np.nan)
        bins_up = np.digitize(np.clip(NDVI_up, EDGES[0], EDGES[-1] - 1e-6), EDGES).astype("uint8")
        bins_up = np.where(np.isfinite(NDVI_up), bins_up, 0)
    else:
        bins_up = bins_canvas.astype("uint8")

    Hs, Ws = bins_up.shape
    palette_rgba = [hex_to_rgba_tuple(c) for c in palette]
    if len(palette_rgba) < 2:
        palette_rgba = [(0,0,0,0), (0,255,0,255)]

    rgba = np.zeros((Hs, Ws, 4), dtype=np.uint8)
    mask_valid = (bins_up > 0)

    max_palette_idx = len(palette_rgba) - 1
    bins_idx = np.where(mask_valid, np.minimum(bins_up, max_palette_idx), 0).astype(np.int32)
    lut = np.array(palette_rgba, dtype=np.uint8)
    rgba[..., :] = lut[bins_idx]
    if nodata_transparent:
        rgba[~mask_valid, 3] = 0
    else:
        rgba[~mask_valid, :] = np.array([255,255,255,255], dtype=np.uint8)

    pil = Image.fromarray(rgba, mode="RGBA")
    pil = pil.resize((out_w, out_h), resample=Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def temporal_fill_median(band_key: str, items: List[Any], aoi_geojson, dst_transform, H, W, want_scl=False, max_items=6):
    stacks = []
    used = 0
    for it in items[:max_items]:
        assets = it.assets or {}
        a = assets.get(band_key) or assets.get(band_key.lower())
        url = prefer_http_from_asset(a) if a else None
        if not url:
            continue
        url = sign_href_if_pc(url)
        try:
            with rasterio.open(url) as ds:
                arr = read_band_window(ds, aoi_to_scene(aoi_geojson, ds.crs.to_string()), H, W, dst_transform, Resampling.bilinear)
                if np.nanmax(arr) > 1.5:
                    arr *= 1/10000.0
                stacks.append(np.where(np.isfinite(arr), arr, np.nan))
                used += 1
        except Exception:
            continue
    if used == 0:
        return None
    stacked = np.stack(stacks, axis=0)
    median = np.nanmedian(stacked, axis=0)
    return median

# re-export some commonly used names for convenience
# so other modules can import utils.geometry_mask, utils.mapping, utils.Transformer
geometry_mask = geometry_mask
mapping = mapping
Transformer = Transformer
