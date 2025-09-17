from fastapi import APIRouter, HTTPException
from models import AvailabilityRequest, AvailabilityResponse, AvailabilityItem
import utils
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=AvailabilityResponse)
def availability(req: AvailabilityRequest):
    geom = req.geometry
    try:
        datetime.strptime(req.start_date, "%Y-%m-%d")
        datetime.strptime(req.end_date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="start_date and end_date must be YYYY-MM-DD")
    prefer_pc = True
    if req.provider and req.provider.lower() == "aws":
        prefer_pc = False
    satellite = (req.satellite or "s2").lower()
    coll_map = {"s2": ["sentinel-2-l2a"], "s1": ["sentinel-1"]}
    collections = coll_map.get(satellite, ["sentinel-2-l2a"])
    try:
        items_pc = utils.search_planetary(collections, geom, f"{req.start_date}/{req.end_date}", limit=500) if prefer_pc else []
        items_aws = utils.search_aws(collections, geom, f"{req.start_date}/{req.end_date}", limit=500) if not prefer_pc else utils.search_aws(collections, geom, f"{req.start_date}/{req.end_date}", limit=500)
        all_items = (items_pc or []) + (items_aws or [])
        if not all_items:
            return {"items": []}
        date_map = {}
        for it in all_items:
            dt = it.properties.get("datetime") or it.properties.get("acquired") or ""
            if not dt:
                continue
            date_key = str(dt)[:10]
            cloud = it.properties.get("eo:cloud_cover") or it.properties.get("cloud_cover") or None
            try:
                cloud = float(cloud) if cloud is not None else None
            except Exception:
                cloud = None
            date_map.setdefault(date_key, []).append(cloud if cloud is not None else 999.0)
        out_items = []
        for d, clouds in sorted(date_map.items()):
            clouds_valid = [c for c in clouds if c is not None and c < 999.0]
            best = float(min(clouds_valid)) if clouds_valid else None
            out_items.append(AvailabilityItem(date=d, cloud_cover=best))
        return {"items": out_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
