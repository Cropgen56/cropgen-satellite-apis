from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class AvailabilityRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"

class AvailabilityItem(BaseModel):
    date: str
    cloud_cover: Optional[float] = None

class AvailabilityResponse(BaseModel):
    items: List[AvailabilityItem]

class CalculateRequest(BaseModel):
    geometry: Dict[str, Any]
    date: str
    index_name: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    width: Optional[int] = 800
    height: Optional[int] = 800
    supersample: Optional[int] = 1
    smooth: Optional[bool] = False
    gaussian_sigma: Optional[float] = 1.0

class AreaStat(BaseModel):
    label: str
    hectares: float
    percent: float

class CalculateResponse(BaseModel):
    date: str
    index_name: str
    image_base64: str
    bounds: Optional[List[float]] = None
    legend: Optional[List[Dict[str, Any]]] = None
    area_stats: Optional[List[AreaStat]] = None
