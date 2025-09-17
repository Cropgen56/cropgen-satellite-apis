from fastapi import FastAPI
from availability_dates_api import router as availability_router
from calculate_index_api import router as calculate_router
from timeseries_vegetation_api import router as veg_router
from timeseries_water_api import router as water_router

app = FastAPI(root_path="/v4")

# Mount routers at the same paths as before
app.include_router(availability_router, prefix="/api/availability")
app.include_router(calculate_router, prefix="/api/calculate")
app.include_router(veg_router, prefix="/api/timeseries")
app.include_router(water_router, prefix="/api/timeseries")