from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from availability_dates_api import router as availability_router
from calculate_index_api import router as calculate_router
from timeseries_vegetation_api import router as veg_router
from timeseries_water_api import router as water_router

app = FastAPI(root_path="/v4")

# Enable CORS for localhost:3000
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers with unique prefixes
app.include_router(availability_router, prefix="/api/availability", tags=["Availability"])
app.include_router(calculate_router, prefix="/api/calculate", tags=["Calculate Index"])
app.include_router(veg_router, prefix="/api/timeseries/vegetation", tags=["Vegetation Timeseries"])
app.include_router(water_router, prefix="/api/timeseries/water", tags=["Water Timeseries"])

# Optional root endpoint
@app.get("/")
def root():
    return {"message": "CropGen API v4 is running"}
