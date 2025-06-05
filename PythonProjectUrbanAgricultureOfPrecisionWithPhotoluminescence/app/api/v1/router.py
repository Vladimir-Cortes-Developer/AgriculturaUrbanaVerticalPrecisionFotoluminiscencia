from fastapi import APIRouter
from app.api.v1.endpoints import predictions

api_router = APIRouter()

# Incluir routers de endpoints
api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["predictions"]
)