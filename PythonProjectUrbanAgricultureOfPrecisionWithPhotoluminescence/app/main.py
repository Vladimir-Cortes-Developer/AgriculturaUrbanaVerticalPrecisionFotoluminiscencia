from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api.v1.router import api_router
from app.core.ml_pipeline import create_production_pipeline
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Variable global para el pipeline
ml_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    # Startup
    global ml_pipeline
    logger.info("Loading ML models...")
    ml_pipeline = create_production_pipeline()
    if ml_pipeline is None:
        raise RuntimeError("Failed to load ML models")
    logger.info("ML models loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Crear aplicación FastAPI
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurar según necesidad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time * 1000:.2f}ms "
        f"status_code={response.status_code}"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# Incluir routers
app.include_router(api_router, prefix=settings.api_v1_str)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Vertical Farming ML API",
        "version": settings.version,
        "docs": f"{settings.api_v1_str}/docs"
    }


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ml_pipeline is not None
    }


# Manejo global de errores
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )