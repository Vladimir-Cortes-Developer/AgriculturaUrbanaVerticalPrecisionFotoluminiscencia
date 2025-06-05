from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionResponse(BaseModel):
    """Respuesta de predicción individual"""
    eficiencia_fotosintetica_pct: float = Field(..., description="Eficiencia fotosintética %")
    fotoluminiscencia_intensidad: float = Field(..., description="Intensidad fotoluminiscencia")
    health_score: float = Field(..., description="Score de salud general")
    timestamp: datetime
    warnings: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Respuesta para predicciones en lote"""
    predictions: List[PredictionResponse]
    batch_size: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Estado de salud de la API"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float
    last_prediction: Optional[datetime]


class FeatureImportanceResponse(BaseModel):
    """Importancia de features"""
    target: str
    features: List[Dict[str, float]]
    model_type: str


class ErrorResponse(BaseModel):
    """Respuesta de error estándar"""
    error: str
    detail: Optional[str]
    timestamp: datetime
    request_id: Optional[str]