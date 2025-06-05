from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import time
from app.models.requests import SensorReading, BatchPredictionRequest
from app.models.responses import PredictionResponse, BatchPredictionResponse
from app.core.ml_pipeline import VerticalFarmingPipeline
from app.dependencies import get_ml_pipeline, get_current_user
# from app.services.prediction_service import PredictionService  # Comentado por ahora
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
        reading: SensorReading,
        pipeline: VerticalFarmingPipeline = Depends(get_ml_pipeline),
        # current_user: dict = Depends(get_current_user)  # Comentado temporalmente
):
    """
    Realizar predicción para una lectura de sensores
    """
    try:
        start_time = time.time()

        # Convertir a dict para el pipeline
        input_data = reading.dict()

        # Realizar predicción
        results = pipeline.predict_both(input_data)

        # Agregar tiempo de procesamiento
        processing_time = (time.time() - start_time) * 1000
        results['processing_time_ms'] = round(processing_time, 2)

        # Log para monitoreo
        logger.info(f"Prediction completed in {processing_time:.2f}ms")

        return PredictionResponse(**results)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
        request: BatchPredictionRequest,
        pipeline: VerticalFarmingPipeline = Depends(get_ml_pipeline),
        # current_user: dict = Depends(get_current_user)  # Comentado temporalmente
):
    """
    Realizar predicciones en lote (máximo 100)
    """
    try:
        start_time = time.time()
        predictions = []

        for reading in request.readings:
            input_data = reading.dict()
            result = pipeline.predict_both(input_data)
            predictions.append(PredictionResponse(**result))

        total_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_processing_time_ms=round(total_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/importance/{target}")
async def get_feature_importance(
        target: str,
        pipeline: VerticalFarmingPipeline = Depends(get_ml_pipeline),
        # current_user: dict = Depends(get_current_user)  # Comentado temporalmente
):
    """
    Obtener importancia de features para un target específico
    """
    if target not in ['eficiencia', 'fotoluminiscencia']:
        raise HTTPException(status_code=400, detail="Target inválido")

    importance = pipeline.get_feature_importance(target)

    if importance is None:
        raise HTTPException(
            status_code=400,
            detail="El modelo no soporta importancia de features"
        )

    return {
        "target": target,
        "features": importance.to_dict('records'),
        "model_type": pipeline.metadata[target]['model_name']
    }


@router.get("/health")
async def health_check():
    """Verificar estado del servicio de predicciones"""
    return {
        "status": "healthy",
        "service": "predictions",
        "timestamp": time.time()
    }