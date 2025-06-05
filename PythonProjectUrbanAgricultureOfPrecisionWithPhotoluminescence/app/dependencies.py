from fastapi import Depends, HTTPException, status
from typing import Optional
import os
import sys

# Agregar el directorio raíz al path de Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ml_pipeline import VerticalFarmingPipeline, create_production_pipeline

# Variable global para el pipeline
_ml_pipeline: Optional[VerticalFarmingPipeline] = None


def get_ml_pipeline() -> VerticalFarmingPipeline:
    """Obtener instancia del pipeline ML"""
    global _ml_pipeline

    if _ml_pipeline is None:
        # Intentar cargar el pipeline
        _ml_pipeline = create_production_pipeline()

        if _ml_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="ML models not loaded. Please check model files."
            )

    return _ml_pipeline


# Por ahora, función dummy para autenticación
async def get_current_user():
    """Verificar autenticación (implementación futura)"""
    return {"id": "user123", "name": "Test User"}