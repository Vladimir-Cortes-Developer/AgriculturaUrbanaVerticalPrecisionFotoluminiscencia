from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from app.core.ml_pipeline import VerticalFarmingPipeline
from app.models.requests import SensorReading
from app.models.responses import PredictionResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    Servicio para manejar predicciones y lógica de negocio relacionada
    """

    def __init__(self, pipeline: VerticalFarmingPipeline):
        self.pipeline = pipeline
        self._cache = {}  # Cache simple en memoria
        self._prediction_history = []

    async def predict_single(self, sensor_data: SensorReading) -> PredictionResponse:
        """
        Realizar predicción individual con validación y logging
        """
        try:
            # Convertir a dict
            input_data = sensor_data.dict()

            # Log de entrada
            logger.info(f"Processing prediction for conditions: T={input_data['temperatura_c']}°C, "
                        f"H={input_data['humedad_rel_pct']}%, CO2={input_data['co2_ppm']}ppm")

            # Realizar predicción (ejecutar en thread pool para no bloquear)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.pipeline.predict_both,
                input_data
            )

            # Validar resultado
            validated_result = self.pipeline.validate_predictions(result)

            # Guardar en historial
            self._add_to_history(validated_result)

            # Log de resultado
            logger.info(f"Prediction completed: Efficiency={result['eficiencia_fotosintetica_pct']:.1f}%, "
                        f"Fluorescence={result['fotoluminiscencia_intensidad']:.1f}")

            return PredictionResponse(**validated_result)

        except Exception as e:
            logger.error(f"Error in prediction service: {str(e)}", exc_info=True)
            raise

    async def predict_batch(self, sensor_readings: List[SensorReading]) -> List[PredictionResponse]:
        """
        Realizar predicciones en lote de manera eficiente
        """
        predictions = []

        # Procesar en paralelo usando asyncio
        tasks = [self.predict_single(reading) for reading in sensor_readings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrar errores y devolver solo predicciones exitosas
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch prediction {i} failed: {str(result)}")
            else:
                predictions.append(result)

        logger.info(f"Batch prediction completed: {len(predictions)}/{len(sensor_readings)} successful")

        return predictions

    def _add_to_history(self, prediction: Dict):
        """Agregar predicción al historial (máximo 1000 registros)"""
        self._prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction
        })

        # Mantener solo las últimas 1000 predicciones
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]

    def get_statistics(self) -> Dict:
        """Obtener estadísticas del servicio"""
        if not self._prediction_history:
            return {
                'total_predictions': 0,
                'avg_efficiency': 0,
                'avg_fluorescence': 0,
                'avg_health_score': 0
            }

        efficiencies = [h['prediction']['eficiencia_fotosintetica_pct']
                        for h in self._prediction_history]
        fluorescences = [h['prediction']['fotoluminiscencia_intensidad']
                         for h in self._prediction_history]
        health_scores = [h['prediction']['health_score']
                         for h in self._prediction_history]

        return {
            'total_predictions': len(self._prediction_history),
            'avg_efficiency': sum(efficiencies) / len(efficiencies),
            'avg_fluorescence': sum(fluorescences) / len(fluorescences),
            'avg_health_score': sum(health_scores) / len(health_scores),
            'last_prediction_time': self._prediction_history[-1]['timestamp']
        }