from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class SensorReading(BaseModel):
    """Modelo para una lectura de sensores"""
    temperatura_c: float = Field(..., ge=10, le=40, description="Temperatura en Celsius")
    humedad_rel_pct: float = Field(..., ge=20, le=95, description="Humedad relativa %")
    co2_ppm: float = Field(..., ge=200, le=2000, description="CO2 en ppm")
    par_umol_m2_s: float = Field(..., ge=50, le=1000, description="PAR en μmol/m²/s")
    pm2_5_ugm3: float = Field(..., ge=0, le=200, description="PM2.5 en μg/m³")
    pm10_ugm3: float = Field(..., ge=0, le=300, description="PM10 en μg/m³")
    no2_ugm3: float = Field(..., ge=0, le=200, description="NO2 en μg/m³")
    o3_ugm3: float = Field(..., ge=0, le=200, description="O3 en μg/m³")
    vocs_mgm3: float = Field(..., ge=0, le=5, description="VOCs en mg/m³")
    aqi_indice: float = Field(..., ge=0, le=100, description="Índice AQI")
    vpd_kpa: float = Field(..., ge=0, le=5, description="VPD en kPa")
    espectro_pico_nm: int = Field(..., ge=400, le=800, description="Pico espectral en nm")
    hora_dia: int = Field(..., ge=0, le=23, description="Hora del día")
    dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana")
    mes: int = Field(..., ge=1, le=12, description="Mes")

    class Config:
        json_schema_extra = {
            "example": {
                "temperatura_c": 23.5,
                "humedad_rel_pct": 65.0,
                "co2_ppm": 800,
                "par_umol_m2_s": 450,
                "pm2_5_ugm3": 10,
                "pm10_ugm3": 20,
                "no2_ugm3": 25,
                "o3_ugm3": 60,
                "vocs_mgm3": 0.3,
                "aqi_indice": 50,
                "vpd_kpa": 1.2,
                "espectro_pico_nm": 680,
                "hora_dia": 12,
                "dia_semana": 3,
                "mes": 6
            }
        }


class BatchPredictionRequest(BaseModel):
    """Modelo para predicciones en lote"""
    readings: List[SensorReading] = Field(..., max_items=100)
    include_features: bool = Field(False, description="Incluir features calculadas")


class PredictionOptions(BaseModel):
    """Opciones para personalizar predicciones"""
    return_confidence: bool = False
    return_features: bool = False
    validate_ranges: bool = True