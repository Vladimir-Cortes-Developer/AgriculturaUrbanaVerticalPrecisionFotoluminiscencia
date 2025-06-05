import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_single_prediction():
    test_data = {
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

    # Agregar token de autenticación en producción
    response = client.post("/api/v1/predict", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert "eficiencia_fotosintetica_pct" in data
    assert "fotoluminiscencia_intensidad" in data
    assert 0 <= data["eficiencia_fotosintetica_pct"] <= 100