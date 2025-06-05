import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class VerticalFarmingPipeline:
    """
    Pipeline de producción para predicción de eficiencia fotosintética
    y fotoluminiscencia en agricultura vertical.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.metadata = {}
        self.feature_ranges = {}
        self.constants = {
            'temp_optima': 23.0,
            'co2_optimo': 800.0,
            'par_optimo': 450.0,
            'humedad_optima': 65.0,
            'vpd_optimo': 1.2
        }

    def load_trained_models(self, models_dir=None):
        """Cargar modelos ya entrenados y configuración"""
        print("Cargando modelos entrenados...")

        try:
            import os
            from pathlib import Path

            # IMPORTANTE: Usar la ubicación del archivo, no el directorio de trabajo
            current_file = Path(__file__).resolve()

            # Navegar desde app/core/ml_pipeline.py hasta la raíz del proyecto
            # current_file.parent = core/
            # current_file.parent.parent = app/
            # current_file.parent.parent.parent = raíz del proyecto
            project_root = current_file.parent.parent.parent

            print(f"Ubicación del archivo ml_pipeline.py: {current_file}")
            print(f"Raíz del proyecto detectada: {project_root}")

            # Usar rutas absolutas basadas en la raíz del proyecto
            if models_dir is None:
                models_dir = project_root / 'models'
            else:
                models_dir = Path(models_dir)

            # Rutas a los archivos necesarios
            training_results_path = models_dir / 'training_results.json'
            production_config_path = models_dir / 'production_config.json'
            feature_metadata_path = project_root / 'data' / 'processed' / 'feature_engineering_metadata.json'

            print(f"Buscando training_results.json en: {training_results_path}")

            # Verificar que existe antes de intentar abrir
            if not training_results_path.exists():
                print(f"ERROR: No existe {training_results_path}")
                print(
                    f"Contenido del directorio models: {list(models_dir.glob('*'))}" if models_dir.exists() else "El directorio models no existe")
                return False

            # Cargar metadata de entrenamiento
            with open(training_results_path, 'r') as f:
                training_results = json.load(f)
            print("✓ training_results.json cargado")

            # Verificar y cargar feature metadata
            if not feature_metadata_path.exists():
                print(f"ERROR: No existe {feature_metadata_path}")
                return False

            with open(feature_metadata_path, 'r') as f:
                feature_metadata = json.load(f)
            print("✓ feature_engineering_metadata.json cargado")

            # Verificar y cargar production config
            if not production_config_path.exists():
                print(f"ERROR: No existe {production_config_path}")
                return False

            with open(production_config_path, 'r') as f:
                production_config = json.load(f)
            print("✓ production_config.json cargado")

            # Guardar rangos de features para validación
            self.feature_ranges = production_config['feature_validation']['feature_ranges']

            # Definir columnas de features
            self.feature_columns = {
                'eficiencia': feature_metadata['selected_features'],
                'fotoluminiscencia': feature_metadata['selected_features'] + ['eficiencia_fotosintetica_pct']
            }

            # Cargar mejores modelos
            best_models = training_results['best_models']

            for target, info in best_models.items():
                model_name = info['name']

                # Construir ruta al modelo
                model_file = models_dir / f"{target}_{model_name.lower()}.joblib"

                print(f"Buscando modelo: {model_file}")

                if model_file.exists():
                    self.models[target] = joblib.load(str(model_file))
                    print(f"  ✓ {target}: {model_name} cargado")

                    # Cargar scaler si existe
                    scaler_file = models_dir / f"scaler_{target}_{model_name.lower()}.joblib"
                    if scaler_file.exists():
                        self.scalers[target] = joblib.load(str(scaler_file))
                        print(f"  ✓ Scaler para {target} cargado")
                else:
                    print(f"  ✗ ERROR: No se encontró modelo para {target}")
                    print(f"  Archivo esperado: {model_file}")
                    return False

                # Guardar metadata
                self.metadata[target] = {
                    'model_name': model_name,
                    'test_mae': info['test_mae'],
                    'test_r2': info['test_r2'],
                    'features': self.feature_columns[target]
                }

            print("✓ Modelos cargados exitosamente")
            return True

        except Exception as e:
            print(f"✗ Error cargando modelos: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_input(self, input_data):
        """Validar datos de entrada completos"""
        # Features básicas requeridas
        required_basic_features = [
            'temperatura_c', 'humedad_rel_pct', 'co2_ppm', 'par_umol_m2_s',
            'pm2_5_ugm3', 'pm10_ugm3', 'no2_ugm3', 'o3_ugm3',
            'vocs_mgm3', 'aqi_indice', 'vpd_kpa', 'espectro_pico_nm',
            'hora_dia', 'dia_semana', 'mes'
        ]

        # Verificar features faltantes
        missing_features = [f for f in required_basic_features if f not in input_data]
        if missing_features:
            raise ValueError(f"Features básicas faltantes: {missing_features}")

        # Validar rangos básicos
        validation_ranges = {
            'temperatura_c': (10, 40),
            'humedad_rel_pct': (20, 95),
            'co2_ppm': (200, 2000),
            'par_umol_m2_s': (50, 1000),
            'pm2_5_ugm3': (0, 200),
            'pm10_ugm3': (0, 300),
            'no2_ugm3': (0, 200),
            'o3_ugm3': (0, 200),
            'vocs_mgm3': (0, 5),
            'aqi_indice': (0, 100),
            'vpd_kpa': (0, 5),
            'espectro_pico_nm': (400, 800),
            'hora_dia': (0, 23),
            'dia_semana': (0, 6),
            'mes': (1, 12)
        }

        for feature, (min_val, max_val) in validation_ranges.items():
            if feature in input_data:
                value = input_data[feature]
                if not (min_val <= value <= max_val):
                    print(f"⚠️  Advertencia: {feature} fuera de rango esperado [{min_val}, {max_val}]: {value}")

        return True

    def _apply_complete_feature_engineering(self, input_data):
        """
        Aplicar feature engineering completo
        TODAS las features necesarias para los modelos
        """
        # Hacer una copia para no modificar el original
        data = input_data.copy()

        # 1. FEATURES DE DISTANCIA ÓPTIMA
        data['temperatura_distancia_optima'] = abs(data['temperatura_c'] - self.constants['temp_optima'])
        data['temp_optima_dist'] = data['temperatura_distancia_optima']  # alias
        data['co2_distancia_optima'] = abs(data['co2_ppm'] - self.constants['co2_optimo'])
        data['par_distancia_optima'] = abs(data['par_umol_m2_s'] - self.constants['par_optimo'])

        # 2. FEATURES DE INTERACCIÓN
        data['temp_humedad_interaction'] = data['temperatura_c'] * data['humedad_rel_pct']
        data['temp_co2_interaction'] = data['temperatura_c'] * data['co2_ppm']
        data['par_co2_interaction'] = data['par_umol_m2_s'] * data['co2_ppm']
        data['humedad_vpd_interaction'] = data['humedad_rel_pct'] * data['vpd_kpa']

        # 3. RATIOS Y PROPORCIONES
        data['co2_par_ratio'] = data['co2_ppm'] / (data['par_umol_m2_s'] + 1)
        data['par_temp_ratio'] = data['par_umol_m2_s'] / (data['temperatura_c'] + 1)

        # 4. TRANSFORMACIONES LOGARÍTMICAS Y RAÍZ
        data['co2_ppm_log'] = np.log(data['co2_ppm'])
        data['co2_ppm_sqrt'] = np.sqrt(data['co2_ppm'])
        data['vocs_mgm3_log'] = np.log(data['vocs_mgm3'] + 0.0001)  # Evitar log(0)

        # 5. FEATURES TEMPORALES CÍCLICAS
        data['mes_sin'] = np.sin(2 * np.pi * data['mes'] / 12)
        data['mes_cos'] = np.cos(2 * np.pi * data['mes'] / 12)

        # 6. AJUSTES (desviación de valores óptimos)
        data['ajuste_temp_c'] = data['temperatura_c'] - self.constants['temp_optima']
        data['ajuste_co2_ppm'] = data['co2_ppm'] - self.constants['co2_optimo']

        # 7. ÍNDICES COMPLEJOS
        # Índice de estrés ambiental (normalizado 0-1)
        stress_temp = data['temperatura_distancia_optima'] / 10  # Normalizar por rango
        stress_co2 = data['co2_distancia_optima'] / 500
        stress_vpd = abs(data['vpd_kpa'] - self.constants['vpd_optimo']) / 3
        data['indice_estres_ambiental'] = (stress_temp + stress_co2 + stress_vpd) / 3
        data['indice_estres_ambiental'] = np.clip(data['indice_estres_ambiental'], 0, 1)

        # Índice de condiciones de luz
        luz_quality = data['par_umol_m2_s'] / 800  # Normalizar por máximo
        spectrum_quality = 1 - abs(data['espectro_pico_nm'] - 680) / 60  # 680nm es óptimo
        data['indice_condiciones_luz'] = (luz_quality + spectrum_quality) / 2
        data['indice_condiciones_luz'] = np.clip(data['indice_condiciones_luz'], 0, 1)

        # Calidad de luz ratio
        data['calidad_luz_ratio'] = data['indice_condiciones_luz'] * (1 - data['indice_estres_ambiental'])

        # 8. EFICIENCIA NORMALIZADA (CRÍTICA)
        # Esta es la feature más importante para el modelo de eficiencia
        # Calculamos una estimación basada en condiciones óptimas
        temp_factor = 1 - (data['temperatura_distancia_optima'] / 10)
        co2_factor = 1 - (data['co2_distancia_optima'] / 500)
        light_factor = data['par_umol_m2_s'] / 800
        humidity_factor = 1 - abs(data['humedad_rel_pct'] - 65) / 30

        # Combinar factores con pesos
        data['eficiencia_normalizada'] = (
                0.3 * temp_factor +
                0.25 * co2_factor +
                0.3 * light_factor +
                0.15 * humidity_factor
        )
        data['eficiencia_normalizada'] = np.clip(data['eficiencia_normalizada'], 0.2, 0.85)

        # 9. PURIFICACIÓN REQUERIDA
        # Basado en contaminantes
        contaminant_level = (
                                    data['pm2_5_ugm3'] / 50 +
                                    data['pm10_ugm3'] / 100 +
                                    data['no2_ugm3'] / 100 +
                                    data['vocs_mgm3'] / 2
                            ) / 4
        data['purificacion_requerida_pct'] = np.clip(contaminant_level * 100, 0, 100)

        # 10. VERIFICAR QUE TENEMOS TODAS LAS FEATURES NECESARIAS
        required_features = self.feature_columns.get('eficiencia', [])
        for feature in required_features:
            if feature not in data:
                print(f"⚠️  Feature faltante después de engineering: {feature}")
                # Intentar calcular o asignar valor por defecto
                if feature in self.feature_ranges:
                    data[feature] = self.feature_ranges[feature]['mean']
                else:
                    data[feature] = 0

        return data

    def predict_eficiencia(self, input_data):
        """Predecir eficiencia fotosintética"""
        if 'eficiencia' not in self.models:
            raise ValueError("Modelo de eficiencia no disponible")

        # Aplicar feature engineering completo
        if isinstance(input_data, dict):
            input_data_eng = self._apply_complete_feature_engineering(input_data)

            # Crear DataFrame con features en el orden correcto
            features_df = pd.DataFrame([input_data_eng])[self.feature_columns['eficiencia']]
            features_array = features_df.values
        else:
            features_array = input_data

        # Aplicar scaler si existe
        if 'eficiencia' in self.scalers:
            features_array = self.scalers['eficiencia'].transform(features_array)

        # Predicción
        prediction = self.models['eficiencia'].predict(features_array)[0]

        # Asegurar rango válido (0-100)
        return float(np.clip(prediction, 0, 100))

    def predict_fotoluminiscencia(self, input_data, eficiencia_value=None):
        """Predecir intensidad de fotoluminiscencia"""
        if 'fotoluminiscencia' not in self.models:
            raise ValueError("Modelo de fotoluminiscencia no disponible")

        # Aplicar feature engineering completo
        if isinstance(input_data, dict):
            input_data_eng = self._apply_complete_feature_engineering(input_data)

            # Agregar eficiencia si se proporciona
            if eficiencia_value is not None:
                input_data_eng['eficiencia_fotosintetica_pct'] = eficiencia_value
            elif 'eficiencia_fotosintetica_pct' not in input_data_eng:
                # Si no se proporciona, predecirla primero
                input_data_eng['eficiencia_fotosintetica_pct'] = self.predict_eficiencia(input_data)

            # Crear DataFrame con features en el orden correcto
            features_df = pd.DataFrame([input_data_eng])[self.feature_columns['fotoluminiscencia']]
            features_array = features_df.values
        else:
            features_array = input_data

        # Aplicar scaler si existe
        if 'fotoluminiscencia' in self.scalers:
            features_array = self.scalers['fotoluminiscencia'].transform(features_array)

        # Predicción
        prediction = self.models['fotoluminiscencia'].predict(features_array)[0]

        # Asegurar rango válido (0-100)
        return float(np.clip(prediction, 0, 100))

    def predict_both(self, input_data):
        """Predecir tanto eficiencia como fotoluminiscencia"""
        try:
            # Validar entrada
            self.validate_input(input_data)

            # Predecir eficiencia primero
            eficiencia = self.predict_eficiencia(input_data)

            # Predecir fotoluminiscencia usando la eficiencia predicha
            fotoluminiscencia = self.predict_fotoluminiscencia(input_data, eficiencia)

            # Calcular métricas adicionales
            health_score = (eficiencia * 0.6 + fotoluminiscencia * 0.4)

            return {
                'eficiencia_fotosintetica_pct': round(eficiencia, 2),
                'fotoluminiscencia_intensidad': round(fotoluminiscencia, 2),
                'health_score': round(health_score, 2),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error en predicción: {e}")
            raise

    def get_model_info(self):
        """Obtener información de modelos cargados"""
        return self.metadata

    def get_feature_importance(self, target='eficiencia'):
        """Obtener importancia de features si el modelo lo soporta"""
        if target not in self.models:
            return None

        model = self.models[target]
        if hasattr(model, 'feature_importances_'):
            features = self.feature_columns[target]
            importance = model.feature_importances_

            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)

            return importance_df

        return None

    def validate_predictions(self, predictions):
        """Validar que las predicciones estén en rangos esperados"""
        warnings = []

        if predictions['eficiencia_fotosintetica_pct'] < 20:
            warnings.append("⚠️  Eficiencia muy baja (<20%)")
        elif predictions['eficiencia_fotosintetica_pct'] > 80:
            warnings.append("⚠️  Eficiencia muy alta (>80%)")

        if predictions['fotoluminiscencia_intensidad'] < 10:
            warnings.append("⚠️  Fotoluminiscencia muy baja (<10)")
        elif predictions['fotoluminiscencia_intensidad'] > 90:
            warnings.append("⚠️  Fotoluminiscencia muy alta (>90)")

        if warnings:
            predictions['warnings'] = warnings

        return predictions


def create_production_pipeline():
    """Crear pipeline para producción"""
    pipeline = VerticalFarmingPipeline()

    if pipeline.load_trained_models():
        print("✓ Pipeline de producción listo")
        return pipeline
    else:
        print("✗ Error creando pipeline de producción")
        return None


def run_diagnostic_test(pipeline):
    """Ejecutar prueba diagnóstica del pipeline"""
    print("\n🔍 EJECUTANDO PRUEBA DIAGNÓSTICA...")

    # Caso de prueba con condiciones óptimas
    optimal_conditions = {
        'temperatura_c': 23.0,
        'humedad_rel_pct': 65.0,
        'co2_ppm': 800,
        'par_umol_m2_s': 450,
        'pm2_5_ugm3': 10,
        'pm10_ugm3': 20,
        'no2_ugm3': 25,
        'o3_ugm3': 60,
        'vocs_mgm3': 0.3,
        'aqi_indice': 50,
        'vpd_kpa': 1.2,
        'espectro_pico_nm': 680,
        'hora_dia': 12,
        'dia_semana': 3,
        'mes': 6
    }

    # Caso de prueba con condiciones sub-óptimas
    suboptimal_conditions = {
        'temperatura_c': 30.0,
        'humedad_rel_pct': 85.0,
        'co2_ppm': 400,
        'par_umol_m2_s': 200,
        'pm2_5_ugm3': 30,
        'pm10_ugm3': 50,
        'no2_ugm3': 50,
        'o3_ugm3': 100,
        'vocs_mgm3': 0.8,
        'aqi_indice': 80,
        'vpd_kpa': 0.5,
        'espectro_pico_nm': 720,
        'hora_dia': 18,
        'dia_semana': 5,
        'mes': 12
    }

    print("\n📊 Condiciones Óptimas:")
    results_optimal = pipeline.predict_both(optimal_conditions)
    validated_optimal = pipeline.validate_predictions(results_optimal)
    for key, value in validated_optimal.items():
        if key != 'warnings':
            print(f"  - {key}: {value}")

    print("\n📊 Condiciones Sub-óptimas:")
    results_suboptimal = pipeline.predict_both(suboptimal_conditions)
    validated_suboptimal = pipeline.validate_predictions(results_suboptimal)
    for key, value in validated_suboptimal.items():
        if key != 'warnings':
            print(f"  - {key}: {value}")

    if 'warnings' in validated_optimal:
        print("\n⚠️  Advertencias (óptimas):", validated_optimal['warnings'])
    if 'warnings' in validated_suboptimal:
        print("\n⚠️  Advertencias (sub-óptimas):", validated_suboptimal['warnings'])

    print("\n✓ Prueba diagnóstica completada")


if __name__ == "__main__":
    # Crear y probar pipeline
    print("=" * 60)
    print("PIPELINE DE PRODUCCIÓN - AGRICULTURA VERTICAL")
    print("=" * 60)

    pipeline = create_production_pipeline()

    if pipeline:
        # Mostrar información de modelos
        print("\n📈 INFORMACIÓN DE MODELOS:")
        for target, info in pipeline.get_model_info().items():
            print(f"\n{target.upper()}:")
            print(f"  - Modelo: {info['model_name']}")
            print(f"  - MAE: {info['test_mae']:.6f}")
            print(f"  - R²: {info['test_r2']:.6f}")
            print(f"  - Features: {len(info['features'])}")

        # Ejecutar prueba diagnóstica
        run_diagnostic_test(pipeline)

        # Ejemplo de uso individual
        print("\n" + "=" * 60)
        print("EJEMPLO DE USO:")
        print("=" * 60)

        ejemplo_input = {
            'temperatura_c': 25.5,
            'humedad_rel_pct': 70.0,
            'co2_ppm': 900,
            'par_umol_m2_s': 500,
            'pm2_5_ugm3': 15,
            'pm10_ugm3': 25,
            'no2_ugm3': 30,
            'o3_ugm3': 65,
            'vocs_mgm3': 0.4,
            'aqi_indice': 55,
            'vpd_kpa': 1.5,
            'espectro_pico_nm': 690,
            'hora_dia': 14,
            'dia_semana': 2,
            'mes': 7
        }

        try:
            resultados = pipeline.predict_both(ejemplo_input)

            print("\n🌱 PREDICCIONES:")
            print(f"  - Eficiencia Fotosintética: {resultados['eficiencia_fotosintetica_pct']}%")
            print(f"  - Fotoluminiscencia: {resultados['fotoluminiscencia_intensidad']}")
            print(f"  - Score de Salud: {resultados['health_score']}")

            # Mostrar importancia de features si está disponible
            importance = pipeline.get_feature_importance('fotoluminiscencia')
            if importance is not None:
                print("\n📊 TOP 5 FEATURES MÁS IMPORTANTES (Fotoluminiscencia):")
                print(importance.head().to_string(index=False))

        except Exception as e:
            print(f"\n✗ Error en predicción: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)