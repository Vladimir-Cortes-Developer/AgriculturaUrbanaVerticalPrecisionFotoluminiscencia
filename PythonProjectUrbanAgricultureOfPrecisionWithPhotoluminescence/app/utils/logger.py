import logging
import sys
from typing import Optional
from datetime import datetime

# Configuración de formato para los logs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Crear y configurar un logger

    Args:
        name: Nombre del logger (generalmente __name__)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name or __name__)

    # Solo configurar si no tiene handlers (evitar duplicados)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formato
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)

        # Agregar handler
        logger.addHandler(console_handler)

    return logger


# Logger principal de la aplicación
app_logger = get_logger("vertical_farming_api")