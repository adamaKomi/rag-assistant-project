"""
Système de logging structuré avec rotation et métadonnées enrichies.
Facilite le debugging et le monitoring en production.
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from pythonjsonlogger import jsonlogger

from .config import settings


class StructuredLogger:
    """Logger structuré avec format JSON pour faciliter l'analyse"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Configuration du logger"""
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Éviter la duplication des handlers
        if not self.logger.handlers:
            # Handler console
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = self._get_console_formatter()
            console_handler.setFormatter(console_formatter)
            
            # Handler fichier avec rotation
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "rag_assistant.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_formatter = self._get_file_formatter()
            file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def _get_console_formatter(self):
        """Formatter pour la console (lisible)"""
        if settings.environment == "development":
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            return jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
    
    def _get_file_formatter(self):
        """Formatter pour les fichiers (JSON structuré)"""
        return jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
        )
    
    def _add_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ajoute du contexte systématique aux logs"""
        context = {
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            context.update(extra)
        return {"extra": context}
    
    def info(self, message: str, **kwargs):
        """Log info avec contexte"""
        self.logger.info(message, **self._add_context(kwargs))
    
    def error(self, message: str, **kwargs):
        """Log erreur avec contexte"""
        self.logger.error(message, **self._add_context(kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning avec contexte"""
        self.logger.warning(message, **self._add_context(kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug avec contexte"""
        self.logger.debug(message, **self._add_context(kwargs))
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log spécialisé pour les performances"""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_product_matching(self, stage: str, results_count: int, confidence: float, **kwargs):
        """Log spécialisé pour le matching de produits"""
        self.info(
            f"Product matching: {stage}",
            matching_stage=stage,
            results_count=results_count,
            confidence=confidence,
            **kwargs
        )


def get_logger(name: str) -> StructuredLogger:
    """Factory pour créer des loggers"""
    return StructuredLogger(name)
