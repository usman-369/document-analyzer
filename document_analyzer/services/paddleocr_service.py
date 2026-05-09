import os

# Suppress PaddleOCR verbose output
os.environ["GLOG_minloglevel"] = "2"
os.environ["FLAGS_enable_pir_api"] = "true"
os.environ["FLAGS_print_model_stats"] = "false"

import threading
from paddleocr import PaddleOCR

from ..config import logger


class PaddleOCRService:
    """
    Thread-safe manager for PaddleOCR instances with automatic language detection.
    """

    LANGUAGES = ("es", "en")
    _ocr_instances = {}
    _lock = threading.Lock()

    @classmethod
    def initialize(cls, langs=LANGUAGES):
        """Initialize PaddleOCR for one or more languages - thread safe"""
        with cls._lock:
            for lang in langs:
                logger.info(f"Initializing PaddleOCR for language '{lang}'...")
                if lang not in cls.LANGUAGES:
                    raise ValueError(
                        f"Unsupported language: {lang}. Allowed: {cls.LANGUAGES}"
                    )

                if lang in cls._ocr_instances:
                    logger.info(
                        f"PaddleOCR for language '{lang}' is already initialized."
                    )
                    continue

                logger.debug(f"Loading PaddleOCR models for language '{lang}'...")
                cls._ocr_instances[lang] = PaddleOCR(
                    lang=lang, use_textline_orientation=True
                )

            logger.info(
                f"PaddleOCR initialized for language(s): {list(cls._ocr_instances.keys())}"
            )
            return cls._ocr_instances

    @classmethod
    def get_instance(cls, lang="es"):
        """Return OCR instance for a specific language, initializing if necessary"""
        if lang not in cls.LANGUAGES:
            raise ValueError(f"Unsupported language: {lang}. Allowed: {cls.LANGUAGES}")

        if lang not in cls._ocr_instances:
            logger.info(f"OCR instance for {lang} not found, initializing now...")
            cls.initialize([lang])

        return cls._ocr_instances[lang]

    @classmethod
    def get_auto_instance(cls, passport_file):
        """
        Automatically detect optimal language and return OCR instance for passport.

        Args:
            passport_file: File object or BytesIO containing passport image

        Returns:
            tuple: (ocr_instance, detected_language)
        """
        try:
            from ..utils.passport_language_detector import detect_passport_language

            # Detect language using the PassportLanguageDetector
            detected_lang, detection_details = detect_passport_language(passport_file)
            logger.info(f"Auto-detected passport language: {detected_lang}")

            # Get appropriate OCR instance
            ocr_instance = cls.get_instance(detected_lang)

            return ocr_instance, detected_lang, detection_details

        except Exception as e:
            logger.error(f"Language detection failed: {e}, defaulting to English")
            return cls.get_instance("en"), "en", None

    @classmethod
    def is_ready(cls, lang=None):
        """Check if PaddleOCR is ready (global or per language)"""
        if lang:
            return lang in cls._ocr_instances
        return bool(cls._ocr_instances)

    @classmethod
    def list_loaded_languages(cls):
        """Get list of currently loaded language models"""
        return list(cls._ocr_instances.keys())

    @classmethod
    def clear_cache(cls):
        """Clear all cached OCR instances (use with caution)"""
        with cls._lock:
            logger.warning("Clearing all PaddleOCR instances from cache")
            cls._ocr_instances.clear()
            logger.info("PaddleOCR cache cleared")
