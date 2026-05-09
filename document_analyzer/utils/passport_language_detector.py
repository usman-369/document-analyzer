import re
import cv2
import numpy as np
from typing import Optional, Dict, Tuple, Union

from .common_utils import ensure_bytesio, preprocess_image
from ..config import (
    logger,
    SPANISH_COUNTRIES,
    SPANISH_KEYWORDS,
    COUNTRY_NAME_MAPPINGS,
)


class PassportLanguageDetector:
    """
    Passport language detector for determining optimal OCR language.

    This class analyzes passport documents to determine whether Spanish or English
    OCR would provide better results based on document content and country indicators.

    Example:
        >>> detector = PassportLanguageDetector()
        >>> language = detector.detect_language(passport_file)
        >>> print(f"Use {language} OCR for this passport")

        # Or use class methods directly:
        >>> language = PassportLanguageDetector.detect_passport_language(passport_file)
    """

    def __init__(self, default_confidence_threshold=3):
        """
        Initialize the detector.

        Args:
            default_confidence_threshold: Minimum score required to detect Spanish
        """
        self.confidence_threshold = default_confidence_threshold
        self.logger = logger

    def _prepare_image(self, passport_file):
        """Convert various input types to OpenCV image."""
        try:
            if isinstance(passport_file, np.ndarray):
                return passport_file.copy()
            elif isinstance(passport_file, str):
                # File path
                return cv2.imread(passport_file)
            else:
                # File object or BytesIO
                passport_stream = ensure_bytesio(passport_file)
                image_data = np.frombuffer(passport_stream.read(), np.uint8)
                return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Error preparing image: {e}")
            return None

    def _get_ocr_instance(self, ocr_instance):
        """Get or create OCR instance."""
        if ocr_instance is not None:
            return ocr_instance

        from ..services.paddleocr_service import PaddleOCRService

        return PaddleOCRService.get_instance("es")

    def _extract_text_from_image(self, image: np.ndarray, ocr):
        """Extract all text from image using OCR."""
        try:
            # Preprocess image for better OCR
            processed_image = preprocess_image(image)

            # Run OCR
            results = ocr.predict(processed_image)

            if not results or not results[0]:
                return ""

            # Combine all text
            all_text = ""
            for line in results[0]:
                if len(line) >= 2:
                    text = line[1][0].upper().strip()
                    all_text += " " + text

            return all_text.strip()

        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""

    def _analyze_language_indicators(self, text):
        """Analyze text for Spanish language indicators."""
        spanish_score = 0
        total_indicators = 0
        found_indicators = {
            "spanish_keywords": [],
            "spanish_countries": [],
            "spanish_chars": [],
            "spanish_patterns": [],
            "country_names": [],
        }

        # Method 1: Spanish keywords
        words = text.split()
        for word in words:
            if word in SPANISH_KEYWORDS:
                spanish_score += 3
                total_indicators += 1
                found_indicators["spanish_keywords"].append(word)

        # Method 2: Spanish country codes
        for country in SPANISH_COUNTRIES:
            if country in text:
                spanish_score += 5
                total_indicators += 1
                found_indicators["spanish_countries"].append(country)

        # Method 3: Spanish-specific characters
        spanish_chars = ["Ñ", "Á", "É", "Í", "Ó", "Ú", "Ü"]
        for char in spanish_chars:
            if char in text:
                spanish_score += 2
                total_indicators += 1
                found_indicators["spanish_chars"].append(char)

        # Method 4: Spanish phrase patterns
        spanish_patterns = [
            r"REPUBLICA\s+DE",
            r"REPÚBLICA\s+DE",
            r"LUGAR\s+DE\s+NAC",
            r"FECHA\s+DE\s+NAC",
            r"DOCUMENTO\s+DE\s+IDENTIDAD",
            r"CEDULA\s+DE",
            r"CÉDULA\s+DE",
            r"PASAPORTE\s+DE",
        ]

        for pattern in spanish_patterns:
            matches = re.findall(pattern, text)
            if matches:
                spanish_score += 3
                total_indicators += len(matches)
                found_indicators["spanish_patterns"].extend(matches)

        # Method 5: Country names
        for country_name, code in COUNTRY_NAME_MAPPINGS.items():
            if country_name in text and code in SPANISH_COUNTRIES:
                spanish_score += 4
                total_indicators += 1
                found_indicators["country_names"].append(country_name)

        return {
            "spanish_score": spanish_score,
            "total_indicators": total_indicators,
            "found_indicators": found_indicators,
        }

    def _calculate_confidence(self, spanish_score, total_indicators):
        """Calculate confidence level based on detection metrics."""
        if spanish_score >= 8 or total_indicators >= 4:
            return "high"
        elif spanish_score >= 3 or total_indicators >= 2:
            return "medium"
        else:
            return "low"

    def detect_with_details(self, passport_file, ocr_instance=None):
        """
        Detect language with detailed analysis information.

        Args:
            passport_file: File object, BytesIO, image path, or numpy array
            ocr_instance: Optional pre-initialized OCR instance

        Returns:
            Tuple[str, Dict]: (detected_language, analysis_details)
        """
        try:
            # Convert input to OpenCV image
            image = self._prepare_image(passport_file)
            if image is None:
                return "en", {"error": "Could not decode image", "confidence": "low"}

            # Get OCR instance
            ocr = self._get_ocr_instance(ocr_instance)

            # Extract text
            all_text = self._extract_text_from_image(image, ocr)
            if not all_text:
                return "en", {"error": "No text extracted", "confidence": "low"}

            # Analyze for language indicators
            analysis = self._analyze_language_indicators(all_text)

            # Make decision
            detected_language = (
                "es" if analysis["spanish_score"] >= self.confidence_threshold else "en"
            )
            confidence = self._calculate_confidence(
                analysis["spanish_score"], analysis["total_indicators"]
            )

            # Prepare detailed results
            details = {
                "detected_language": detected_language,
                "confidence": confidence,
                "spanish_score": analysis["spanish_score"],
                "total_indicators": analysis["total_indicators"],
                "found_indicators": analysis["found_indicators"],
                "text_sample": (
                    all_text[:300] + "..." if len(all_text) > 300 else all_text
                ),
                "analysis_method": "PassportLanguageDetector",
            }

            self.logger.info(
                f"Passport language detected: {detected_language} (confidence: {confidence})"
            )
            return detected_language, details

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "en", {"error": str(e), "confidence": "low"}

    @classmethod
    def detect_passport_language(
        cls, passport_file, ocr_instance=None, confidence_threshold=3
    ):
        """
        Class method for detecting passport language.

        Args:
            passport_file: File object, BytesIO, or image array
            ocr_instance: Optional pre-initialized OCR instance
            confidence_threshold: Minimum score for Spanish detection

        Returns:
            Tuple[str, Dict]: (detected_language, analysis_details)
        """
        detector = cls(confidence_threshold)
        return detector.detect_with_details(passport_file, ocr_instance)


# Convenience functions for easy import and use
def detect_passport_language(passport_file, ocr_instance=None):
    """
    Convenience function for detecting passport language.

    Args:
        passport_file: File object, BytesIO, or image array
        ocr_instance: Optional pre-initialized OCR instance

    Returns:
        str: 'es' for Spanish, 'en' for English
    """
    detected_language, _ = PassportLanguageDetector.detect_with_details(
        passport_file, ocr_instance
    )
    return detected_language


def get_passport_language_details(passport_file, ocr_instance=None):
    """
    Get detailed passport language detection information.

    Args:
        passport_file: File object, BytesIO, or image array
        ocr_instance: Optional pre-initialized OCR instance

    Returns:
        Dict: Detailed detection information including confidence and indicators
    """
    _, details = PassportLanguageDetector.detect_with_details(
        passport_file, ocr_instance
    )
    return details
