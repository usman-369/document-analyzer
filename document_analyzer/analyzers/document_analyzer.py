import cv2
import time
import numpy as np

from ..utils import ensure_bytesio
from ..config import (
    DocumentAnalyzerLoggerAdapter,
    logger,
    CEDULA_INDICATORS,
    PASSPORT_INDICATORS,
)

# Start and end messages for logging
START_MSG = "======= DocumentAnalyzer Started ======="
END_MSG = "======= DocumentAnalyzer Ended ======="
ERROR_END_MSG = "======= DocumentAnalyzer Ended With Error ======="


class DocumentAnalyzer:
    """
    Unified analyzer for both Cedula and Passport documents with document type detection.
    """

    def __init__(
        self,
        document_file,
        user_email=None,
        ocr_instance=None,
        normalize_input=True,
        preprocess_image=True,
    ):
        """
        Initialize DocumentAnalyzer.

        Args:
            document_file: File object or BytesIO containing the document image
            user_email: Optional user email for logging
            ocr_instance: Optional pre-initialized OCR instance
        """
        self.start_time = time.time()
        self.document_file = document_file
        self.user_email = user_email

        # Custom logger adapter
        self.logger = DocumentAnalyzerLoggerAdapter(
            logger, {"user_email": self.user_email}
        )
        self.logger.info(START_MSG)

        # Load and prepare image for type detection
        document_stream = ensure_bytesio(self.document_file)
        self.document_np = cv2.imdecode(
            np.frombuffer(document_stream.read(), np.uint8), cv2.IMREAD_COLOR
        )

        if self.document_np is None:
            raise ValueError(
                "Could not decode image - invalid format or corrupted file"
            )

        # Use provided OCR instance or get default
        if ocr_instance is not None:
            self.ocr = ocr_instance
            self.logger.debug(
                "Using provided PaddleOCR instance for document detection"
            )
        else:
            from ..services.paddleocr_service import PaddleOCRService

            self.ocr = PaddleOCRService.get_instance("es")
            self.logger.debug(
                "Using default Spanish 'es' PaddleOCR instance for document detection"
            )

    def detect_document_type(self):
        """Detect whether the document is a cedula or passport."""
        self.logger.debug("Starting document type detection")

        try:
            # Get text from image
            results = self.ocr.predict(self.document_np)
            if not results or not results[0]:
                self.logger.warning("No text detected for document type detection")
                return "unknown"

            result = results[0]
            all_text = " ".join([text.upper() for text in result["rec_texts"]]).strip()

            self.logger.debug(f"Detected text sample: '{all_text[:100]}...'")

            # Check for MRZ pattern (strong passport indicator)
            if any("<" in text and len(text) > 20 for text in result["rec_texts"]):
                self.logger.info("MRZ pattern detected - identifying as passport")
                return "passport"

            # Count indicators
            cedula_score = sum(
                1 for indicator in CEDULA_INDICATORS if indicator in all_text
            )
            passport_score = sum(
                1 for indicator in PASSPORT_INDICATORS if indicator in all_text
            )

            self.logger.debug(
                f"Cedula score: {cedula_score}, Passport score: {passport_score}"
            )

            if cedula_score > passport_score and cedula_score > 0:
                self.logger.info("Document identified as cedula")
                return "cedula"
            elif passport_score > cedula_score and passport_score > 0:
                self.logger.info("Document identified as passport")
                return "passport"
            else:
                self.logger.warning("Document type could not be determined confidently")
                return "unknown"

        except Exception as e:
            self.logger.error(f"Error in document type detection: {str(e)}")
            return "unknown"

    def analyze_document(self):
        """Analyze document by detecting type and using appropriate analyzer."""
        try:
            # Detect document type
            doc_type = self.detect_document_type()

            if doc_type == "unknown":
                self.logger.warning("Could not determine document type")
                self.logger.info(ERROR_END_MSG)
                return {
                    "success": "none",
                    "document_info": {},
                    "signature": None,
                    "raw_extracted_data": [],
                }

            # Reset file pointer before further analysis
            self.document_file.seek(0)

            if doc_type == "cedula":
                from .cedula_analyzer import CedulaAnalyzer

                analyzer = CedulaAnalyzer(self.document_file, self.user_email)
                results = analyzer.analyze_cedula()
                results = results.copy()
                results["document_info"] = results.pop("cedula_info")
            elif doc_type == "passport":
                from .passport_analyzer import PassportAnalyzer

                analyzer = PassportAnalyzer(self.document_file, self.user_email)
                results = analyzer.analyze_passport()
                results = results.copy()
                results["document_info"] = results.pop("passport_info")

            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.logger.info(f"Document analysis took: {elapsed_time:.2f} seconds")
            self.logger.info(END_MSG)

            return results

        except Exception as e:
            self.logger.error(f"Error in DocumentAnalyzer: {str(e)}")
            self.logger.info(ERROR_END_MSG)
            return {
                "success": "none",
                "document_info": {},
                "signature": None,
                "raw_extracted_data": [],
            }


# Convenience function for easy import and use
def analyze_document(document_file, user_email=None, ocr_instance=None):
    """Convenience function for document analysis using DocumentAnalyzer.

    Args:
        document_file: Input document image.
        user_email (str, optional): User email for logging.
        ocr_instance (PaddleOCR, optional): PaddleOCR instance.

    Returns:
        dict: Analysis results.
    """
    analyzer = DocumentAnalyzer(document_file, user_email, ocr_instance)
    return analyzer.analyze_document()
