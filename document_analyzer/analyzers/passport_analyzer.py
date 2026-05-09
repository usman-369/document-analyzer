import cv2
import time
import numpy as np

from ..config import DocumentAnalyzerLoggerAdapter, logger
from ..utils import (
    ensure_bytesio,
    preprocess_image,
    extract_mrz_data,
    extract_place_of_birth,
    extract_data_with_boxes,
    detect_passport_language,
)

# Start and end messages for logging
START_MSG = "======= PassportAnalyzer Started ======="
END_MSG = "======= PassportAnalyzer Ended ======="
ERROR_END_MSG = "======= PassportAnalyzer Ended With Error ======="


class PassportAnalyzer:
    """A comprehensive analyzer for passport documents.

    This class provides complete functionality for analyzing passports including
    OCR text extraction, MRZ parsing, and field extraction.

    The analyzer can process various image formats and extract key information:
    - Personal details (dates, places, nationality)
    - Document identifiers (passport numbers, expiry dates)
    - MRZ data parsing

    Attributes:
        user_email (str): Optional user email for logging context.
        logger (DocumentAnalyzerLoggerAdapter): Custom logger with user context.
        passport_np (np.ndarray): Image data as OpenCV-compatible numpy array.
        ocr (PaddleOCR): OCR engine instance configured for detected language.

    Examples:
        >>> analyzer = PassportAnalyzer("passport_image.jpg", "user@example.com")
        >>> results = analyzer.analyze_passport()
        >>> print(results['passport_info']['id_number'])
    """

    def __init__(
        self,
        passport_file,
        user_email=None,
        ocr_instance=None,
        lang_detector_instance=None,
        normalize_input=True,
        preprocess_image=True,
    ):
        """Initialize the PassportAnalyzer with an image file.

        Args:
            passport_file: Input passport image in various formats:
                - File path (str)
                - File-like object (Django upload, etc.)
                - BytesIO object
            user_email (str, optional): User email for logging context.
            ocr_instance (PaddleOCR, optional): Pre-initialized PaddleOCR instance.
                If not provided, language will be detected and appropriate model used.
            lang_detector_instance: Language detector instance for passport language detection.
            normalize_input (bool): Whether to normalize input.
            preprocess_image (bool): Whether to preprocess image.

        Raises:
            ValueError: If image cannot be decoded or is corrupted.
            IOError: If file path cannot be read.
        """
        self.start_time = time.time()

        # Custom logger adapter
        self.logger = DocumentAnalyzerLoggerAdapter(logger, {"user_email": user_email})

        try:
            # Convert to BytesIO (if not already)
            passport_stream = ensure_bytesio(passport_file)
            # Read image bytes into OpenCV-compatible format (BGR)
            self.passport_np = cv2.imdecode(
                np.frombuffer(passport_stream.read(), np.uint8), cv2.IMREAD_COLOR
            )

            if self.passport_np is None:
                raise ValueError(
                    "Could not decode image - invalid format or corrupted file"
                )
        except Exception as e:
            self.logger.error(f"Failed to load input file: {e}")
            raise

        # Use provided OCR instance or detect language and use appropriate model
        if ocr_instance is not None:
            # Explicit OCR instance takes priority
            self.ocr = ocr_instance
            self.logger.debug("Using provided PaddleOCR instance for passport analysis")
        else:
            # Always run language detection (except if explicit OCR provided)
            passport_file.seek(0)
            detected_lang = detect_passport_language(
                passport_file, ocr_instance=lang_detector_instance, logger=self.logger
            )
            self.logger.info(f"Detected passport language: '{detected_lang}'")

            from ..services.paddleocr_service import PaddleOCRService

            self.ocr = PaddleOCRService.get_instance(detected_lang)
            self.logger.debug(
                f"Using PaddleOCR instance for language: '{detected_lang}'"
            )

    def parse_passport_information(self, extracted_data):
        """Parse required passport fields from OCR extracted data.

        Extracts and parses specific information fields from the OCR text data
        including MRZ data (dates, nationality, passport number) and place of birth
        using pattern matching and contextual analysis.

        Args:
            extracted_data (list): List of text data dictionaries from OCR
                extraction, each containing text, bbox, confidence, and position information.

        Returns:
            dict: Dictionary containing parsed passport information with keys:
                - date_of_birth (str): Birth date in DD-MMM-YYYY format
                - place_of_birth (str): Place of birth
                - nationality (str): Nationality (3-letter code)
                - expiry_date (str): Document expiry date in DD-MMM-YYYY format
                - passport_number (str): Passport number

        Note:
            - Dates are converted from MRZ format (YYMMDD) to DD-MMM-YYYY
            - MRZ data is parsed from the machine-readable zone at bottom of passport
            - Place of birth is extracted from text fields using indicators

        Examples:
            >>> extracted = analyzer.extract_data_with_boxes(image)
            >>> info = analyzer.parse_passport_information(extracted)
            >>> print(info['passport_number'])  # e.g., "AB1234567"
            >>> print(info['date_of_birth'])    # e.g., "15-MAR-1985"
        """
        self.logger.debug("Starting passport information parsing")

        # Extract MRZ data (passport number, nationality, dates)
        mrz_data = extract_mrz_data(extracted_data, logger=self.logger)

        # Extract place of birth from text fields
        place_of_birth = extract_place_of_birth(extracted_data, logger=self.logger)

        passport_info = {
            "date_of_birth": mrz_data.get("date_of_birth", ""),
            "place_of_birth": place_of_birth,
            "nationality": mrz_data.get("nationality", ""),
            "expiry_date": mrz_data.get("expiry_date", ""),
            "passport_number": mrz_data.get("passport_number", ""),
        }

        self.logger.debug(f"Parsed passport info: {passport_info}")

        return passport_info

    def analyze_passport(self):
        """Main function to analyze a passport image.

        Orchestrates the complete analysis pipeline including image preprocessing,
        OCR text extraction, MRZ parsing, information extraction, and result compilation.
        This is the primary entry point for passport analysis.

        Returns:
            dict: Complete analysis results containing:
                - success (str): Analysis status - "passport_info" or "none"
                - passport_info (dict): Parsed document information
                - signature (None): Always None for passports
                - raw_extracted_data (list): OCR results for debugging
                - error (str): Error message if analysis fails

        Note:
            Success status indicates what information was successfully extracted:
                - "passport_info": All or most required fields extracted
                - "none": Could not extract sufficient information

        Raises:
            Exception: Caught internally and returned in error field of result dict.

        Examples:
            >>> analyzer = PassportAnalyzer("passport.jpg")
            >>> result = analyzer.analyze_passport()
            >>> if result['success'] == 'passport_info':
            ...     print("Analysis successful")
            ...     print(f"Passport: {result['passport_info']['passport_number']}")
            ...     print(f"DOB: {result['passport_info']['dob']}")
            ...     print(f"POB: {result['passport_info']['pob']}")
        """
        try:
            self.logger.info(START_MSG)

            # Load image if path is provided
            image_path_or_array = self.passport_np
            if isinstance(image_path_or_array, str):
                self.logger.debug(f"Loading image from path: '{image_path_or_array}'")
                image = cv2.imread(image_path_or_array)
                if image is None:
                    self.logger.error(
                        f"Couldn't load image from '{image_path_or_array}'"
                    )
                    raise ValueError(
                        f"Couldn't load image from '{image_path_or_array}'"
                    )
            else:
                self.logger.debug("Using provided image array")
                image = image_path_or_array.copy()

            # Preprocess the image
            processed_image = preprocess_image(image, logger=self.logger)

            # Extract data with bounding boxes
            extracted_data = extract_data_with_boxes(
                processed_image, ocr=self.ocr, logger=self.logger
            )

            if not extracted_data:
                self.logger.warning("Couldn't extract data from the passport image")
                return {
                    "success": "none",
                    "passport_info": {},
                    "signature": None,
                    "raw_extracted_data": [],
                }

            self.logger.info(f"Extracted {len(extracted_data)} data boxes")

            # Parse passport information
            raw_passport_info = self.parse_passport_information(extracted_data)

            # Convert to desired field names (matching cedula format)
            passport_info = {
                "type": "passport",
                "dob": raw_passport_info.get("date_of_birth", ""),
                "pob": raw_passport_info.get("place_of_birth", ""),
                "nationality": raw_passport_info.get("nationality", ""),
                "expiry": raw_passport_info.get("expiry_date", ""),
                "id_number": raw_passport_info.get("passport_number", ""),
            }

            success_status = "passport_info"

            self.logger.info(
                f"Success: '{success_status.capitalize()}' | Date of Birth: '{passport_info['dob']}' "
                f"| Place of Birth: '{passport_info['pob']}' | Nationality: '{passport_info['nationality']}' "
                f"| Expiry: '{passport_info['expiry']}' | Passport Number: '{passport_info['id_number']}'"
            )

            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.logger.info(f"Passport analysis took: {elapsed_time:.2f} seconds")
            self.logger.info(END_MSG)

            return {
                "success": success_status,
                "passport_info": passport_info,
                "signature": None,
                "raw_extracted_data": extracted_data,  # For debugging
            }

        except Exception as e:
            self.logger.error(f"Error in PassportAnalyzer: {str(e)}")
            self.logger.info(ERROR_END_MSG)
            return {
                "success": "none",
                "passport_info": {},
                "signature": None,
                "raw_extracted_data": [],
            }


# Convenience function for easy import and use
def analyze_passport(
    passport_file, user_email=None, ocr_instance=None, lang_detector_instance=True
):
    """Convenience function for passport analysis using PassportAnalyzer.

    Args:
        passport_file: Input passport image.
        user_email (str, optional): User email for logging.
        ocr_instance (PaddleOCR, optional): Pre-initialized OCR instance.
        lang_detector_instance: Language detector instance.

    Returns:
        dict: Analysis results.
    """
    analyzer = PassportAnalyzer(
        passport_file, user_email, ocr_instance, lang_detector_instance
    )
    return analyzer.analyze_passport()
