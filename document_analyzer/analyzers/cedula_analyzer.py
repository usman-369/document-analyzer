import re
import cv2
import time
import base64
import numpy as np

from ..config import DocumentAnalyzerLoggerAdapter, logger
from ..utils import (
    ensure_bytesio,
    preprocess_image,
    identify_signature_box,
    extract_signature_image,
    extract_data_with_boxes,
    convert_spanish_date_to_english,
)

# Start and end messages for logging
START_MSG = "======= CedulaAnalyzer Started ======="
END_MSG = "======= CedulaAnalyzer Ended ======="
ERROR_END_MSG = "======= CedulaAnalyzer Ended With Error ======="


class CedulaAnalyzer:
    """A comprehensive analyzer for Panamanian cédula documents.

    This class provides complete functionality for analyzing Panamanian identity
    cards (cédulas) including OCR text extraction, field parsing, and signature
    detection and extraction.

    The analyzer can process various image formats and extract key information:
    - Personal details (dates, places, nationality)
    - Document identifiers (ID numbers, expiry dates)
    - Handwritten signatures (detection and image extraction)

    Attributes:
        user_email (str): Optional user email for logging context.
        logger (DocumentAnalyzerLoggerAdapter): Custom logger with user context.
        cedula_stream (BytesIO): Image data in BytesIO format.
        cedula_np (np.ndarray): Image data as OpenCV-compatible numpy array.
        ocr (PaddleOCR): OCR engine instance configured for Spanish text.

    Examples:
        >>> analyzer = CedulaAnalyzer("cedula_image.jpg", "user@example.com")
        >>> results = analyzer.analyze_cedula()
        >>> print(results['cedula_info']['id_number'])
        >>> # Using with file-like object
        >>> with open("cedula.jpg", "rb") as f:
        ...     analyzer = CedulaAnalyzer(f)
        ...     results = analyzer.analyze_cedula()
    """

    def __init__(
        self,
        cedula_file,
        user_email=None,
        ocr_instance=None,
        normalize_input=True,
        preprocess_image=True,
    ):
        """Initialize the CedulaAnalyzer with an image file.

        Args:
            cedula_file: Input cédula image in various formats:
                - File path (str)
                - File-like object (Django upload, etc.)
                - BytesIO object
            user_email (str, optional): User email for logging context.
                Helps track analysis requests in logs.
            ocr_instance (PaddleOCR, optional): Pre-initialized PaddleOCR instance.
                If not provided, will use default Spanish 'es' model and if not
                initialized, will create a new instance.

        Raises:
            ValueError: If image cannot be decoded or is corrupted.
            IOError: If file path cannot be read.
            Exception: For other initialization errors (logged and re-raised).

        Note:
            The image is automatically converted to OpenCV BGR format for
            consistent processing regardless of input format.
        """
        self.start_time = time.time()

        # Custom logger adapter
        self.logger = DocumentAnalyzerLoggerAdapter(logger, {"user_email": user_email})

        try:
            # Convert to BytesIO (if not already)
            cedula_stream = ensure_bytesio(cedula_file)
            # Read image bytes into OpenCV-compatible format (BGR)
            self.cedula_np = cv2.imdecode(
                np.frombuffer(cedula_stream.read(), np.uint8), cv2.IMREAD_COLOR
            )

            if self.cedula_np is None:
                raise ValueError(
                    "Could not decode image - invalid format or corrupted file"
                )
        except Exception as e:
            self.logger.error(f"Failed to load input file: {e}")
            raise

        # Use provided OCR instance or get default
        if ocr_instance is not None:
            self.ocr = ocr_instance
            self.logger.debug("Using provided PaddleOCR instance for cedula analysis")
        else:
            from ..services.paddleocr_service import PaddleOCRService

            self.ocr = PaddleOCRService.get_instance("es")
            self.logger.debug(
                "Using default Spanish 'es' PaddleOCR instance for cedula analysis"
            )

    def parse_cedula_information(self, extracted_data):
        """Parse required cédula fields from OCR extracted data.

        Extracts and parses specific information fields from the OCR text data
        including dates, places, nationality, and identification numbers using
        pattern matching and contextual analysis.

        Args:
            extracted_data (list): List of text data dictionaries from OCR
            extraction, each containing text, bbox,
            confidence, and position information.

        Returns:
            dict: Dictionary containing parsed cédula information with keys:
                - fecha_nacimiento (str): Birth date in DD-MMM-YYYY format
                - lugar_nacimiento (str): Place of birth
                - nacionalidad (str): Nationality
                - fecha_expiracion (str): Document expiry date
                - cedula_number (str): ID number in X-XXX-XXXX format

        Note:
            - Dates are kept in original Spanish format (e.g., "AGO" for August)
            - Text patterns are matched case-insensitively
            - Contextual clues help disambiguate similar patterns
            - Empty strings returned for fields that cannot be found

        Examples:
            >>> extracted = analyzer.extract_data_with_boxes(image)
            >>> info = analyzer.parse_cedula_information(extracted)
            >>> print(info['cedula_number'])  # e.g., "1-234-5678"
            >>> print(info['fecha_nacimiento'])  # e.g., "14-AGO-1947"
        """
        self.logger.debug("Starting cedula information parsing")

        cedula_info = {
            "fecha_nacimiento": "",  # Date of birth
            "lugar_nacimiento": "",  # Place of birth
            "nacionalidad": "",  # Nationality
            "fecha_expiracion": "",  # Expiry date
            "cedula_number": "",  # ID number
        }

        for item in extracted_data:
            text = item["text"].strip().upper()
            original_text = item["text"].strip()

            # Skip very short text
            if len(text) < 2:
                continue

            # Parse cédula number (flexible format: [A-Z or digits]-[digits]-[digits])
            cedula_match = re.search(r"([A-Z]+|\d+)-(\d+)-(\d+)", original_text.upper())
            if cedula_match and not cedula_info["cedula_number"]:
                cedula_info["cedula_number"] = (
                    f"{cedula_match.group(1)}-{cedula_match.group(2)}-{cedula_match.group(3)}"
                )
                self.logger.info(
                    f"Found cedula number: '{cedula_info['cedula_number']}'"
                )

            # Parse dates (format: DD-MMM-YYYY, e.g., 14-AGO-1947)
            date_match = re.search(r"(\d{2}-\w{3}-\d{4})", original_text)
            if date_match:
                date_str = date_match.group(1)

                # Determine which date it is based on the context
                if "NACIMIENTO" in text:
                    cedula_info["fecha_nacimiento"] = date_str
                    self.logger.info(f"Found birth date: '{date_str}'")
                elif "EXPIRA" in text:
                    cedula_info["fecha_expiracion"] = date_str
                    self.logger.info(f"Found expiry date: '{date_str}'")
                else:
                    # If we don't have birth date yet, assume this is the birth date
                    # Otherwise, assume it's expiry date
                    if not cedula_info["fecha_nacimiento"]:
                        cedula_info["fecha_nacimiento"] = date_str
                        self.logger.info(f"Assumed birth date: '{date_str}'")
                    elif not cedula_info["fecha_expiracion"]:
                        cedula_info["fecha_expiracion"] = date_str
                        self.logger.info(f"Assumed expiry date: '{date_str}'")

            # Parse place of birth (look for "LUGAR DE NACIMIENTO:" pattern)
            if "LUGAR DE NACIMIENTO:" in text and not cedula_info["lugar_nacimiento"]:
                place = original_text.replace("LUGAR DE NACIMIENTO:", "").strip()
                if place:
                    cedula_info["lugar_nacimiento"] = place
                    self.logger.info(f"Found place of birth: '{place}'")
            elif (
                text.startswith("LUGAR DE NACIMIENTO:")
                and not cedula_info["lugar_nacimiento"]
            ):
                # Handle cases where the label might be at the start
                place = text.replace("LUGAR DE NACIMIENTO:", "").strip()
                if place:
                    cedula_info["lugar_nacimiento"] = place
                    self.logger.info(f"Found place of birth: '{place}'")

            # Parse nationality (look for "NACIONALIDAD:" pattern)
            if "NACIONALIDAD:" in text and not cedula_info["nacionalidad"]:
                nationality = original_text.replace("NACIONALIDAD:", "").strip()
                if nationality:
                    cedula_info["nacionalidad"] = nationality
                    self.logger.info(f"Found nationality: '{nationality}'")
            elif text.startswith("NACIONALIDAD:") and not cedula_info["nacionalidad"]:
                # Handle cases where the lable might be at the start
                nationality = text.replace("NACIONALIDAD:", "").strip()
                if nationality:
                    cedula_info["nacionalidad"] = nationality
                    self.logger.info(f"Found nationality: '{nationality}'")

        self.logger.debug(f"Parsed cedula info: {cedula_info}")

        return cedula_info

    def analyze_cedula(self):
        """Main function to analyze a cédula image.

        Orchestrates the complete analysis pipeline including image preprocessing,
        OCR text extraction, information parsing, signature detection and extraction,
        and result compilation. This is the primary entry point for cédula analysis.

        Returns:
            dict: Complete analysis results containing:
                - success (str): Analysis status - "both", "cedula_info",
                  "signature", or "none"
                - cedula_info (dict): Parsed document information with English dates
                - signature (str): Base64-encoded signature image or None
                - raw_extracted_data (list): OCR results for debugging
                - error (str): Error message if analysis fails

        Note:
            Success status indicates what information was successfully extracted:
                - "both": All required fields AND signature extracted
                - "cedula_info": All required fields but no signature
                - "signature": Signature found but missing required fields
                - "none": Neither complete info nor signature extracted

        Raises:
            Exception: Caught internally and returned in error field of result dict.

        Examples:
            >>> analyzer = CedulaAnalyzer("cedula.jpg")
            >>> result = analyzer.analyze_cedula()
            >>> if result['success'] == 'both':
            ...     print("Complete analysis successful")
            ...     print(f"ID: {result['cedula_info']['id_number']}")
            ...     print(f"DOB: {result['cedula_info']['dob']}")
            ...     # Save signature
            ...     with open("signature.png", "wb") as f:
            ...         f.write(base64.b64decode(result['signature']))
        """
        try:
            self.logger.info(START_MSG)

            # Load image if path is provided
            image_path_or_array = self.cedula_np
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
                self.logger.warning("Couldn't extract data from the cedula image")
                return {
                    "success": "none",
                    "cedula_info": {},
                    "signature": None,
                    "raw_extracted_data": [],
                }

            self.logger.info(f"Extracted {len(extracted_data)} data boxes")

            # Parse cédula information
            raw_cedula_info = self.parse_cedula_information(extracted_data)

            # Convert to desired field names
            cedula_info = {
                "type": "cedula",
                "dob": convert_spanish_date_to_english(
                    raw_cedula_info.get("fecha_nacimiento", "")
                ),
                "pob": raw_cedula_info.get("lugar_nacimiento", ""),
                "nationality": raw_cedula_info.get("nacionalidad", ""),
                "expiry": convert_spanish_date_to_english(
                    raw_cedula_info.get("fecha_expiracion", "")
                ),
                "id_number": raw_cedula_info.get("cedula_number", ""),
            }

            # Identify and extract signature using EXPIRA-based method
            signature_box = identify_signature_box(
                extracted_data, image.shape, logger=self.logger
            )
            signature_base64 = None

            if signature_box:
                self.logger.info(f"Signature box identified")
                signature_image = extract_signature_image(
                    image, signature_box, logger=self.logger
                )
                if signature_image is not None:
                    # Convert signature image to base64
                    try:
                        _, buffer = cv2.imencode(".png", signature_image)
                        signature_base64 = base64.b64encode(buffer).decode("utf-8")
                        self.logger.debug("Signature converted to base64")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to convert signature to base64: {str(e)}"
                        )
                        signature_base64 = None
                else:
                    self.logger.warning("Failed to extract signature image")
            else:
                self.logger.warning("No signature box identified")

            # Determine success status
            has_all_cedula_fields = all(
                [
                    cedula_info["dob"],
                    cedula_info["pob"],
                    cedula_info["expiry"],
                    cedula_info["id_number"],
                ]
            )
            has_signature = signature_base64 is not None

            if has_all_cedula_fields and has_signature:
                success_status = "both"
            elif has_all_cedula_fields:
                success_status = "cedula_info"
            elif has_signature:
                success_status = "signature"
            else:
                success_status = "none"

            self.logger.info(
                f"Success: '{success_status.capitalize()}' | Date of Birth: '{cedula_info['dob']}' "
                f"| Place of Birth: '{cedula_info['pob']}' | Nationality: '{cedula_info['nationality']}' "
                f"| Expiry: '{cedula_info['expiry']}' | ID Number: '{cedula_info['id_number']}' "
                f"| Signature: '{signature_base64}'"
            )

            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.logger.info(f"Cedula analysis took: {elapsed_time:.2f} seconds")
            self.logger.info(END_MSG)

            return {
                "success": success_status,
                "cedula_info": cedula_info,
                "signature": signature_base64,
                "raw_extracted_data": extracted_data,  # For debugging
            }

        except Exception as e:
            self.logger.error(f"Error in CedulaAnalyzer: {str(e)}")
            self.logger.info(ERROR_END_MSG)
            return {
                "success": "none",
                "cedula_info": {},
                "signature": None,
                "raw_extracted_data": [],
            }


# Convenience function for easy import and use
def analyze_cedula(cedula_file, user_email=None, ocr_instance=None):
    """Convenience function for cedula analysis using CedulaAnalyzer.

    Args:
        cedula_file: Input cedula image.
        user_email (str, optional): User email for logging.
        ocr_instance (PaddleOCR, optional): PaddleOCR instance.

    Returns:
        dict: Analysis results.
    """
    analyzer = CedulaAnalyzer(cedula_file, user_email, ocr_instance)
    return analyzer.analyze_cedula()
