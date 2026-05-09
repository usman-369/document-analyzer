import cv2
import numpy as np
from io import BytesIO

from ..config import logger as default_logger

# =============================================================================
# FILE HANDLING UTILITY
# =============================================================================


def ensure_bytesio(file):
    """Convert various file types to BytesIO for consistent handling.

    Args:
        file: Input file in various formats - can be BytesIO, file path (str),
            or file-like object (Django uploads, etc.).

    Returns:
        BytesIO: File content as BytesIO object with position reset to 0.

    Raises:
        IOError: If file path cannot be read.
        AttributeError: If file-like object doesn't have read() method.

    Examples:
        >>> with open("image.jpg", "rb") as f:
        ...     bio = ensure_bytesio(f)
        >>> bio = ensure_bytesio("/path/to/image.jpg")
        >>> bio = ensure_bytesio(existing_bytesio_object)
    """
    if isinstance(file, BytesIO):
        file.seek(0)
        return file
    elif isinstance(file, str):
        # Handle file path
        with open(file, "rb") as f:
            return BytesIO(f.read())
    else:
        # Handle file-like objects (Django uploads, etc.)
        return BytesIO(file.read())


# =============================================================================
# IMAGE PREPROCESSING UTILITY
# =============================================================================


def preprocess_image(image, logger=None):
    """Preprocess the cédula image for better OCR results.

    Applies various image enhancement techniques including brightness adjustment,
    sharpness enhancement, and noise reduction based on image characteristics.

    Args:
        image (np.ndarray): Input image in BGR format from OpenCV.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        np.ndarray: Preprocessed image in BGR format ready for OCR processing.

    Note:
        The function automatically determines which enhancements to apply based
        on image statistics like brightness and sharpness variance.

    Examples:
        >>> import cv2
        >>> image = cv2.imread("cedula.jpg")
        >>> processed = preprocess_image(image)
        >>> # Image is now enhanced for better OCR results
    """
    if logger is None:
        logger = default_logger

    logger.debug("Starting image preprocessing")

    # Convert to grayscale (if needed)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Brightness adjustment (if needed)
    mean_brightness = np.mean(gray)
    logger.debug(f"Image mean brightness: {mean_brightness:.2f}")

    if mean_brightness < 80:
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=20)
        logger.debug("Applied brightness enhancement for dark image")
    elif mean_brightness > 200:
        gray = cv2.convertScaleAbs(gray, alpha=0.9, beta=-10)
        logger.debug("Applied brightness reduction for bright image")

    # Minimal sharpness adjustment (if needed)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.debug(f"Image sharpness variance: {laplacian_var:.2f}")

    if laplacian_var < 50:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        logger.debug("Applied sharpening filter for blurry image")

    # Very light noise reduction (if needed)
    if mean_brightness < 100 or laplacian_var > 2000:
        processed = cv2.bilateralFilter(gray, 5, 50, 50)
        logger.debug("Applied noise reduction")
    else:
        processed = gray

    # Convert back to BGR for OCR
    processed_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    logger.debug("Image preprocessing completed")

    return processed_image


# =============================================================================
# OCR DATA EXTRACTION UTILITIES
# =============================================================================


def create_text_data(bbox, text, confidence):
    """Create standardized text data structure from OCR results.

    Processes bounding box coordinates to calculate center point, dimensions,
    and aspect ratio for text analysis and signature detection.

    Args:
        bbox (list): List of 4 coordinate pairs [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            representing the bounding box corners.
        text (str): Extracted text content from OCR.
        confidence (float): OCR confidence score between 0.0 and 1.0.

    Returns:
        dict or None: Standardized text data structure containing:
            - bbox: Original bounding box coordinates
            - text: Cleaned text content (stripped)
            - confidence: OCR confidence score
            - center_x, center_y: Calculated center point coordinates
            - width, height: Bounding box dimensions
            - aspect_ratio: Width/height ratio
            Returns None if bbox is invalid.

    Examples:
        >>> bbox = [[10, 20], [100, 20], [100, 40], [10, 40]]
        >>> data = create_text_data(bbox, "Sample Text", 0.95)
        >>> print(data['center_x'])  # 55.0
        >>> print(data['aspect_ratio'])  # 4.5
    """
    if not bbox or len(bbox) < 4:
        return None

    # Calculate center point and dimensions
    x_coords = [point[0] for point in bbox[:4]]
    y_coords = [point[1] for point in bbox[:4]]
    center_x = sum(x_coords) / 4
    center_y = sum(y_coords) / 4

    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    return {
        "bbox": bbox,
        "text": text.strip(),
        "confidence": confidence,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height > 0 else 0,
    }


def extract_data_with_boxes(image, ocr, logger=None):
    """Extract text data with bounding boxes using PaddleOCR.

    Performs OCR on the preprocessed image and returns structured text data
    with bounding box information for each detected text element.

    Args:
        image (np.ndarray): Preprocessed image in BGR format ready for OCR.
        ocr: PaddleOCR instance configured for text extraction.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        list: List of dictionaries containing text data structures. Each dict
            contains bbox, text, confidence, center coordinates, dimensions,
            and aspect ratio. Empty list if OCR fails.

    Raises:
        Exception: Logs OCR extraction errors but returns empty list instead
            of raising to maintain graceful error handling.

    Examples:
        >>> from paddleocr import PaddleOCR
        >>> ocr = PaddleOCR(lang="es")
        >>> extracted = extract_data_with_boxes(processed_image, ocr)
        >>> for item in extracted:
        ...     print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
    """
    if logger is None:
        logger = default_logger

    try:
        logger.debug("Starting OCR data extraction")

        results = ocr.predict(image)
        result = results[0]
        extracted_data = []

        texts = result["rec_texts"]
        scores = result["rec_scores"]
        polys = result["rec_polys"]

        for _, (text, score, poly) in enumerate(zip(texts, scores, polys)):
            if text and text.strip():
                # Convert poly to the expected bbox format
                bbox = poly.tolist() if hasattr(poly, "tolist") else poly
                extracted_data.append(create_text_data(bbox, text, score))

        logger.debug(f"OCR extracted {len(extracted_data)} text elements")

        return extracted_data

    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return []
