import re
import cv2

from ..config import logger as default_logger

# =============================================================================
# SIGNATURE DETECTION UTILITIES
# =============================================================================


def find_expira_block(extracted_data, logger=None):
    """Find the EXPIRA block to use as reference for signature detection.

    Searches through OCR extracted data to locate text containing "EXPIRA"
    which serves as a reference point for signature location on cédula documents.

    Args:
        extracted_data (list): List of text data dictionaries from OCR extraction.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        dict or None: Text data dictionary containing "EXPIRA" text if found,
            None if no EXPIRA block is detected.

    Note:
        The EXPIRA block typically contains the document expiry date and serves
        as a reliable landmark for signature positioning on cédula documents.

    Examples:
        >>> expira_block = find_expira_block(extracted_data)
        >>> if expira_block:
        ...     print(f"Found EXPIRA at y={expira_block['center_y']}")
    """
    if logger is None:
        logger = default_logger

    for _, item in enumerate(extracted_data):
        text = item["text"].upper()
        if "EXPIRA" in text:
            logger.debug(f"Found EXPIRA block: '{text}'")
            return item

    logger.debug("No EXPIRA block found")

    return None


def fallback_signature_detection(extracted_data, image_shape, logger=None):
    """Fallback signature detection when EXPIRA block is not found.

    Implements alternative signature detection by analyzing text boxes in the
    bottom portion of the document where signatures are typically located.

    Args:
        extracted_data (list): List of text data dictionaries from OCR extraction.
        image_shape (tuple): Image dimensions as (height, width, channels).
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        dict or None: Text data dictionary with lowest confidence in bottom area,
            which likely represents handwritten signature text.
            Returns None if no suitable candidates found.

    Note:
        This method assumes signatures appear in the bottom 40% of the document
        and have lower OCR confidence due to handwriting characteristics.

    Examples:
        >>> fallback = fallback_signature_detection(extracted_data, image.shape)
        >>> if fallback:
        ...     print(f"Fallback signature: {fallback['text']}")
    """
    if logger is None:
        logger = default_logger

    logger.debug("Using fallback signature detection")

    height, _ = image_shape[:2]

    # Filter boxes in the bottom area where signature is expected
    bottom_boxes = []
    for box_data in extracted_data:
        # Check if box is in the bottom 40% of the image
        if box_data["center_y"] > height * 0.6:
            bottom_boxes.append(box_data)

    if not bottom_boxes:
        logger.debug("No boxes found in the bottom area for fallback detection")
        return None

    # Return the box with lowest confidence in bottom area
    fallback = min(bottom_boxes, key=lambda x: x["confidence"])
    logger.debug(f"Fallback signature confidence: {fallback['confidence']:.3f})")

    return fallback


def identify_signature_box(extracted_data, image_shape, logger=None):
    """Identify signature based on EXPIRA block position and text characteristics.

    Main signature detection function that uses the EXPIRA block as a reference
    point and applies scoring algorithms to identify the most likely signature
    area among text boxes below the expiry date.

    Args:
        extracted_data (list): List of text data dictionaries from OCR extraction.
        image_shape (tuple): Image dimensions as (height, width, channels).
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        dict or None: Text data dictionary identified as signature with highest
            signature score. Returns None if no suitable signature
            candidate is found.

    Note:
        Scoring algorithm considers multiple factors:
        - Low OCR confidence (handwriting is harder to read)
        - Non-alphanumeric characters (signature flourishes)
        - Wide aspect ratio (signatures span horizontally)
        - Mixed case patterns
        - Low alphabetic ratio
        - Penalties for obvious document text patterns

    Examples:
        >>> signature = identify_signature_box(extracted_data, image.shape)
        >>> if signature:
        ...     print(f"Signature detected: '{signature['text']}'")
        ...     print(f"Confidence: {signature['confidence']:.3f}")
    """
    if logger is None:
        logger = default_logger

    logger.debug("Starting signature box identification")

    if not extracted_data:
        logger.warning("No extracted data available for signature identification")
        return None

    # Find the EXPIRA block
    expira_block = find_expira_block(extracted_data, logger=logger)

    if not expira_block:
        return fallback_signature_detection(extracted_data, image_shape, logger=logger)

    expira_y = expira_block["center_y"]
    expira_bottom = expira_y + (expira_block["height"] / 2)

    # Find all boxes below the EXPIRA block
    below_expira_boxes = []
    for _, box_data in enumerate(extracted_data):
        # Check if box center is below the bottom of EXPIRA block
        if box_data["center_y"] > expira_bottom:
            below_expira_boxes.append(box_data)

    logger.debug(f"Found {len(below_expira_boxes)} box(es) below the EXPIRA block")

    if not below_expira_boxes:
        logger.warning("No box(es) found below the EXPIRA block")
        return None

    # Score each candidate based on the signature characteristics
    signature_candidates = []

    for _, candidate in enumerate(below_expira_boxes):
        text = candidate["text"]
        confidence = candidate["confidence"]

        signature_score = 0
        reasons = []

        # Score 1: Lower confidence is better for signatures (handwriting is harder to OCR)
        if confidence < 0.2:
            signature_score += 6
            reasons.append("very low confidence")
        elif confidence < 0.4:
            signature_score += 4
            reasons.append("low confidence")
        elif confidence < 0.6:
            signature_score += 2
            reasons.append("medium-low confidence")

        # Score 2: Non-alphanumeric characters (signatures often have curves, flourishes)
        special_chars = len([c for c in text if not c.isalnum() and c not in " .,-"])
        if special_chars > 2:
            signature_score += 4
            reasons.append("many special characters")
        elif special_chars > 0:
            signature_score += 2
            reasons.append("some special characters")

        # Score 3: Aspect ratio (signatures are typically wider)
        if candidate["aspect_ratio"] > 4:
            signature_score += 4
            reasons.append("very wide aspect ratio")
        elif candidate["aspect_ratio"] > 2.5:
            signature_score += 2
            reasons.append("wide aspect ratio")
        elif candidate["aspect_ratio"] > 1.5:
            signature_score += 1
            reasons.append("moderately wide")

        # Score 4: Mixed case or unusual patterns
        if any(c.islower() for c in text) and any(c.isupper() for c in text):
            signature_score += 3
            reasons.append("mixed case")

        # Score 5: Low alphabetic ratio (signatures may have unclear characters)
        alpha_ratio = len([c for c in text if c.isalpha()]) / len(text) if text else 0
        if alpha_ratio < 0.5 and len(text) > 2:
            signature_score += 3
            reasons.append("low alpha ratio")
        elif alpha_ratio < 0.7 and len(text) > 3:
            signature_score += 1
            reasons.append("medium alpha ratio")

        # Score 6: Short text (signatures can be brief or poorly recognized)
        if len(text) <= 3:
            signature_score += 2
            reasons.append("short text")

        # Score 7: Contains handwriting-like patterns
        handwriting_patterns = ["j", "g", "y", "f", "p", "q"]  # Letters with descenders
        if any(letter in text.lower() for letter in handwriting_patterns):
            signature_score += 1
            reasons.append("handwriting patterns")

        # PENALTIES: Heavily penalize obvious non-signature text
        penalty = 0

        # Penalty 1: ID number pattern (flexible format: [A-Z or digits]-[digits]-[digits])
        if re.search(r"([A-Z]+|\d+)-(\d+)-(\d+)", text):
            penalty += 20
            reasons.append("PENALTY: ID number pattern")

        # Penalty 2: Date patterns
        if re.search(r"\d{2}-\w{3}-\d{4}", text.upper()):
            penalty += 15
            reasons.append("PENALTY: date pattern")

        # Penalty 3: Clear document text
        from ..config import DOCUMENT_KEYWORDS_ES

        if any(keyword in text.upper() for keyword in DOCUMENT_KEYWORDS_ES):
            penalty += 15
            reasons.append("PENALTY: document text")

        # Penalty 4: Very high confidence (printed text is usually high confidence)
        if confidence > 0.85:
            penalty += 4
            reasons.append("PENALTY: very high confidence")
        elif confidence > 0.7:
            penalty += 2
            reasons.append("PENALTY: high confidence")

        # Penalty 5: Pure numeric text
        if text.replace("-", "").replace(" ", "").isdigit() and len(text) > 2:
            penalty += 12
            reasons.append("PENALTY: numeric text")

        # Penalty 6: Very long text (signatures are usually short)
        if len(text) > 20:
            penalty += 5
            reasons.append("PENALTY: very long text")

        final_score = signature_score - penalty
        signature_candidates.append((candidate, final_score))

    # Step 4: Select best candidate - Sort by score (highest first)
    signature_candidates.sort(key=lambda x: x[1], reverse=True)

    # Allow slightly negative scores
    if signature_candidates and signature_candidates[0][1] > -5:
        best_candidate = signature_candidates[0][0]
        best_score = signature_candidates[0][1]
        logger.info(
            f"Selected signature candidate: score={best_score} (base={signature_score}, penalty={penalty}) - {', '.join(reasons)}"
        )
        return best_candidate

    # If no good candidate found, return the one with lowest confidence
    if below_expira_boxes:
        fallback = min(below_expira_boxes, key=lambda x: x["confidence"])
        logger.warning(f"No good signature candidate found, using fallback")

        return fallback

    logger.warning("No signature candidate identified")

    return None


# =============================================================================
# SIGNATURE EXTRACTION UTILITIES
# =============================================================================


def process_signature_to_bw(signature_image, logger=None):
    """Enhanced signature processing using adaptive thresholding.

    Processes extracted signature region to enhance visibility of handwritten
    signatures by converting to black and white using adaptive thresholding.

    Args:
        signature_image (np.ndarray): Raw signature region extracted from document.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        np.ndarray: Binary (black and white) signature image with enhanced
            contrast suitable for further processing or display.

    Note:
        The function applies:
        - Grayscale conversion if needed
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Normalization for consistent contrast
        - Adaptive thresholding for binary conversion

    Examples:
        >>> processed_sig = process_signature_to_bw(signature_region)
        >>> cv2.imwrite("signature_bw.png", processed_sig)
    """
    if logger is None:
        logger = default_logger

    logger.debug("Processing signature with adaptive threshold method")

    # Convert to grayscale
    if len(signature_image.shape) == 3:
        gray = cv2.cvtColor(signature_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = signature_image.copy()

    # Enhance contrast more aggressively for faint signatures
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Additional contrast enhancement
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # Apply adaptive threshold
    final_signature = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    logger.debug("Signature processing completed using adaptive threshold")

    return final_signature


def extract_signature_image(image, signature_box, logger=None):
    """Extract and process the signature image from the bounding box.

    Extracts the signature region from the full document image based on the
    identified signature bounding box, applies padding for complete capture,
    and processes the result for optimal signature visibility.

    Args:
        image (np.ndarray): Original cédula document image in BGR format.
        signature_box (dict): Text data dictionary containing signature bbox
            and other properties from signature identification.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        np.ndarray or None: Processed binary signature image ready for use,
            or None if extraction fails or signature_box is invalid.

    Note:
        - Applies generous padding (50% width, 40% height) to capture signature
          elements that may extend beyond OCR detection boundaries
        - Automatically handles image boundary constraints
        - Processes extracted region to black and white for clarity

    Raises:
        None: Function handles errors gracefully and returns None on failure.

    Examples:
        >>> signature_img = extract_signature_image(image, signature_box)
        >>> if signature_img is not None:
        ...     cv2.imwrite("extracted_signature.png", signature_img)
        ...     print("Signature extracted successfully")
    """
    if logger is None:
        logger = default_logger

    logger.debug("Starting signature image extraction")

    if not signature_box:
        logger.warning("No signature box provided for extraction")
        return None

    bbox = signature_box["bbox"]

    # Get bounding box coordinates
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]

    x1, y1 = int(min(x_coords)), int(min(y_coords))
    x2, y2 = int(max(x_coords)), int(max(y_coords))

    # Signatures often extend beyond OCR detection boxes
    padding_x = max(30, int((x2 - x1) * 0.5))  # 50% of width or 30px minimum
    padding_y = max(20, int((y2 - y1) * 0.4))  # 40% of height or 20px minimum

    # Expand the region
    x1_expanded = max(0, x1 - padding_x)
    y1_expanded = max(0, y1 - padding_y)
    x2_expanded = min(image.shape[1], x2 + padding_x)
    y2_expanded = min(image.shape[0], y2 + padding_y)

    logger.debug(
        f"Extracting signature region: ({x1_expanded},{y1_expanded}) to ({x2_expanded},{y2_expanded})"
    )

    # Extract signature region
    signature_region = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

    if signature_region.size == 0:
        logger.error("Empty signature region extracted")
        return None

    # Process signature to black and white
    processed_signature = process_signature_to_bw(signature_region, logger=logger)

    logger.debug("Signature extraction and processing completed")

    return processed_signature
