import re
import cv2
import numpy as np

from ..config import logger as default_logger

# =============================================================================
# VISUALIZATION UTILITY
# =============================================================================


def draw_bounding_boxes(image, extracted_data, signature_box=None, logger=None):
    """Draw bounding boxes on the cedula image for visualization.

    Creates a visualization of the OCR results by drawing bounding boxes around
    detected text areas. Special highlighting is applied to signature areas.

    Args:
        image (np.ndarray): Original cédula image in BGR format.
        extracted_data (list): List of text data dictionaries from OCR extraction.
        signature_box (dict, optional): Specific text data dict identified as
            signature area. Will be highlighted in red.
        logger (logging.Logger, optional): Logger instance for debug messages.
            Defaults to module's default logger.

    Returns:
        np.ndarray: Copy of input image with bounding boxes and labels drawn.
            Regular text boxes are green, signature box is red.

    Note:
        - Text labels are truncated to prevent overcrowding
        - Background rectangles ensure label readability
        - Label positioning adapts to avoid image boundaries

    Examples:
        >>> vis_image = draw_bounding_boxes(image, extracted_data, signature_box)
        >>> cv2.imshow("Visualization", vis_image)
        >>> cv2.waitKey(0)
    """
    if logger is None:
        logger = default_logger

    logger.debug("Drawing bounding boxes for visualization")

    # Create a copy of the image to draw on
    vis_image = image.copy()

    # Draw all text bounding boxes
    for i, item in enumerate(extracted_data):
        bbox = item["bbox"]
        text = item["text"]
        _ = item["confidence"]

        # Convert bbox to integer coordinates
        points = np.array(bbox, dtype=np.int32)

        # Determine color based on whether this is the signature box
        if signature_box and item is signature_box:
            color = (0, 0, 255)  # Red for signature
            thickness = 3
            label = (
                f"SIGNATURE: {text[:20]}..." if len(text) > 20 else f"SIGNATURE: {text}"
            )
        else:
            color = (0, 255, 0)  # Green for regular text
            thickness = 2
            label = f"{i+1}: {text[:15]}..." if len(text) > 15 else f"{i+1}: {text}"

        # Draw the bounding box
        cv2.polylines(vis_image, [points], True, color, thickness)

        # Draw text label above the box
        label_y = int(min([p[1] for p in points])) - 10
        if label_y < 20:
            label_y = int(max([p[1] for p in points])) + 25

        # Add background rectangle for text readability
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            vis_image,
            (int(min([p[0] for p in points])), label_y - text_height - 5),
            (int(min([p[0] for p in points])) + text_width, label_y + 5),
            (255, 255, 255),
            -1,
        )

        # Draw the text
        cv2.putText(
            vis_image,
            label,
            (int(min([p[0] for p in points])), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    logger.debug("Bounding box visualization completed")

    return vis_image


# =============================================================================
# DATE CONVERSION UTILITY
# =============================================================================


def convert_spanish_date_to_english(date_str):
    """Convert Spanish date format to English format.

    Converts date strings from Spanish month abbreviations to English
    equivalents while maintaining the DD-MMM-YYYY format commonly used
    in Panamanian documents.

    Args:
        date_str (str): Date string in Spanish format (e.g., "14-AGO-1947").
            Can also be None or non-string values.

    Returns:
        str: Date string in English format (e.g., "14-AUG-1947").
            Returns original input if conversion is not possible.

    Note:
        - Only processes strings matching DD-MMM-YYYY pattern
        - Case-insensitive matching for month abbreviations
        - Preserves original format if no Spanish months are detected
        - Handles edge cases gracefully by returning original input

    Examples:
        >>> convert_spanish_date_to_english("14-AGO-1947")
        '14-AUG-1947'
        >>> convert_spanish_date_to_english("23-MAR-1940")
        '23-MAR-1940'  # MAR is same in both languages
        >>> convert_spanish_date_to_english("invalid-date")
        'invalid-date'  # Returns original if no match
        >>> convert_spanish_date_to_english(None)
        None  # Handles None input gracefully
    """
    if not date_str or not isinstance(date_str, str):
        return date_str

    date_pattern = r"(\d{1,2})-([A-Z]{3})-(\d{4})"
    match = re.search(date_pattern, date_str.upper())

    if not match:
        return date_str

    from ..config import SPANISH_TO_ENGLISH_MONTHS

    day, spanish_month, year = match.groups()
    english_month = SPANISH_TO_ENGLISH_MONTHS.get(spanish_month, spanish_month)

    return f"{day}-{english_month}-{year}"
