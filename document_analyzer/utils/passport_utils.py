import re

from ..config import FORBIDDEN_TERMS, BIRTH_PLACE_INDICATORS, ENGLISH_MONTHS


def clean_passport_number(raw_number):
    """
    Clean passport number by fixing OCR mistakes.
    - Replace 0 with O when it should be a letter.
    - Remove invalid characters.
    """
    number = raw_number.strip().replace("<", "")

    # Replace 0 with O if surrounded by letters (common mistake)
    number = re.sub(r"([A-Z])0([A-Z])", r"\1O\2", number)
    number = re.sub(r"([A-Z])0", r"\1O", number)
    number = re.sub(r"0([A-Z])", r"O\1", number)

    # Keep only alphanumeric
    number = re.sub(r"[^A-Z0-9]", "", number)

    return number


def parse_mrz_date(date_str, logger=None):
    """Parse MRZ date format (YYMMDD) to DD-MMM-YYYY format."""
    if len(date_str) != 6:
        return ""

    try:
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        # Fix: for expiry_date, YY >= 30 should still map to 2000+
        # Assume all passport dates are between 1950–2099
        if year >= 50:
            year += 1900
        else:
            year += 2000

        if 1 <= month <= 12:
            return f"{day:02d}-{ENGLISH_MONTHS[month]}-{year}"

    except (ValueError, IndexError):
        if logger:
            logger.warning(f"Invalid MRZ date format: {date_str}")

    return ""


def parse_mrz_lines(mrz_lines, logger=None):
    """Parse MRZ lines to extract passport information."""
    passport_info = {
        "date_of_birth": "",
        "nationality": "",
        "expiry_date": "",
        "passport_number": "",
    }

    if len(mrz_lines) < 2:
        if logger:
            logger.warning("Insufficient MRZ lines for parsing")
        return passport_info

    try:
        line2 = mrz_lines[1]

        # Passport Number
        raw_passport_number = line2[0:9]
        passport_info["passport_number"] = clean_passport_number(raw_passport_number)

        # Nationality
        passport_info["nationality"] = line2[10:13]

        # DOB (YYMMDD at index 13–19)
        dob_str = line2[13:19]
        passport_info["date_of_birth"] = parse_mrz_date(dob_str, logger)

        # Expiry Date (YYMMDD at index 21–27)
        expiry_str = line2[21:27]
        passport_info["expiry_date"] = parse_mrz_date(expiry_str, logger)

        if logger:
            logger.debug(f"Parsed MRZ data: {passport_info}")

    except Exception as e:
        if logger:
            logger.error(f"Error parsing MRZ: {str(e)}")

    return passport_info


def aggressive_clean_pob(text):
    """Aggressively clean POB text to remove document field contamination."""
    if not text:
        return ""

    # First basic cleanup
    cleaned = text.strip(" :/.,;-")

    # Remove OCR artifacts
    artifacts = ["<<<", ">>>", "<<", ">>", "||", "|"]
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")

    # AGGRESSIVE: Split by common separators and take only the first meaningful part
    separators = [
        r"\s+m/",
        r"\s+rte",
        r"\s+/Place",
        r"\s+Place\s+of",
        r"\s+Lugar\s+de",
        r"\s+Date\s+of",
        r"\s+Authority",
        r"\s+Fecha\s+de",
        r"\s+SURAT",
    ]

    for separator in separators:
        parts = re.split(separator, cleaned, flags=re.IGNORECASE)
        if len(parts) > 1:
            cleaned = parts[0].strip()
            break

    # Remove any trailing fragments that look like document fields
    unwanted_endings = [
        r"\s+m$",
        r"\s+rt$",
        r"\s+rte$",
        r"\s+/P$",
        r"\s+Pl$",
        r"\s+Place$",
        r"\s+of$",
        r"\s+Issue$",
        r"\s+Auth$",
    ]

    for ending in unwanted_endings:
        cleaned = re.sub(ending, "", cleaned, flags=re.IGNORECASE)

    # Final cleanup
    cleaned = cleaned.strip(" /:-.,")

    return cleaned


def is_clean_place_name(text):
    """Very strict validation for place names."""
    if not text or len(text) < 3:
        return False

    text_upper = text.upper().strip()

    # Reject if contains document field indicators
    for term in FORBIDDEN_TERMS:
        if term in text_upper:
            return False

    # Must be mostly alphabetic (allow spaces, commas, but not too many special chars)
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(" ", "").replace(",", ""))

    if total_chars > 0 and alpha_chars / total_chars < 0.7:  # At least 70% letters
        return False

    # Reasonable length for place names
    if len(text) > 40:
        return False

    return True


def extract_mrz_data(extracted_data, logger=None):
    """Extract and parse MRZ (Machine Readable Zone) data.

    Args:
        extracted_data (list): List of text data from OCR extraction.
        logger: Logger instance for logging.

    Returns:
        dict: Parsed MRZ data containing passport info.
    """
    if logger:
        logger.debug("Starting MRZ data extraction")

    mrz_lines = []

    # Find MRZ lines (typically at bottom, contain mostly uppercase and special chars)
    for item in extracted_data:
        text = item["text"].strip()
        # MRZ lines are typically long, contain < characters, and are mostly uppercase
        if len(text) > 20 and "<" in text and text.isupper():
            mrz_lines.append(text)

    if not mrz_lines:
        if logger:
            logger.warning("No MRZ lines found")
        return {}

    # Sort MRZ lines by vertical position (top to bottom)
    mrz_items = [
        (item, item["text"])
        for item in extracted_data
        if len(item["text"]) > 20 and "<" in item["text"] and item["text"].isupper()
    ]
    mrz_items.sort(key=lambda x: x[0]["center_y"])
    mrz_lines = [item[1] for item in mrz_items]

    if logger:
        logger.debug(f"Found {len(mrz_lines)} MRZ lines")

    return parse_mrz_lines(mrz_lines, logger)


def extract_place_of_birth(extracted_data, logger=None):
    """Extract place of birth from passport OCR data.

    Args:
        extracted_data (list): List of text data from OCR extraction.
        logger: Logger instance for logging.

    Returns:
        str: Extracted place of birth or empty string.
    """
    if logger:
        logger.debug("Searching for place of birth")

    # Method 1: Look for indicators and extract carefully
    for i, item in enumerate(extracted_data):
        text = item["text"].upper()

        for indicator in BIRTH_PLACE_INDICATORS:
            if indicator in text:
                if logger:
                    logger.debug(f"Found birth place indicator: '{indicator}'")

                # Extract from same line (after indicator)
                parts = text.split(indicator)
                if len(parts) > 1 and parts[1].strip():
                    candidate = aggressive_clean_pob(parts[1])
                    if is_clean_place_name(candidate):
                        if logger:
                            logger.debug(f"Found POB on same line: '{candidate}'")
                        return candidate

                # Check next few lines with aggressive cleaning
                for offset in range(1, min(4, len(extracted_data) - i)):
                    if i + offset < len(extracted_data):
                        next_text = extracted_data[i + offset]["text"]
                        candidate = aggressive_clean_pob(next_text)

                        if is_clean_place_name(candidate):
                            if logger:
                                logger.debug(
                                    f"Found POB on line +{offset}: '{candidate}'"
                                )
                            return candidate

    return ""
