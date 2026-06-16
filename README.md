# Document Analyzer

Document Analyzer is a Python package for extracting structured information from identity documents using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). It supports Panamanian ID cards (Cédulas) in Spanish, and passports with standard ICAO Machine Readable Zones (MRZ) in Spanish or English. The package automatically detects the document type and language, loading the appropriate OCR instance accordingly. It is specifically designed to work with mobile phone photos of documents rather than scans or PDFs, and includes automatic image preprocessing to improve extraction accuracy from lower-quality images.

## Version

| Version | Notes |
|---------|-------|
| 0.1.2 | Fixed missing paddlepaddle dependency |
| 0.1.1 | Python version compatibility fix (>=3.8), added package classifiers and keywords |
| 0.1.0 | Initial release |

## Features

- **Cédula Extraction** — Extract ID number, date of birth, place of birth, expiry date, and handwritten signature detection from Panamanian identity cards
- **Passport Extraction** — Extract ID number, date of birth, place of birth, nationality, and expiry date from passports with standard ICAO Machine Readable Zones (MRZ). Works with any country's passport that follows the ICAO standard format.
- **Automatic Document Detection** — Intelligently detect whether an image contains a Cédula or Passport
- **Image Preprocessing** — Automatically enhance poor quality images before OCR processing
- **CLI Support** — Full command-line interface for document analysis without writing code
- **JSON Output** — Structured JSON results for easy integration into other systems
- **Multi-Language Support** — Cédulas are processed in Spanish only. Passports support automatic language detection between Spanish and English, with the appropriate PaddleOCR instance loaded based on detected language.

## Requirements

- Python 3.8 or higher
- PaddleOCR 3.2.0

## Installation

```bash
pip install document-analyzer
```

## CLI Usage

The package includes a command-line interface accessible via the `document-analyzer` command.

### Basic Usage with Auto-Detection

Analyze a document with automatic type detection:

```bash
document-analyzer analyze photo.jpg
```

The output is printed as JSON to stdout.

### Specify Document Type

If you know the document type, you can skip auto-detection for faster processing:

```bash
document-analyzer analyze cedula.jpg --type cedula
document-analyzer analyze passport.jpg --type passport
```

### Save Output to File

Save analysis results to a JSON file instead of printing to stdout:

```bash
document-analyzer analyze photo.jpg --save result.json
```

### Verbose Mode

Enable debug-level logging to see detailed processing information:

```bash
document-analyzer analyze photo.jpg -v
```

Combine with `--save` for logging while saving results:

```bash
document-analyzer analyze photo.jpg --save result.json -v
```

### Help

View all available options:

```bash
document-analyzer analyze --help
```

## Library Usage

You can use Document Analyzer as a Python library in your own code. Here are examples for the main use cases.

### Auto-Detection with DocumentAnalyzer

```python
from document_analyzer import DocumentAnalyzer

# Initialize with image path
analyzer = DocumentAnalyzer("photo.jpg")

# Detect document type
doc_type = analyzer.detect_document_type()
print(f"Detected: {doc_type}")  # "cedula" or "passport" or "unknown"
```

### Extract from Cédula

```python
from document_analyzer import CedulaAnalyzer

# Initialize with image path
analyzer = CedulaAnalyzer("cedula.jpg")

# Analyze the document
results = analyzer.analyze_cedula()
print(results)

# Optional: provide user email for logging context
analyzer = CedulaAnalyzer("cedula.jpg", user_email="user@example.com")
```

### Extract from Passport

```python
from document_analyzer import PassportAnalyzer

# Initialize with image path
analyzer = PassportAnalyzer("passport.jpg")

# Analyze the document
results = analyzer.analyze_passport()
print(results)

# Optional: provide user email for logging context
analyzer = PassportAnalyzer("passport.jpg", user_email="user@example.com")
```

### Convenience Functions

You can also use high-level functions for simpler code:

```python
from document_analyzer import analyze_document, analyze_cedula, analyze_passport

# Auto-detect and analyze
result = analyze_document("photo.jpg")

# Analyze specific document type
cedula_result = analyze_cedula("cedula.jpg")
passport_result = analyze_passport("passport.jpg")
```

## Output

Analysis results are returned as dictionaries containing structured information about the extracted data. Below are example outputs for both document types with realistic but fictional Panamanian data.

### Cédula Output Example

```json
{
    "success": "both",
    "cedula_info": {
        "type": "cedula",
        "id_number": "8-123-456",
        "dob": "15-May-1990",
        "pob": "Panama",
        "nationality": "Panamanian",
        "expiry": "22-Mar-2030"
    },
    "signature": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

The `success` field can be `"both"` (all info + signature), `"cedula_info"` (all info but no signature), `"signature"` (signature only), or `"none"` (extraction failed).

### Passport Output Example

```json
{
    "success": "passport_info",
    "passport_info": {
        "type": "passport",
        "id_number": "PA123456789",
        "dob": "20-Nov-1988",
        "pob": "Colón",
        "nationality": "PAN",
        "expiry": "10-Sep-2032"
    },
    "signature": null
}
```

The `success` field can be `"passport_info"` (extraction successful) or `"none"` (extraction failed).

## Image Requirements

Document Analyzer is designed to work with mobile phone photos of documents. Here are the technical requirements:

- **Supported Formats** — JPEG, PNG, BMP, TIFF, GIF
- **Orientation** — Portrait orientation works best
- **Quality** — Mobile phone camera quality is acceptable; the package includes automatic preprocessing to handle lower quality images
- **Coverage** — Entire document should be visible in the frame
- **Lighting** — Avoid strong shadows or glare across the document

The package includes automatic image preprocessing that attempts to enhance poor quality images before OCR processing. This can help improve accuracy for images with:

- Low contrast
- Poor lighting conditions
- Motion blur
- Dust or slight damage

**Note on PDFs:** PDF files are not listed in supported formats because they have not been tested. PDFs are not officially supported and may not work as expected. Use image files (JPG, PNG, etc.) for best results.

## GPU Acceleration

PaddleOCR supports GPU acceleration via CUDA for significantly faster processing on NVIDIA GPUs. However, Document Analyzer has only been tested and validated on CPU hardware (Intel i5, 10th generation).

If you want to experiment with GPU acceleration, you will need to:

1. Configure PaddleOCR to use your CUDA-enabled GPU according to the PaddleOCR documentation
2. Ensure your system has CUDA and cuDNN properly configured
3. Test thoroughly in your environment before deploying to production

CPU processing is stable and recommended for production use.

## Logging

Document Analyzer uses Python's standard `logging` module with the logger namespace `document_analyzer`. This allows you to configure logging behavior in your own applications.

### Basic Configuration

```python
import logging

# Enable debug logging from document_analyzer
logging.basicConfig(level=logging.DEBUG)
```

### Django Configuration

If you're using Django and want to capture logs from Document Analyzer, add this to your `settings.py`:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'document_analyzer.log',
        },
    },
    'loggers': {
        'document_analyzer': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    },
}
```

### Flask Configuration

For Flask applications:

```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    handler = RotatingFileHandler('document_analyzer.log', maxBytes=10000000, backupCount=10)
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    
    # Get the document_analyzer logger
    doc_logger = logging.getLogger('document_analyzer')
    doc_logger.addHandler(handler)
    doc_logger.setLevel(logging.DEBUG)
```

## Limitations

Be aware of the following limitations when using Document Analyzer:

- **Cédula Support** — Cédula extraction is specifically designed for Panamanian identity cards in Spanish only. Non-Panamanian identity documents are not supported. Passport extraction works with any standard ICAO MRZ passport regardless of country.

- **Cédula Language** — Panamanian Cédulas are processed in Spanish only. English or other languages are not supported for Cédulas.

- **Image Quality Dependency** — Extraction accuracy depends on image quality. Very poor lighting, severe blur, or damaged documents may produce incomplete or inaccurate results. While the package includes preprocessing to improve poor quality images, there are limits to what can be recovered.

- **PDF Support Not Tested** — PDFs are not officially supported and have not been tested. The package is designed for and tested with image files (JPG, PNG, etc.).

- **Passport MRZ Dependency** — Passport extraction relies primarily on the Machine Readable Zone (MRZ) at the bottom of the document page. If the MRZ is obscured, cut off, or damaged in the photo, extraction accuracy will be significantly affected. Ensure the entire document including the bottom strip is clearly visible in the frame.

- **Place of Birth for Non-Panamanian Passports** — Place of birth is the only passport field extracted from the document's written fields rather than the MRZ. This works reliably for Panamanian passports. For other countries it may be inaccurate or missing depending on how that country formats and labels the biographical page of their passport.

- **CPU Testing Only** — The package has only been tested on CPU hardware (Intel i5, 10th generation). GPU acceleration via CUDA may work but is not officially supported or validated.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Author

- **Name:** Usman Ghani
- **GitHub:** [usman-369](https://github.com/usman-369)
- ![Built with AI](https://img.shields.io/badge/Built%20with-AI-black?style=for-the-badge&logo=githubcopilot)
