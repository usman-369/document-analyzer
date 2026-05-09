"""Command line interface for document_analyzer."""

import sys
import json
import base64
import argparse
from pathlib import Path

from .services import PaddleOCRService
from .analyzers import analyze_document


def save_signature(signature_b64, output_path):
    """Save base64 signature to file."""
    if signature_b64:
        signature_data = base64.b64decode(signature_b64)
        with open(output_path, "wb") as f:
            f.write(signature_data)
        return True
    return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Analyze Cedula or Passport documents")
    parser.add_argument("input_file", help="Path to document image")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument(
        "-s", "--save-signature", help="Save signature to specified path"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--user-email", help="User email for logging")
    parser.add_argument(
        "--type",
        choices=["auto", "cedula", "passport"],
        default="auto",
        help="Document type (default: auto)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize OCR service
        if args.verbose:
            print("Initializing OCR service...")
        ocr = PaddleOCRService.initialize()

        # Analyze document
        if args.verbose:
            print(f"Analyzing document: {input_path}")

        if args.type == "auto":
            result = analyze_document(str(input_path), args.user_email, ocr)
        elif args.type == "cedula":
            from .analyzers import CedulaAnalyzer

            analyzer = CedulaAnalyzer(str(input_path), args.user_email, ocr)
            result = analyzer.analyze_cedula()
            result["document_type"] = "cedula"
        else:  # passport
            from .analyzers import PassportAnalyzer

            analyzer = PassportAnalyzer(str(input_path), args.user_email, ocr)
            result = analyzer.analyze_passport()
            result["document_type"] = "passport"

        # Save signature if requested
        if args.save_signature and result.get("signature"):
            if save_signature(result["signature"], args.save_signature):
                if args.verbose:
                    print(f"Signature saved to: {args.save_signature}")
            else:
                print("Warning: No signature found to save", file=sys.stderr)

        # Prepare output
        output_data = {
            "success": result["success"],
            "document_type": result["document_type"],
            "extracted_info": result.get(f"{result['document_type']}_info", {}),
            "has_signature": result["signature"] is not None,
            "error": result.get("error"),
        }

        # Remove signature from JSON output (too large)
        if "signature" in result:
            del result["signature"]
        if "raw_extracted_data" in result:
            del result["raw_extracted_data"]

        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))

        # Exit with appropriate code
        if result["success"] == "none":
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
