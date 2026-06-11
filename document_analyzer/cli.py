import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from .analyzers import DocumentAnalyzer, CedulaAnalyzer, PassportAnalyzer
from .config import logger as project_logger

__version__ = "0.1.0"

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".pdf"}


class CLIError(Exception):
    """Custom exception for CLI errors."""

    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(self.message)


def validate_input_file(file_path: str) -> Path:
    """
    Validate that input file exists, is readable, and has supported format.

    Args:
        file_path: Path to the input file

    Returns:
        Path object if valid

    Raises:
        CLIError: If file is invalid or unsupported
    """
    try:
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise CLIError(f"Error: File not found: {file_path}", exit_code=1)

        # Check if it's a file (not a directory)
        if not path.is_file():
            raise CLIError(f"Error: Path is not a file: {file_path}", exit_code=1)

        # Check if file is empty
        if path.stat().st_size == 0:
            raise CLIError(f"Error: File is empty: {file_path}", exit_code=1)

        # Check file format
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise CLIError(
                f"Error: Unsupported file format '{path.suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
                exit_code=1,
            )

        # Check if file is readable
        try:
            with open(path, "rb") as f:
                f.read(1)
        except PermissionError:
            raise CLIError(
                f"Error: Permission denied reading file: {file_path}", exit_code=1
            )

        return path

    except CLIError:
        raise
    except Exception as e:
        raise CLIError(f"Error: Failed to validate file: {str(e)}", exit_code=1)


def validate_output_path(output_path: str) -> Path:
    """
    Validate that output path directory is writable.

    Args:
        output_path: Path to the output file

    Returns:
        Path object if valid

    Raises:
        CLIError: If output directory is not writable
    """
    try:
        path = Path(output_path)
        output_dir = path.parent

        # Create parent directories if they don't exist
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise CLIError(
                    f"Error: Permission denied creating directory: {output_dir}",
                    exit_code=1,
                )

        # Check if directory is writable by attempting to write a test file
        test_file = output_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise CLIError(
                f"Error: Output directory is not writable: {output_dir}", exit_code=1
            )

        return path

    except CLIError:
        raise
    except Exception as e:
        raise CLIError(f"Error: Failed to validate output path: {str(e)}", exit_code=1)


def detect_and_analyze(
    file_path: Path, user_email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Auto-detect document type and analyze it.

    Args:
        file_path: Path to the document file
        user_email: Optional user email for logging

    Returns:
        Dictionary with analysis results

    Raises:
        CLIError: If detection or analysis fails
    """
    try:
        project_logger.debug(f"Attempting auto-detection on: {file_path}")

        # Use DocumentAnalyzer for auto-detection
        analyzer = DocumentAnalyzer(str(file_path), user_email=user_email)
        doc_type = analyzer.detect_document_type()

        project_logger.debug(f"Detected document type: {doc_type}")

        if doc_type == "unknown":
            raise CLIError(
                "Error: Could not determine document type. "
                "Please specify --type (cedula or passport).",
                exit_code=1,
            )

        # Now analyze with the appropriate analyzer
        if doc_type == "cedula":
            cedula_analyzer = CedulaAnalyzer(str(file_path), user_email=user_email)
            result = cedula_analyzer.analyze_cedula()
            result["document_type"] = "cedula"
            return result
        elif doc_type == "passport":
            passport_analyzer = PassportAnalyzer(str(file_path), user_email=user_email)
            result = passport_analyzer.analyze_passport()
            result["document_type"] = "passport"
            return result
        else:
            raise CLIError("Error: Unknown document type after detection.", exit_code=2)

    except CLIError:
        raise
    except ValueError as e:
        # Invalid image format or corrupted file
        raise CLIError(f"Error: Invalid or corrupted image file: {str(e)}", exit_code=2)
    except Exception as e:
        project_logger.error(f"Auto-detection failed: {str(e)}", exc_info=True)
        raise CLIError(f"Error: Failed to analyze document: {str(e)}", exit_code=2)


def analyze_cedula(file_path: Path, user_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a cédula document.

    Args:
        file_path: Path to the cédula image
        user_email: Optional user email for logging

    Returns:
        Dictionary with analysis results

    Raises:
        CLIError: If analysis fails
    """
    try:
        project_logger.debug(f"Analyzing cédula: {file_path}")
        analyzer = CedulaAnalyzer(str(file_path), user_email=user_email)
        result = analyzer.analyze_cedula()
        result["document_type"] = "cedula"
        return result

    except ValueError as e:
        raise CLIError(f"Error: Invalid or corrupted image file: {str(e)}", exit_code=2)
    except Exception as e:
        project_logger.error(f"Cédula analysis failed: {str(e)}", exc_info=True)
        raise CLIError(f"Error: Failed to analyze cédula: {str(e)}", exit_code=2)


def analyze_passport(
    file_path: Path, user_email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a passport document.

    Args:
        file_path: Path to the passport image
        user_email: Optional user email for logging

    Returns:
        Dictionary with analysis results

    Raises:
        CLIError: If analysis fails
    """
    try:
        project_logger.debug(f"Analyzing passport: {file_path}")
        analyzer = PassportAnalyzer(str(file_path), user_email=user_email)
        result = analyzer.analyze_passport()
        result["document_type"] = "passport"
        return result

    except ValueError as e:
        raise CLIError(f"Error: Invalid or corrupted image file: {str(e)}", exit_code=2)
    except Exception as e:
        project_logger.error(f"Passport analysis failed: {str(e)}", exc_info=True)
        raise CLIError(f"Error: Failed to analyze passport: {str(e)}", exit_code=2)


def format_result_json(result: Dict[str, Any]) -> str:
    """Format analysis result as JSON string."""
    return json.dumps(result, indent=4, default=str)


def setup_logging(verbose: bool) -> None:
    """Configure logging for the entire package.

    With -v: Show all DEBUG logs from the document_analyzer package.
    Without -v: Show only WARNING and ERROR logs from the document_analyzer package.

    This is scoped to document_analyzer only to avoid capturing logs from
    other libraries.
    """
    level = logging.DEBUG if verbose else logging.WARNING

    # Get the package logger (scoped to document_analyzer)
    package_logger = logging.getLogger("document_analyzer")

    # Prevent duplicate handlers if setup_logging is called multiple times
    if not package_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        package_logger.addHandler(handler)
    else:
        # Update existing handlers to use the new level
        for handler in package_logger.handlers:
            handler.setLevel(level)

    package_logger.setLevel(level)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="document-analyzer",
        description="Analyze Panamanian identity document (cédula) and passports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze doc.jpg
  %(prog)s analyze cedula.jpg --type cedula
  %(prog)s analyze doc.jpg --save result.json -v
        """,
    )

    # Global options
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze command
    analyze_cmd = subparsers.add_parser("analyze", help="Analyze a document")

    analyze_cmd.add_argument("path", help="Path to the document image file")

    analyze_cmd.add_argument(
        "--type",
        choices=["auto", "cedula", "passport"],
        default="auto",
        help="Document type (default: auto-detect)",
    )

    analyze_cmd.add_argument(
        "--save",
        metavar="FILE",
        help="Save result to file instead of printing to stdout",
    )

    analyze_cmd.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging"
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for user error, 2 for processing error)
    """
    parser = create_parser()

    try:
        args = parser.parse_args(argv)

        # Handle no command
        if not args.command:
            parser.print_help()
            return 0

        # Setup logging
        setup_logging(args.verbose)

        # Validate input file
        project_logger.debug(f"Validating input file: {args.path}")
        input_path = validate_input_file(args.path)

        # Analyze document based on type
        if args.type == "auto":
            project_logger.debug("Using auto-detection for document type")
            result = detect_and_analyze(input_path)
        elif args.type == "cedula":
            result = analyze_cedula(input_path)
        elif args.type == "passport":
            result = analyze_passport(input_path)
        else:
            raise CLIError(f"Unknown document type: {args.type}", exit_code=1)

        # Format output as JSON
        output_json = format_result_json(result)

        # Handle output destination
        if args.save:
            project_logger.debug(f"Validating output path: {args.save}")
            output_path = validate_output_path(args.save)

            try:
                with open(output_path, "w") as f:
                    f.write(output_json)
                print(f"Result saved to: {output_path}")
                project_logger.debug(f"Result saved to: {output_path}")
            except PermissionError:
                raise CLIError(
                    f"Error: Permission denied writing to: {args.save}", exit_code=1
                )
            except Exception as e:
                raise CLIError(
                    f"Error: Failed to write output file: {str(e)}", exit_code=1
                )
        else:
            # Print to stdout
            print(output_json)

        return 0

    except CLIError as e:
        print(e.message, file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 1
    except Exception as e:
        project_logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: An unexpected error occurred: {str(e)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
