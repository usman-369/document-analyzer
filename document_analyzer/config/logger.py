import logging

logger = logging.getLogger("requests")


class DocumentAnalyzerLoggerAdapter(logging.LoggerAdapter):
    """A logging adapter that adds user context to document_analyzer log messages.

    This adapter extends the standard LoggerAdapter to automatically prepend
    log messages with the DocumentAnalyzer prefix and user email information,
    providing better traceability and context for debugging and monitoring
    purposes.

    The adapter formats log messages in the pattern:
        [DocumentAnalyzer] (user_email) original_message

    Args:
        logger: The base logging.Logger instance to wrap.
        extra (dict, optional): Context dictionary. Should contain 'user_email'
            for user identification. Defaults to "unknown" if missing.

    Example:
        >>> extra_info = {"user_email": "user@example.com"}
        >>> adapter = DocumentAnalyzerLoggerAdapter(logger, extra_info)
        >>> adapter.info("Processing document")
        # Output: [DocumentAnalyzer] (user@example.com) Processing document
    """

    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        super().__init__(logger, extra)

    def process(self, msg, kwargs):
        user_email = self.extra.get("user_email", "unknown")
        return f"[DocumentAnalyzer] ({user_email}) {msg}", kwargs
