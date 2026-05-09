import os
import sys
from .config import logger


def startup_services():
    try:
        if any(cmd in sys.argv for cmd in ["runserver", "gunicorn", "uwsgi"]):
            run_main = os.environ.get("RUN_MAIN")
            is_dev_reloader = run_main == "true"
            is_manual_runserver = "--noreload" in sys.argv
            is_prod = "gunicorn" in sys.argv or "uwsgi" in sys.argv

            if is_dev_reloader or is_manual_runserver or is_prod:
                from .services import PaddleOCRService

                if PaddleOCRService.is_ready():
                    logger.info("PaddleOCR already initialized, reusing instance.")
                else:
                    PaddleOCRService.initialize()
    except ImportError:
        logger.error("Failed to import PaddleOCRService during app startup")
    except Exception as e:
        logger.error("Failed to initialize PaddleOCR during app startup: %s", e)
