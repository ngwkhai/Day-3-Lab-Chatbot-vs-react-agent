import logging
import json
import os
from datetime import datetime
from typing import Any, Dict

class IndustryLogger:
    """
    Structured logger that simulates industry practices.
    Logs to both console and a file in JSON format.
    """
    def __init__(self, name: str = "AI-Lab-Agent", log_dir: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Avoid attaching duplicate handlers if instantiated more than once.
        if self.logger.handlers:
            return

        # Console Handler (always available).
        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        # File Handler (JSON) -- best effort. Serverless platforms such as
        # Vercel expose a read-only filesystem (only /tmp is writable), so we
        # degrade gracefully to console-only logging instead of crashing on import.
        log_dir = log_dir or os.getenv("LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
            file_handler = logging.FileHandler(log_file)
            self.logger.addHandler(file_handler)
        except OSError:
            # Read-only filesystem: keep running with console logging only.
            pass

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Logs an event with a timestamp and type."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data
        }
        self.logger.info(json.dumps(payload))

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str, exc_info=True):
        self.logger.error(msg, exc_info=exc_info)

# Global logger instance
logger = IndustryLogger()
