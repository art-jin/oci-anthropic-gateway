"""Logging configuration for OCI-Anthropic Gateway."""

import logging
import os

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(levelname)s:%(name)s:%(message)s'
)

# Get gateway logger
logger = logging.getLogger("oci-gateway")
logger.setLevel(LOG_LEVEL)

# Suppress OCI SDK verbose logging unless DEBUG is enabled
if LOG_LEVEL != "DEBUG":
    logging.getLogger("oci").setLevel(logging.WARNING)
    logging.getLogger("oci.base_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str = "oci-gateway") -> logging.Logger:
    """Get configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """Set log level for gateway logger.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    logger.setLevel(level.upper())
    if level.upper() == "DEBUG":
        logging.getLogger("oci").setLevel(logging.INFO)
        logging.getLogger("oci.base_client").setLevel(logging.INFO)
    else:
        logging.getLogger("oci").setLevel(logging.WARNING)
        logging.getLogger("oci.base_client").setLevel(logging.WARNING)
