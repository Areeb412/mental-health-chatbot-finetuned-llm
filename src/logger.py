import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a named logger.
    
    Args:
        name:  Logger name (shown in log messages, e.g. "app", "inference")
        level: Minimum severity to log (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    
    Returns:
        Configured Logger instance
    
    Usage:
        logger = setup_logger("my_module")
        logger.info("Something happened")      → INFO:my_module: Something happened
        logger.error("Something broke", exc_info=True)  → includes stack trace
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Don't add duplicate handlers if function is called multiple times
    if logger.handlers:
        return logger

    # Create a handler that writes to stdout (visible in Docker logs)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format: [timestamp] LEVEL    module_name: message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger