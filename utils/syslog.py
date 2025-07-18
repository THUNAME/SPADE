import sys
import logging

logger = None


def _get_curr_logger():
    global logger
    if logger is None:
        # create logger
        logger = logging.getLogger("ServicePred")
        logger.setLevel(logging.DEBUG)
        hdr = logging.StreamHandler(sys.stdout)
        hdr.setLevel(logging.DEBUG)
        hdr.setFormatter(
            logging.Formatter(
                "\033[36;1m[%(name)s] [%(asctime)s] [%(process)d] [%(levelname)s]:\033[0m %(message)s"
            )
        )
        logger.addHandler(hdr)
    return logger


def info(msg: str):
    logger = _get_curr_logger()
    logger.info(f"\033[1m{msg}\033[0m")


def warning(msg: str):
    logger = _get_curr_logger()
    logger.warning(f"\033[31;1m{msg}\033[0m")


def debug(msg: str):
    logger = _get_curr_logger()
    logger.debug(f"\033[32;1m{msg}\033[0m")
