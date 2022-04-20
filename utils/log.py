import logging
import os
from datetime import datetime

from utils.utility import makedirs


LOGGER_NAME = 'run log'
LOG_FOLDER = 'logs'
makedirs(LOG_FOLDER)
LOG_PATH = os.path.join(LOG_FOLDER, datetime.now().strftime('%Y%m%d%H%M%S') + '.log')
    
logger = logging.getLogger(LOGGER_NAME)
template=str("%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s")
formatter = logging.Formatter(template)
fileHandler = logging.FileHandler(LOG_PATH, mode='a')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

logger.setLevel(logging.INFO)


