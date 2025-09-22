"""
Lightweight module configuring a few named loggers used by the GUI/server.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger_imu = logging.getLogger("imu")
logger_bt_server = logging.getLogger("bt-server")
logger_bt_runner = logging.getLogger("bt-runner")
