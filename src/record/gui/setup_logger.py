"""
Common GUI logger instances and default formatter.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger_gui_imu_client = logging.getLogger(
    "gui-imu-client"
)  # -> /workspaces/record/record/gui/imu_client.py
logger_gui_sequence = logging.getLogger(
    "sequence"
)  # -> /workspaces/record/record/gui/sequence.py

LOG_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(message)s]"
)
